"""Resumable decision-model evaluation for question-only and rationale-augmented conditions."""

from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import ExperimentConfig
from .decision_client import DecisionClient
from .models import ArcRecord, DecisionResult
from .prompts import DecisionCondition, build_decision_state
from .storage import RunStore, sanitized_error, stable_hash, write_csv_atomic


CONDITIONS: tuple[DecisionCondition, ...] = (
    "question_only",
    "normal_reasoning",
    "poetic_reasoning",
)
DECISION_FIELDS = (
    "run_id",
    "example_id",
    "dataset_config",
    "split",
    "source_row_index",
    "condition",
    "decision_model",
    "decision_model_checkpoint",
    "prompt_hash",
    "state_hash",
    "predicted_answer_label",
    "predicted_answer_text",
    "gold_answer_label",
    "gold_answer_text",
    "is_correct",
    "confidence",
    "choice_probabilities_json",
    "request_status",
    "attempt_count",
    "latency_ms",
    "decision_input_tokens",
    "decision_output_tokens",
    "decision_total_tokens",
    "decision_input_token_estimate",
    "decision_input_state_json",
    "decision_raw_response_json",
    "latest_attempt_id",
    "request_id",
    "error_type",
    "error_message",
)


def _read_csv_by_example(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing generation output {path}. Run the generate stage first.")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["example_id"]: row for row in rows if row.get("example_id")}


def _eligible_rationale(
    row: Mapping[str, str] | None,
    *,
    field_name: str,
    include_flagged: bool,
) -> tuple[str | None, str | None]:
    if row is None:
        return None, "missing_generation_row"
    rationale = row.get(field_name) or ""
    if not rationale.strip():
        return None, "missing_rationale"
    if row.get("generation_status") != "success" and not include_flagged:
        return None, f"generation_status_{row.get('generation_status') or 'unknown'}"
    if row.get("leakage_status") == "hard_flagged" and not include_flagged:
        return None, "hard_flagged_rationale"
    return rationale, None


def _cache_key(
    record: ArcRecord,
    *,
    condition: DecisionCondition,
    config: ExperimentConfig,
    state_hash: str,
) -> str:
    return stable_hash(
        {
            "stage": "decision",
            "example_id": record.example_id,
            "condition": condition,
            "decision_model": config.decision_model,
            "laya_finetuned_checkpoint": config.laya_finetuned_checkpoint,
            "state_hash": state_hash,
        }
    )


def _base_projection(
    record: ArcRecord,
    *,
    condition: DecisionCondition,
    config: ExperimentConfig,
    state_hash: str | None,
    state: dict[str, Any] | None,
    attempt_count: int,
    attempt_id: str,
    latency_ms: int | None,
) -> dict[str, Any]:
    return {
        "run_id": config.output_dir.name,
        "example_id": record.example_id,
        "dataset_config": record.dataset_config,
        "split": record.split,
        "source_row_index": record.source_row_index,
        "condition": condition,
        "decision_model": config.decision_model,
        "decision_model_checkpoint": config.laya_finetuned_checkpoint,
        "prompt_hash": stable_hash("Select the answer label best supported by the supplied question data."),
        "state_hash": state_hash,
        "predicted_answer_label": None,
        "predicted_answer_text": None,
        "gold_answer_label": record.gold_answer_label,
        "gold_answer_text": record.gold_answer_text,
        "is_correct": None,
        "confidence": None,
        "choice_probabilities_json": {},
        "request_status": "failed",
        "attempt_count": attempt_count,
        "latency_ms": latency_ms,
        "decision_input_tokens": None,
        "decision_output_tokens": None,
        "decision_total_tokens": None,
        "decision_input_token_estimate": None,
        "decision_input_state_json": state,
        "decision_raw_response_json": None,
        "latest_attempt_id": attempt_id,
        "request_id": None,
        "error_type": None,
        "error_message": None,
    }


def _result_projection(
    record: ArcRecord,
    *,
    condition: DecisionCondition,
    config: ExperimentConfig,
    state_hash: str,
    state: dict[str, Any],
    attempt_count: int,
    attempt_id: str,
    latency_ms: int,
    result: DecisionResult,
) -> dict[str, Any]:
    if result.predicted_answer_label not in record.choice_map:
        raise ValueError("Decision client returned a label outside the record's choices.")
    projection = _base_projection(
        record,
        condition=condition,
        config=config,
        state_hash=state_hash,
        state=state,
        attempt_count=attempt_count,
        attempt_id=attempt_id,
        latency_ms=latency_ms,
    )
    projection.update(
        {
            "decision_model": result.model or config.decision_model,
            "predicted_answer_label": result.predicted_answer_label,
            "predicted_answer_text": record.choice_map[result.predicted_answer_label],
            "is_correct": result.predicted_answer_label == record.gold_answer_label,
            "confidence": result.confidence,
            "choice_probabilities_json": result.probabilities,
            "request_status": "success",
            "decision_input_tokens": result.usage.input_tokens,
            "decision_output_tokens": result.usage.output_tokens,
            "decision_total_tokens": result.usage.total_tokens,
            "decision_raw_response_json": result.raw_response,
            "request_id": result.request_id,
        }
    )
    return projection


def _write_decision_projection(store: RunStore) -> None:
    condition_order = {condition: index for index, condition in enumerate(CONDITIONS)}
    rows = sorted(
        store.latest_projections("decision"),
        key=lambda row: (int(row["source_row_index"]), condition_order.get(row.get("condition"), 99)),
    )
    write_csv_atomic(store.root / "decision_predictions.csv", rows, DECISION_FIELDS)


def _append_skipped(
    store: RunStore,
    *,
    record: ArcRecord,
    condition: DecisionCondition,
    config: ExperimentConfig,
    reason: str,
) -> None:
    attempt_id = str(uuid.uuid4())
    unavailable_state_hash = _unavailable_state_hash(record, condition=condition, reason=reason)
    cache_key = _cache_key(
        record,
        condition=condition,
        config=config,
        state_hash=unavailable_state_hash,
    )
    projection = _base_projection(
        record,
        condition=condition,
        config=config,
        state_hash=unavailable_state_hash,
        state=None,
        attempt_count=0,
        attempt_id=attempt_id,
        latency_ms=None,
    )
    projection.update({"request_status": "skipped", "error_type": "RationaleUnavailable", "error_message": reason})
    store.append_attempt(
        "decision",
        {
            "attempt_id": attempt_id,
            "run_id": config.output_dir.name,
            "example_id": record.example_id,
            "condition": condition,
            "cache_key": cache_key,
            "attempt_number": 0,
            "terminal": True,
            "status": "skipped",
            "projection": projection,
        },
    )


def _unavailable_state_hash(record: ArcRecord, *, condition: DecisionCondition, reason: str) -> str:
    return stable_hash({"example_id": record.example_id, "condition": condition, "unavailable_reason": reason})


def evaluate_decisions(
    records: Iterable[ArcRecord],
    *,
    config: ExperimentConfig,
    store: RunStore,
    client: DecisionClient,
) -> None:
    """Run all independent conditions and retain every response or observable skip."""
    store.initialize(config)
    existing = store.completed_keys("decision")
    if existing and not config.resume and not config.force_regenerate:
        raise FileExistsError(
            "Decision attempt ledger already has completed records. Use --resume or --force-regenerate explicitly."
        )
    normal_rows = _read_csv_by_example(store.root / "front_cot.csv")
    poetic_rows = _read_csv_by_example(store.root / "front_poetic.csv")

    for record in records:
        for condition in CONDITIONS:
            if condition == "question_only":
                rationale, unavailable_reason = None, None
            elif condition == "normal_reasoning":
                rationale, unavailable_reason = _eligible_rationale(
                    normal_rows.get(record.example_id),
                    field_name="cot_reasoning",
                    include_flagged=config.include_flagged_reasoning,
                )
            else:
                rationale, unavailable_reason = _eligible_rationale(
                    poetic_rows.get(record.example_id),
                    field_name="poetic_cot_reasoning",
                    include_flagged=config.include_flagged_reasoning,
                )
            if unavailable_reason is not None:
                unavailable_key = _cache_key(
                    record,
                    condition=condition,
                    config=config,
                    state_hash=_unavailable_state_hash(record, condition=condition, reason=unavailable_reason),
                )
                if unavailable_key in existing and config.resume and not config.force_regenerate:
                    continue
                _append_skipped(
                    store,
                    record=record,
                    condition=condition,
                    config=config,
                    reason=unavailable_reason,
                )
                continue

            state = build_decision_state(record, condition=condition, rationale=rationale)
            state_hash = stable_hash(state)
            cache_key = _cache_key(record, condition=condition, config=config, state_hash=state_hash)
            if cache_key in existing and config.resume and not config.force_regenerate:
                continue
            for attempt_number in range(1, config.max_retries + 2):
                attempt_id = str(uuid.uuid4())
                started = time.monotonic()
                try:
                    result = client.decide(state, record)
                    latency_ms = int((time.monotonic() - started) * 1000)
                    projection = _result_projection(
                        record,
                        condition=condition,
                        config=config,
                        state_hash=state_hash,
                        state=state,
                        attempt_count=attempt_number,
                        attempt_id=attempt_id,
                        latency_ms=latency_ms,
                        result=result,
                    )
                    store.append_attempt(
                        "decision",
                        {
                            "attempt_id": attempt_id,
                            "run_id": config.output_dir.name,
                            "example_id": record.example_id,
                            "condition": condition,
                            "cache_key": cache_key,
                            "attempt_number": attempt_number,
                            "terminal": True,
                            "status": "success",
                            "latency_ms": latency_ms,
                            "state": state,
                            "state_hash": state_hash,
                            "result": {
                                "predicted_answer_label": result.predicted_answer_label,
                                "confidence": result.confidence,
                                "probabilities": result.probabilities,
                                "model": result.model,
                                "usage": result.usage.to_dict(),
                                "request_id": result.request_id,
                                "raw_response": result.raw_response,
                            },
                            "projection": projection,
                        },
                    )
                    break
                except Exception as error:  # Retain transport and parsing failures for later analysis.
                    latency_ms = int((time.monotonic() - started) * 1000)
                    terminal = attempt_number == config.max_retries + 1
                    projection = _base_projection(
                        record,
                        condition=condition,
                        config=config,
                        state_hash=state_hash,
                        state=state,
                        attempt_count=attempt_number,
                        attempt_id=attempt_id,
                        latency_ms=latency_ms,
                    )
                    projection.update(
                        {
                            "error_type": type(error).__name__,
                            "error_message": sanitized_error(error),
                        }
                    )
                    store.append_attempt(
                        "decision",
                        {
                            "attempt_id": attempt_id,
                            "run_id": config.output_dir.name,
                            "example_id": record.example_id,
                            "condition": condition,
                            "cache_key": cache_key,
                            "attempt_number": attempt_number,
                            "terminal": terminal,
                            "status": "failed",
                            "latency_ms": latency_ms,
                            "state": state,
                            "state_hash": state_hash,
                            "error_type": type(error).__name__,
                            "error_message": sanitized_error(error),
                            "projection": projection if terminal else None,
                        },
                    )
                    if terminal:
                        break
                if attempt_number <= config.max_retries:
                    time.sleep(min(2 ** (attempt_number - 1), 30))
    _write_decision_projection(store)


def rebuild_decision_csv(store: RunStore) -> None:
    """Regenerate decision_predictions.csv from saved attempt records only."""
    _write_decision_projection(store)
