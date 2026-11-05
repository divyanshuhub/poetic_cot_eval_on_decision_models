"""Evaluate local fine-tuned decision models on a vLLM CoT CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from .config import DECISION_MODELS, ExperimentConfig, canonical_dataset_config
from .dataset import DATASET_ID, normalize_arc_row
# from .decision_client import DecisionClient, create_decision_client, local_systemone_config
from .decision_client import (
    DecisionClient,
    create_decision_client,
    jev_systemone_config,
    local_systemone_config,
)
from .evaluation import DECISION_FIELDS
from .metrics import as_bool, as_number, numeric_summary, paired_accuracy_difference
from .models import ArcRecord, Choice, DecisionResult
from .prompts import build_choice_question, public_question_payload
from .storage import (
    RunStore,
    read_json,
    sanitized_error,
    stable_hash,
    write_csv_atomic,
    write_json_atomic,
)
from .vllm_cot import DATASET_CSV_SCHEMAS, split_standard_cot


COT_CONDITIONS = (
    "question_only",
    "standard_reasoning",
    "poisoned_reasoning",
    "poetic_reasoning",
)
RATIONALE_COLUMNS = {
    "standard_reasoning": "standard_cot",
    "poisoned_reasoning": "poisoned_cot",
    "poetic_reasoning": "poetic_cot",
}
STATE_FIELDS = {
    "standard_reasoning": "standard_rationale",
    "poisoned_reasoning": "poisoned_rationale",
    "poetic_reasoning": "poetic_rationale",
}
REQUIRED_CSV_COLUMNS = ("question", "choices", "answerKey", *RATIONALE_COLUMNS.values())
SERVER_BASE_MODELS = {"kev": "jaredpalmer/kev-4b", "clef": "Cloudflare/clef-flash"}
DATASET_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "arc": ("question", "choices", "answerKey", *RATIONALE_COLUMNS.values()),
    "typed_decisions": ("state", "question", "choices", "answerKey", *RATIONALE_COLUMNS.values()),
    "proofwriter": ("theory", "question", "answer", *RATIONALE_COLUMNS.values()),
    "logiqa": ("context", "query", "options", "correct_option", *RATIONALE_COLUMNS.values()),
}


def required_csv_columns_for_dataset(dataset_name: str) -> tuple[str, ...]:
    norm = dataset_name.lower().replace("-", "_")
    return DATASET_REQUIRED_COLUMNS.get(norm, DATASET_CSV_SCHEMAS.get(norm, REQUIRED_CSV_COLUMNS))


def detect_dataset_from_csv(path: Path) -> tuple[str, str, str]:
    """Auto-detect (dataset_name, dataset_type, split) from filename and/or CSV headers."""
    filename = path.name.lower()
    dataset_name: str | None = None
    dataset_type: str | None = None
    split: str | None = None

    if "test" in filename:
        split = "test"
    elif "validation" in filename or "valid" in filename or "val" in filename:
        split = "validation"
    elif "train" in filename:
        split = "train"

    if "logiqa" in filename:
        dataset_name = "logiqa"
        dataset_type = "default"
    elif "proofwriter" in filename:
        dataset_name = "proofwriter"
        dataset_type = "default"
    elif "typed_decisions" in filename or "typeddecisions" in filename or "typed-decisions" in filename:
        dataset_name = "typed_decisions"
        dataset_type = "all"
    elif "arc" in filename:
        dataset_name = "arc"
        dataset_type = "challenge" if "challenge" in filename else "easy"

    if dataset_name is None and path.is_file():
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                headers = {h.strip().lower() for h in next(reader, [])}
                if "query" in headers and "options" in headers:
                    dataset_name = "logiqa"
                    dataset_type = "default"
                elif "theory" in headers:
                    dataset_name = "proofwriter"
                    dataset_type = "default"
                elif "state" in headers and "choices" in headers:
                    dataset_name = "typed_decisions"
                    dataset_type = "all"
                elif "question" in headers and "choices" in headers:
                    dataset_name = "arc"
                    dataset_type = "easy"
        except Exception:
            pass

    resolved_dataset = dataset_name or "arc"
    resolved_type = dataset_type or (
        "easy" if resolved_dataset == "arc"
        else ("all" if resolved_dataset in ("typed_decisions", "typed-decisions") else "default")
    )
    resolved_split = split or "test"
    return resolved_dataset, resolved_type, resolved_split
SUMMARY_FIELDS = (
    "condition",
    "decision_model",
    "decision_model_checkpoint",
    "eligible_examples",
    "completed_predictions",
    "coverage",
    "accuracy",
    "shared_completion_examples",
    "shared_completion_accuracy",
    "question_only_difference_pp",
    "question_only_difference_ci_low_pp",
    "question_only_difference_ci_high_pp",
    "cot_tokens_reported_examples",
    "cot_tokens_total",
    "cot_tokens_mean",
    "cot_tokens_std",
    "cot_tokens_delta_percent_vs_question_only",
)
TOKEN_LATENCY_FIELDS = (
    "stage",
    "variant",
    "latency_count",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "input_token_count",
    "input_token_mean",
    "output_token_count",
    "output_token_mean",
)
ERROR_FIELDS = (
    "stage",
    "attempt_id",
    "example_id",
    "condition",
    "status",
    "error_type",
    "error_message",
    "timestamp",
)


@dataclass(frozen=True)
class CotCsvInput:
    """Normalized ARC records and their corresponding CSV rationale cells."""

    path: Path
    sha256: str
    records: tuple[ArcRecord, ...]
    rationales: dict[str, dict[str, str]]
    unavailable: dict[str, dict[str, str]]
    raw_cots: dict[str, dict[str, str]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_output_dir(value: str | Path) -> Path:
    """Resolve the run root and keep it inside the experiment project."""
    root = _project_root().resolve()
    requested = Path(value).expanduser()
    output_dir = (requested if requested.is_absolute() else root / requested).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Output directory must be inside the project folder: {root}.") from error
    if output_dir == root:
        raise ValueError("Output directory must be a subdirectory of the project folder.")
    return output_dir


def default_output_dir(input_csv: str | Path, decision_model: str) -> Path:
    """Create a per-input, per-model output directory by default."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(input_csv).stem).strip("_")
    suffix = decision_model if decision_model.lower() == "jev" else f"{decision_model}_finetuned"
    return Path("outputs") / "cot_evaluations" / f"{stem}_{suffix}"


def resolve_finetuned_model_identity(
    decision_model: str,
    *,
    laya_finetuned_checkpoint: str | Path | None,
    kev_finetuned_checkpoint: str | Path | None = None,
    clef_finetuned_checkpoint: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_type: str | None = None,
) -> tuple[str, str | None]:
    """Resolve and require a fine-tuned checkpoint identity for the selected backend."""
    normalized = decision_model.lower()
    if normalized == "jev":
        if laya_finetuned_checkpoint is not None:
            raise ValueError(
                "--laya-finetuned-checkpoint cannot be used with JEV."
            )
        api_config = jev_systemone_config(timeout=60.0)
        return api_config.model_id, None

    finetunes_root = _project_root() / "outputs" / "finetunes"
    available_ckpts = [
        str(p.relative_to(_project_root()))
        for p in finetunes_root.glob("*/checkpoint")
        if p.is_dir()
    ]
    available_hint = f" Available in outputs/finetunes: {', '.join(sorted(available_ckpts))}." if available_ckpts else ""

    if normalized == "laya":
        if laya_finetuned_checkpoint is None and dataset_name:
            norm_ds = dataset_name.lower().replace("-", "_")
            cfg_slug = (dataset_type or ("easy" if norm_ds == "arc" else "all" if norm_ds in ("typed_decisions", "typed-decisions") else "default")).lower().replace("-", "_")
            candidate_name = f"laya_arc_{cfg_slug}" if norm_ds == "arc" else f"laya_{norm_ds}_{cfg_slug}"
            candidate = finetunes_root / candidate_name / "checkpoint"
            if candidate.is_dir():
                laya_finetuned_checkpoint = candidate

        if laya_finetuned_checkpoint is None:
            raise ValueError(f"Laya evaluation requires --laya-finetuned-checkpoint.{available_hint}")
        checkpoint = Path(laya_finetuned_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = _project_root() / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Laya fine-tuned checkpoint directory not found: {checkpoint}.{available_hint}")
        return str(checkpoint), str(checkpoint)

    if normalized not in SERVER_BASE_MODELS:
        raise ValueError(f"Unsupported decision model {decision_model!r}.")
    env_name = f"{normalized.upper()}_MODEL_ID"
    if normalized == "kev" and kev_finetuned_checkpoint:
        checkpoint = Path(kev_finetuned_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = _project_root() / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Kev fine-tuned checkpoint directory not found: {checkpoint}.{available_hint}")
        os.environ[env_name] = str(checkpoint)
    elif normalized == "clef" and clef_finetuned_checkpoint:
        checkpoint = Path(clef_finetuned_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = _project_root() / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Clef fine-tuned checkpoint directory not found: {checkpoint}.{available_hint}")
        os.environ[env_name] = str(checkpoint)
    else:
        current_env_val = os.environ.get(env_name, "").strip()
        is_invalid_local_dir = (
            current_env_val.startswith(("/", "./", "../", "outputs/"))
            and not Path(current_env_val).expanduser().is_dir()
        )
        if (not current_env_val or is_invalid_local_dir) and dataset_name:
            norm_ds = dataset_name.lower().replace("-", "_")
            cfg_slug = (dataset_type or ("easy" if norm_ds == "arc" else "all" if norm_ds in ("typed_decisions", "typed-decisions") else "default")).lower().replace("-", "_")
            candidate_name = f"{normalized}_arc_{cfg_slug}" if norm_ds == "arc" else f"{normalized}_{norm_ds}_{cfg_slug}"
            candidate = finetunes_root / candidate_name / "checkpoint"
            if candidate.is_dir():
                os.environ[env_name] = str(candidate)

    if not os.environ.get(env_name, "").strip():
        raise ValueError(
            f"Set {env_name} in arc_decision_reasoning/.env or pass --{normalized}-finetuned-checkpoint "
            "to the fine-tuned model/checkpoint served locally. Use the same checkpoint path in the server and evaluator; "
            f"the example default points to the released base model.{available_hint}"
        )
    server_config = local_systemone_config(normalized, timeout=60.0)
    if server_config.model_id == SERVER_BASE_MODELS[normalized]:
        raise ValueError(f"{env_name} still selects the released base model, not a fine-tuned checkpoint.{available_hint}")
    return server_config.model_id, None


def load_cot_csv(
    input_csv: str | Path,
    *,
    dataset_type: str,
    split: str,
    limit: int | None = None,
    dataset_name: str = "arc",
) -> CotCsvInput:
    """Read the CoT CSV and normalize its fields for the selected dataset."""
    path = Path(input_csv).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CoT input CSV not found: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when provided.")
    norm_dataset = dataset_name.lower().replace("-", "_")
    canonical_config = canonical_dataset_config(dataset_type) if norm_dataset == "arc" else dataset_type
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    records: list[ArcRecord] = []
    rationales: dict[str, dict[str, str]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    raw_cots: dict[str, dict[str, str]] = {}
    expected_cols = required_csv_columns_for_dataset(norm_dataset)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(expected_cols) - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"CoT input CSV is missing required columns: {', '.join(missing_columns)}.")
        for source_row_index, row in enumerate(reader):
            if limit is not None and len(records) >= limit:
                break
            if norm_dataset == "logiqa":
                options_raw = row.get("options") or "[]"
                try:
                    options_list = json.loads(options_raw) if isinstance(options_raw, str) else options_raw
                except Exception as error:
                    raise ValueError(f"Invalid options JSON on CSV row {source_row_index + 2}.") from error
                choices = tuple(Choice(str(i), str(opt).strip()) for i, opt in enumerate(options_list))
                choice_map = {c.label: c.text for c in choices}
                correct_idx = str(row.get("correct_option") if row.get("correct_option") is not None else "").strip()
                if correct_idx not in choice_map:
                    raise ValueError(f"correct_option {correct_idx!r} not in options on CSV row {source_row_index + 2}.")
                record = ArcRecord(
                    example_id=f"logiqa_{canonical_config}__{split}__{source_row_index:06d}",
                    dataset_id="lucasmccabe/logiqa",
                    dataset_config=canonical_config,
                    split=split,
                    source_row_index=source_row_index,
                    question=str(row.get("query") or "").strip(),
                    choices=choices,
                    gold_answer_label=correct_idx,
                    gold_answer_text=choice_map[correct_idx],
                    metadata={"context": str(row.get("context") or "").strip()},
                )
            elif norm_dataset == "proofwriter":
                pw_choices = (Choice("True", "True"), Choice("False", "False"), Choice("Unknown", "Unknown"))
                choice_map = {c.label: c.text for c in pw_choices}
                ans_raw = str(row.get("answer") if row.get("answer") is not None else "").strip()
                ans_key = {"true": "True", "false": "False", "unknown": "Unknown"}.get(ans_raw.lower(), ans_raw)
                if ans_key not in choice_map:
                    raise ValueError(f"ProofWriter answer {ans_raw!r} not in {list(choice_map)} on row {source_row_index + 2}.")
                record = ArcRecord(
                    example_id=f"proofwriter_{canonical_config}__{split}__{source_row_index:06d}",
                    dataset_id="tasksource/proofwriter",
                    dataset_config=canonical_config,
                    split=split,
                    source_row_index=source_row_index,
                    question=str(row.get("question") or "").strip(),
                    choices=pw_choices,
                    gold_answer_label=ans_key,
                    gold_answer_text=choice_map[ans_key],
                    metadata={"theory": str(row.get("theory") or "").strip(), "allProofs": str(row.get("allProofs") or "").strip()},
                )
            elif norm_dataset in ("typed_decisions", "typed-decisions"):
                raw_choices = row.get("choices") or ""
                try:
                    choices_list = json.loads(raw_choices) if isinstance(raw_choices, str) else raw_choices
                except Exception as error:
                    raise ValueError(f"Invalid choices JSON on CSV row {source_row_index + 2}.") from error
                choices = tuple(Choice(str(item["label"]), str(item["text"])) for item in choices_list)
                choice_map = {c.label: c.text for c in choices}
                ans_key = str(row.get("answerKey") if row.get("answerKey") is not None else "").strip()
                if ans_key not in choice_map:
                    raise ValueError(f"answerKey {ans_key!r} not in choices on row {source_row_index + 2}.")
                raw_state = row.get("state") or ""
                try:
                    state_obj = json.loads(raw_state) if isinstance(raw_state, str) and raw_state.startswith("{") else raw_state
                except Exception:
                    state_obj = raw_state
                record = ArcRecord(
                    example_id=f"td_{canonical_config}__{split}__{source_row_index:06d}",
                    dataset_id="LocalLLaMA/typed-decisions",
                    dataset_config=canonical_config,
                    split=split,
                    source_row_index=source_row_index,
                    question=str(row.get("question") or "").strip(),
                    choices=choices,
                    gold_answer_label=ans_key,
                    gold_answer_text=choice_map[ans_key],
                    metadata={"state": state_obj, "question_name": str(row.get("question") or "").strip()},
                )
            else:
                raw_choices = row.get("choices") or ""
                try:
                    choices_list = json.loads(raw_choices) if isinstance(raw_choices, str) else raw_choices
                except Exception as error:
                    raise ValueError(f"Invalid choices JSON on CSV row {source_row_index + 2}.") from error
                record = normalize_arc_row(
                    {
                        "question": row.get("question"),
                        "choices": choices_list,
                        "answerKey": row.get("answerKey"),
                    },
                    dataset_config=canonical_config,
                    split=split,
                    source_row_index=source_row_index,
                )
            records.append(record)
            raw = {column: (row.get(column) or "").strip() for column in RATIONALE_COLUMNS.values()}
            raw_cots[record.example_id] = raw
            rationale_values: dict[str, str] = {}
            unavailable_values: dict[str, str] = {}

            standard = raw["standard_cot"]
            if not standard:
                unavailable_values["standard_reasoning"] = "missing_standard_cot"
            else:
                try:
                    _, clean_standard = split_standard_cot(standard, record)
                    rationale_values["standard_reasoning"] = clean_standard
                except ValueError as error:
                    unavailable_values["standard_reasoning"] = f"invalid_standard_cot: {error}"

            for condition, column in (
                ("poisoned_reasoning", "poisoned_cot"),
                ("poetic_reasoning", "poetic_cot"),
            ):
                rationale = raw[column]
                if rationale:
                    rationale_values[condition] = rationale
                else:
                    unavailable_values[condition] = f"missing_{column}"

            rationales[record.example_id] = rationale_values
            unavailable[record.example_id] = unavailable_values

    if not records:
        raise ValueError(f"No records found in {path}.")
    return CotCsvInput(
        path=path,
        sha256=digest,
        records=tuple(records),
        rationales=rationales,
        unavailable=unavailable,
        raw_cots=raw_cots,
    )


def build_cot_decision_state(
    record: ArcRecord,
    *,
    condition: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Build a bounded decision state without answer keys or standard final-answer text."""
    state = public_question_payload(record)
    if record.metadata:
        if "context" in record.metadata:
            state["context"] = record.metadata["context"]
        if "theory" in record.metadata:
            state["theory"] = record.metadata["theory"]
        if "state" in record.metadata:
            st = record.metadata["state"]
            if isinstance(st, dict):
                state.update(st)
            else:
                state["state"] = st
    if condition == "question_only":
        if rationale is not None:
            raise ValueError("question_only must not receive a rationale.")
        return state
    field = STATE_FIELDS.get(condition)
    if field is None:
        raise ValueError(f"Unsupported CoT condition {condition!r}.")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{condition} requires non-empty rationale text.")
    state[field] = rationale
    return state


def _cache_key(
    record: ArcRecord,
    *,
    condition: str,
    state_hash: str,
    source_csv_sha256: str,
    model_identity: str,
) -> str:
    return stable_hash(
        {
            "stage": "decision",
            "example_id": record.example_id,
            "condition": condition,
            "state_hash": state_hash,
            "source_csv_sha256": source_csv_sha256,
            "decision_model": model_identity,
        }
    )


def _base_projection(
    record: ArcRecord,
    *,
    condition: str,
    config: ExperimentConfig,
    model_identity: str,
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
        "decision_model_checkpoint": model_identity,
        "prompt_hash": stable_hash(build_choice_question(record)["instructions"]),
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
    condition: str,
    config: ExperimentConfig,
    model_identity: str,
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
        model_identity=model_identity,
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


def _append_skipped(
    store: RunStore,
    *,
    record: ArcRecord,
    condition: str,
    config: ExperimentConfig,
    model_identity: str,
    source_csv_sha256: str,
    reason: str,
) -> str:
    attempt_id = str(uuid.uuid4())
    state_hash = stable_hash({"example_id": record.example_id, "condition": condition, "unavailable": reason})
    cache_key = _cache_key(
        record,
        condition=condition,
        state_hash=state_hash,
        source_csv_sha256=source_csv_sha256,
        model_identity=model_identity,
    )
    projection = _base_projection(
        record,
        condition=condition,
        config=config,
        model_identity=model_identity,
        state_hash=state_hash,
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
    return cache_key


def _evaluate_cot_decisions_sequential(
    source: CotCsvInput,
    *,
    config: ExperimentConfig,
    store: RunStore,
    model_identity: str,
    client: DecisionClient,
    _quiet: bool = False,
) -> None:
    """Evaluate question-only and each supplied CoT as independent conditions."""
    existing = store.completed_keys("decision")
    if existing and not config.resume and not config.force_regenerate:
        raise FileExistsError("Decision attempts already exist. Use --resume or --force-regenerate.")

    for record in tqdm(source.records, desc="decision evaluation", unit="row", disable=_quiet):
        available = source.rationales[record.example_id]
        unavailable = source.unavailable[record.example_id]
        for condition in COT_CONDITIONS:
            rationale = None if condition == "question_only" else available.get(condition)
            reason = None if condition == "question_only" else unavailable.get(condition)
            if reason is not None:
                state_hash = stable_hash({"example_id": record.example_id, "condition": condition, "unavailable": reason})
                unavailable_key = _cache_key(
                    record,
                    condition=condition,
                    state_hash=state_hash,
                    source_csv_sha256=source.sha256,
                    model_identity=model_identity,
                )
                if unavailable_key in existing and config.resume and not config.force_regenerate:
                    continue
                existing.add(
                    _append_skipped(
                        store,
                        record=record,
                        condition=condition,
                        config=config,
                        model_identity=model_identity,
                        source_csv_sha256=source.sha256,
                        reason=reason,
                    )
                )
                continue

            state = build_cot_decision_state(record, condition=condition, rationale=rationale)
            state_hash = stable_hash(state)
            cache_key = _cache_key(
                record,
                condition=condition,
                state_hash=state_hash,
                source_csv_sha256=source.sha256,
                model_identity=model_identity,
            )
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
                        model_identity=model_identity,
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
                    existing.add(cache_key)
                    break
                except Exception as error:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    terminal = attempt_number == config.max_retries + 1
                    projection = _base_projection(
                        record,
                        condition=condition,
                        config=config,
                        model_identity=model_identity,
                        state_hash=state_hash,
                        state=state,
                        attempt_count=attempt_number,
                        attempt_id=attempt_id,
                        latency_ms=latency_ms,
                    )
                    projection.update({"error_type": type(error).__name__, "error_message": sanitized_error(error)})
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
                    time.sleep(min(2 ** (attempt_number - 1), 30) + __import__("random").uniform(0, 1))



def evaluate_cot_decisions(
    source, *, config, store, model_identity, client, concurrency=1,
):
    """Bounded parallel HTTP evaluation; only the main thread writes attempts."""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from dataclasses import replace
    from queue import Queue, Empty
    import random

    if concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if config.decision_model.lower() == "laya" and concurrency != 1:
        raise ValueError("Laya local inference requires --concurrency 1.")
    if concurrency == 1:
        return _evaluate_cot_decisions_sequential(
            source, config=config, store=store,
            model_identity=model_identity, client=client,
        )

    existing = store.completed_keys("decision")
    if existing and not config.resume and not config.force_regenerate:
        raise FileExistsError("Decision attempts already exist. Use --resume or --force-regenerate.")
    events = Queue()
    stopping = __import__('threading').Event()

    class LedgerProxy:
        def completed_keys(self, stage):
            return set(existing)

        def append_attempt(self, stage, attempt):
            events.put((stage, attempt))

    class WorkerClient:
        def decide(self, state, record):
            if stopping.is_set():
                raise RuntimeError("Evaluation interrupted before request dispatch.")
            return client.decide(state, record)

    def drain():
        while True:
            try:
                stage, attempt = events.get_nowait()
            except Empty:
                break
            store.append_attempt(stage, attempt)

    def run_row(record):
        if stopping.is_set():
            return
        # Each row keeps the original condition ordering and cache semantics.
        row_source = replace(source, records=[record])
        _evaluate_cot_decisions_sequential(
            row_source, config=config, store=LedgerProxy(),
            model_identity=model_identity, client=WorkerClient(),
            _quiet=True,
        )

    records = iter(source.records)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = set()
    progress = tqdm(total=len(source.records), desc="decision evaluation", unit="row")
    try:
        for _ in range(concurrency):
            record = next(records, None)
            if record is None:
                break
            pending.add(pool.submit(run_row, record))
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            drain()
            for future in done:
                future.result()
                progress.update(1)
                record = next(records, None)
                if record is not None:
                    pending.add(pool.submit(run_row, record))
    finally:
        stopping.set()
        for future in pending:
            future.cancel()
        # Allow already dispatched requests to finish, then persist their events.
        pool.shutdown(wait=True, cancel_futures=True)
        drain()
        progress.close()


def _successful_prediction(row: Mapping[str, Any]) -> bool | None:
    if row.get("request_status") != "success":
        return None
    return as_bool(row.get("is_correct"))


def _paired_cot_input_token_summary(
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize reported input tokens on the paired four-condition cohort."""
    by_example: dict[str, dict[str, int]] = defaultdict(dict)
    for row in decision_rows:
        condition = str(row.get("condition") or "")
        if condition not in COT_CONDITIONS or row.get("request_status") != "success":
            continue
        count = as_number(row.get("decision_input_tokens"))
        if count is None or count < 0 or not count.is_integer():
            continue
        by_example[str(row.get("example_id"))][condition] = int(count)

    paired_ids = sorted(
        example_id
        for example_id, counts in by_example.items()
        if all(condition in counts for condition in COT_CONDITIONS)
    )
    token_values = {
        condition: [by_example[example_id][condition] for example_id in paired_ids]
        for condition in COT_CONDITIONS
    }
    baseline_values = token_values["question_only"]
    baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else None

    summaries: dict[str, dict[str, Any]] = {}
    for condition, values in token_values.items():
        mean = sum(values) / len(values) if values else None
        percentage = None
        if condition == "question_only":
            percentage = 0.0 if baseline_mean not in (None, 0) else None
        elif baseline_mean not in (None, 0) and mean is not None:
            percentage = (mean - baseline_mean) / baseline_mean * 100
        summaries[condition] = {
            "cot_tokens_reported_examples": len(values),
            "cot_tokens_total": sum(values) if values else None,
            "cot_tokens_mean": mean,
            "cot_tokens_std": pstdev(values) if values else None,
            "cot_tokens_delta_percent_vs_question_only": percentage,
        }
    return summaries


def _write_cot_analysis(
    source: CotCsvInput,
    store: RunStore,
    *,
    config: ExperimentConfig,
    model_identity: str,
) -> dict[str, Any]:
    decision_rows = sorted(
        store.latest_projections("decision"),
        key=lambda row: (
            int(row["source_row_index"]),
            COT_CONDITIONS.index(row["condition"]) if row.get("condition") in COT_CONDITIONS else len(COT_CONDITIONS),
        ),
    )
    write_csv_atomic(store.root / "decision_predictions.csv", decision_rows, DECISION_FIELDS)

    decisions_by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    correct_by_example: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in decision_rows:
        example_id = str(row["example_id"])
        condition = str(row["condition"])
        decisions_by_example[example_id][condition] = row
        correctness = _successful_prediction(row)
        if correctness is not None:
            correct_by_example[example_id][condition] = correctness

    shared_ids = sorted(
        example_id
        for example_id, outcomes in correct_by_example.items()
        if all(condition in outcomes for condition in COT_CONDITIONS)
    )
    shared_vectors = {
        condition: [correct_by_example[example_id][condition] for example_id in shared_ids]
        for condition in COT_CONDITIONS
    }
    paired: dict[str, dict[str, Any]] = {}
    for baseline, comparison in itertools.combinations(COT_CONDITIONS, 2):
        key = f"{comparison}_minus_{baseline}"
        paired[key] = paired_accuracy_difference(shared_vectors[baseline], shared_vectors[comparison])

    paired_from_question_only = {
        condition: paired.get(f"{condition}_minus_question_only")
        for condition in COT_CONDITIONS
        if condition != "question_only"
    }
    cot_token_summary = _paired_cot_input_token_summary(decision_rows)
    summary_rows: list[dict[str, Any]] = []
    for condition in COT_CONDITIONS:
        condition_rows = [row for row in decision_rows if row.get("condition") == condition]
        successful = [row for row in condition_rows if _successful_prediction(row) is not None]
        correct_count = sum(_successful_prediction(row) is True for row in successful)
        difference = None if condition == "question_only" else paired_from_question_only[condition]
        summary_rows.append(
            {
                "condition": condition,
                "decision_model": config.decision_model,
                "decision_model_checkpoint": model_identity,
                "eligible_examples": len(source.records),
                "completed_predictions": len(successful),
                "coverage": len(successful) / len(source.records) if source.records else None,
                "accuracy": correct_count / len(successful) if successful else None,
                "shared_completion_examples": len(shared_ids),
                "shared_completion_accuracy": (
                    sum(shared_vectors[condition]) / len(shared_ids) if shared_ids else None
                ),
                "question_only_difference_pp": (
                    0.0 if condition == "question_only" else difference["difference_pp"]
                ),
                "question_only_difference_ci_low_pp": (
                    0.0 if condition == "question_only" else difference["ci_low_pp"]
                ),
                "question_only_difference_ci_high_pp": (
                    0.0 if condition == "question_only" else difference["ci_high_pp"]
                ),
                **cot_token_summary[condition],
            }
        )
    write_csv_atomic(store.root / "summary.csv", summary_rows, SUMMARY_FIELDS)

    token_latency_rows: list[dict[str, Any]] = []
    for condition in COT_CONDITIONS:
        rows = [row for row in decision_rows if row.get("condition") == condition]
        latency = numeric_summary(row.get("latency_ms") for row in rows)
        input_tokens = numeric_summary(row.get("decision_input_tokens") for row in rows)
        output_tokens = numeric_summary(row.get("decision_output_tokens") for row in rows)
        token_latency_rows.append(
            {
                "stage": "decision",
                "variant": condition,
                "latency_count": latency["count"],
                "latency_mean_ms": latency["mean"],
                "latency_median_ms": latency["median"],
                "latency_p95_ms": latency["p95"],
                "input_token_count": input_tokens["count"],
                "input_token_mean": input_tokens["mean"],
                "output_token_count": output_tokens["count"],
                "output_token_mean": output_tokens["mean"],
            }
        )
    write_csv_atomic(store.root / "token_latency_summary.csv", token_latency_rows, TOKEN_LATENCY_FIELDS)

    per_question_rows: list[dict[str, Any]] = []
    for record in source.records:
        example_id = record.example_id
        row: dict[str, Any] = {
            "example_id": example_id,
            "dataset_config": record.dataset_config,
            "split": record.split,
            "source_row_index": record.source_row_index,
            "question": record.question,
            "choices_json": [choice.to_dict() for choice in record.choices],
            "gold_answer_label": record.gold_answer_label,
            "gold_answer_text": record.gold_answer_text,
            "standard_cot": source.raw_cots[example_id]["standard_cot"],
            "standard_rationale_sent": source.rationales[example_id].get("standard_reasoning"),
            "standard_rationale_status": source.unavailable[example_id].get("standard_reasoning", "available"),
            "poisoned_cot": source.raw_cots[example_id]["poisoned_cot"],
            "poetic_cot": source.raw_cots[example_id]["poetic_cot"],
        }
        for condition in COT_CONDITIONS:
            prediction = decisions_by_example.get(example_id, {}).get(condition, {})
            row[f"{condition}_status"] = prediction.get("request_status")
            row[f"{condition}_prediction"] = prediction.get("predicted_answer_label")
            row[f"{condition}_correct"] = prediction.get("is_correct")
            row[f"{condition}_confidence"] = prediction.get("confidence")
            row[f"{condition}_attempt_id"] = prediction.get("latest_attempt_id")
        per_question_rows.append(row)
    per_question_fields = list(per_question_rows[0]) if per_question_rows else []
    write_csv_atomic(store.root / "per_question_comparison.csv", per_question_rows, per_question_fields)

    error_rows = []
    for attempt in store.attempts("decision"):
        if attempt.get("status") == "success":
            continue
        error_rows.append(
            {
                "stage": "decision",
                "attempt_id": attempt.get("attempt_id"),
                "example_id": attempt.get("example_id"),
                "condition": attempt.get("condition"),
                "status": attempt.get("status"),
                "error_type": attempt.get("error_type"),
                "error_message": attempt.get("error_message"),
                "timestamp": attempt.get("timestamp"),
            }
        )
    write_csv_atomic(store.root / "errors.csv", error_rows, ERROR_FIELDS)
    analysis_dir = store.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        analysis_dir / "paired_comparisons.json",
        {"bootstrap_samples": 2000, "seed": 7, "shared_conditions": list(COT_CONDITIONS), "comparisons": paired},
    )
    return {
        "summary_rows": summary_rows,
        "paired_comparisons": paired,
        "per_question_count": len(per_question_rows),
        "error_count": len(error_rows),
    }


def run_cot_csv_evaluation(
    *,
    input_csv: str | Path,
    dataset_type: str,
    split: str,
    decision_model: str,
    model_identity: str,
    output_dir: str | Path,
    dataset_name: str = "arc",
    laya_finetuned_checkpoint: str | None = None,
    limit: int | None = None,
    max_retries: int = 2,
    request_timeout: float = 60.0,
    concurrency: int = 1,
    resume: bool = False,
    force_regenerate: bool = False,
    client: DecisionClient | None = None,
) -> dict[str, Any]:
    """Evaluate the four CSV-defined decision conditions and write experiment artifacts."""
    if concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if decision_model.lower() == "laya" and concurrency != 1:
        raise ValueError("Laya local inference requires --concurrency 1.")
    source = load_cot_csv(
        input_csv,
        dataset_type=dataset_type,
        split=split,
        limit=limit,
        dataset_name=dataset_name,
    )
    norm_dataset = dataset_name.lower().replace("-", "_")
    canonical_config = canonical_dataset_config(dataset_type) if norm_dataset == "arc" else dataset_type
    config = ExperimentConfig(
        output_dir=Path(output_dir).resolve(),
        dataset_config=canonical_config,
        split=split,
        decision_model=decision_model,
        laya_finetuned_checkpoint=laya_finetuned_checkpoint,
        gemini_model="not_used_local_cot_csv",
        gemini_api_key_env="NOT_USED",
        limit=limit,
        resume=resume,
        max_retries=max_retries,
        request_timeout=request_timeout,
        force_regenerate=force_regenerate,
    )
    store = RunStore(config.output_dir)
    metadata = {
        "input_csv": str(source.path),
        "input_csv_sha256": source.sha256,
        "dataset_id": source.records[0].dataset_id if source.records else DATASET_ID,
        "dataset_name": norm_dataset,
        "dataset_config": canonical_config,
        "split": split,
        "limit": limit,
        "selected_rows": len(source.records),
        "decision_model": decision_model,
        "model_identity": model_identity,
        "conditions": list(COT_CONDITIONS),
        "standard_final_answer_removed_from_state": True,
    }
    store.initialize(config)
    manifest = read_json(store.manifest_path)
    previous_metadata = manifest.get("cot_csv_evaluation")
    if previous_metadata is not None:
        for key, value in metadata.items():
            if previous_metadata.get(key) != value:
                raise ValueError(
                    f"Output directory was initialized with a different {key}; use a new --output-dir."
                )
    elif store.dataset_records_path.exists() or store.attempts("decision"):
        raise ValueError("Output directory already contains another experiment; use a new --output-dir.")

    if store.dataset_records_path.exists():
        existing_records = store.load_records()
        if stable_hash([record.to_dict() for record in existing_records]) != stable_hash(
            [record.to_dict() for record in source.records]
        ):
            raise ValueError("Stored normalized records differ from the input CSV; use a new --output-dir.")
    else:
        store.write_dataset(
            source.records,
            dataset_fingerprint=source.sha256,
            datasets_version="vllm_cot_csv",
        )

    store.record_decision_model(config)
    manifest = read_json(store.manifest_path)
    manifest_config = manifest.setdefault("config", {})
    for unused_key in (
        "gemini_model",
        "gemini_api_key_env",
        "poem_form",
        "temperature",
        "seed",
        "gemini_requests_per_minute",
        "include_flagged_reasoning",
    ):
        manifest_config.pop(unused_key, None)
    manifest["cot_csv_evaluation"] = metadata
    write_json_atomic(store.manifest_path, manifest)

    decision_client = client or create_decision_client(
        decision_model,
        timeout=request_timeout,
        laya_finetuned_checkpoint=laya_finetuned_checkpoint,
    )
    evaluate_cot_decisions(
        source,
        config=config,
        store=store,
        model_identity=model_identity,
        client=decision_client,
        concurrency=concurrency,
    )
    analysis = _write_cot_analysis(source, store, config=config, model_identity=model_identity)
    return {
        "input_csv": str(source.path),
        "output_dir": str(store.root),
        "decision_model": decision_model,
        **analysis,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned local decision model or hosted JEV on question-only and three vLLM CoT conditions."
    )
    parser.add_argument("--input-csv", required=True, help="CoT CSV file (e.g. ARC, LogiQA, TypedDecisions, ProofWriter).")
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset",
        default=None,
        choices=("arc", "typed_decisions", "proofwriter", "logiqa"),
        help="Target reasoning dataset: arc, typed_decisions, proofwriter, or logiqa (auto-detected from CSV if omitted).",
    )
    parser.add_argument(
        "--dataset-config",
        "--dataset-type",
        dest="dataset_config",
        default=None,
        help="Dataset configuration / subset (e.g. easy, challenge, all, default; auto-detected if omitted).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "validation"),
        default=None,
        help="Data split (default: test, or auto-detected from filename).",
    )
    parser.add_argument("--decision-model", choices=DECISION_MODELS, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--laya-finetuned-checkpoint")
    parser.add_argument(
        "--kev-finetuned-checkpoint",
        "--kev-model-id",
        dest="kev_finetuned_checkpoint",
        help="Override KEV_MODEL_ID with this fine-tuned Kev checkpoint directory.",
    )
    parser.add_argument(
        "--clef-finetuned-checkpoint",
        "--clef-model-id",
        dest="clef_finetuned_checkpoint",
        help="Override CLEF_MODEL_ID with this fine-tuned Clef checkpoint directory.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous HTTP requests; Laya requires 1.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run four-condition evaluation for one selected fine-tuned local model or hosted JEV."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        input_csv = Path(args.input_csv).expanduser()
        if not input_csv.is_absolute():
            input_csv = _project_root() / input_csv
        input_csv = input_csv.resolve()

        detected_ds, detected_cfg, detected_split = detect_dataset_from_csv(input_csv)
        dataset_name = (args.dataset or detected_ds).lower().replace("-", "_")
        dataset_type = args.dataset_config or detected_cfg
        split = args.split or detected_split

        model_identity, laya_checkpoint = resolve_finetuned_model_identity(
            args.decision_model,
            laya_finetuned_checkpoint=args.laya_finetuned_checkpoint,
            kev_finetuned_checkpoint=args.kev_finetuned_checkpoint,
            clef_finetuned_checkpoint=args.clef_finetuned_checkpoint,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
        )
        requested_output = args.output_dir or default_output_dir(input_csv, args.decision_model)
        output_dir = resolve_output_dir(requested_output)
        result = run_cot_csv_evaluation(
            input_csv=input_csv,
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            split=split,
            decision_model=args.decision_model,
            model_identity=model_identity,
            output_dir=output_dir,
            laya_finetuned_checkpoint=laya_checkpoint,
            limit=args.limit,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            resume=args.resume,
            force_regenerate=args.force_regenerate,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())