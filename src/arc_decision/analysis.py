"""Rebuild all analysis artifacts from persisted records without making model calls."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluation import CONDITIONS
from .metrics import as_bool, as_number, numeric_summary, paired_accuracy_difference
from .models import ArcRecord
from .storage import RunStore, write_csv_atomic, write_json_atomic


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact {path}.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_example(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["example_id"]): dict(row) for row in rows if row.get("example_id")}


def _completed(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("request_status") == "success" and as_bool(row.get("is_correct")) is not None]


def _shared_condition_vectors(rows: Iterable[Mapping[str, str]]) -> dict[str, list[bool]]:
    grouped: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in _completed(rows):
        correctness = as_bool(row.get("is_correct"))
        if correctness is not None:
            grouped[str(row["example_id"])][str(row["condition"])] = correctness
    vectors = {condition: [] for condition in CONDITIONS}
    for outcomes in grouped.values():
        if all(condition in outcomes for condition in CONDITIONS):
            for condition in CONDITIONS:
                vectors[condition].append(outcomes[condition])
    return vectors


def _generation_output_token_count(row: Mapping[str, str], field_name: str) -> int | None:
    if row.get("generation_status") not in {"success", "flagged"}:
        return None
    value = as_number(row.get(field_name))
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _paired_generation_token_summary(
    normal_rows: list[dict[str, str]],
    poetic_rows: list[dict[str, str]],
) -> dict[str, Any]:
    normal_by_example = _by_example(normal_rows)
    poetic_by_example = _by_example(poetic_rows)
    normal_tokens: list[int] = []
    poetic_tokens: list[int] = []
    for example_id in sorted(normal_by_example.keys() & poetic_by_example.keys()):
        normal_count = _generation_output_token_count(normal_by_example[example_id], "cot_output_tokens")
        poetic_count = _generation_output_token_count(poetic_by_example[example_id], "poetic_cot_output_tokens")
        if normal_count is not None and poetic_count is not None:
            normal_tokens.append(normal_count)
            poetic_tokens.append(poetic_count)

    pair_examples = len(normal_tokens)
    if not pair_examples:
        return {
            "pair_examples": 0,
            "normal_total": None,
            "normal_mean": None,
            "poetic_total": None,
            "poetic_mean": None,
            "total_delta": None,
            "mean_delta": None,
            "delta_percent": None,
        }

    normal_total = sum(normal_tokens)
    poetic_total = sum(poetic_tokens)
    total_delta = poetic_total - normal_total
    return {
        "pair_examples": pair_examples,
        "normal_total": normal_total,
        "normal_mean": normal_total / pair_examples,
        "poetic_total": poetic_total,
        "poetic_mean": poetic_total / pair_examples,
        "total_delta": total_delta,
        "mean_delta": total_delta / pair_examples,
        "delta_percent": total_delta / normal_total * 100 if normal_total else None,
    }


def _condition_generation_token_fields(condition: str, token_summary: Mapping[str, Any]) -> dict[str, Any]:
    field_names = (
        "gemini_token_pair_examples",
        "gemini_output_tokens_total",
        "gemini_output_tokens_mean",
        "gemini_output_tokens_total_delta",
        "gemini_output_tokens_mean_delta",
        "gemini_output_tokens_delta_percent",
    )
    if condition == "question_only":
        return {field_name: None for field_name in field_names}

    pair_examples = token_summary["pair_examples"]
    if not pair_examples:
        return {
            "gemini_token_pair_examples": 0,
            "gemini_output_tokens_total": None,
            "gemini_output_tokens_mean": None,
            "gemini_output_tokens_total_delta": None,
            "gemini_output_tokens_mean_delta": None,
            "gemini_output_tokens_delta_percent": None,
        }

    is_normal = condition == "normal_reasoning"
    normal_total = token_summary["normal_total"]
    return {
        "gemini_token_pair_examples": pair_examples,
        "gemini_output_tokens_total": normal_total if is_normal else token_summary["poetic_total"],
        "gemini_output_tokens_mean": token_summary["normal_mean"] if is_normal else token_summary["poetic_mean"],
        "gemini_output_tokens_total_delta": 0 if is_normal else token_summary["total_delta"],
        "gemini_output_tokens_mean_delta": 0.0 if is_normal else token_summary["mean_delta"],
        "gemini_output_tokens_delta_percent": (
            0.0 if is_normal and normal_total else None if is_normal else token_summary["delta_percent"]
        ),
    }


def _summary_rows(
    decision_rows: list[dict[str, str]],
    eligible_count: int,
    normal_rows: list[dict[str, str]],
    poetic_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completed_rows = _completed(decision_rows)
    vectors = _shared_condition_vectors(decision_rows)
    token_summary = _paired_generation_token_summary(normal_rows, poetic_rows)
    paired = {
        "normal_reasoning_minus_question_only": paired_accuracy_difference(
            vectors["question_only"], vectors["normal_reasoning"]
        ),
        "poetic_reasoning_minus_question_only": paired_accuracy_difference(
            vectors["question_only"], vectors["poetic_reasoning"]
        ),
        "normal_reasoning_minus_poetic_reasoning": paired_accuracy_difference(
            vectors["poetic_reasoning"], vectors["normal_reasoning"]
        ),
    }
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in decision_rows if row.get("condition") == condition]
        successful = [row for row in completed_rows if row.get("condition") == condition]
        correct = sum(as_bool(row.get("is_correct")) is True for row in successful)
        shared = vectors[condition]
        comparison_key = {
            "question_only": None,
            "normal_reasoning": "normal_reasoning_minus_question_only",
            "poetic_reasoning": "poetic_reasoning_minus_question_only",
        }[condition]
        comparison = paired.get(comparison_key) if comparison_key else None
        rows.append(
            {
                "condition": condition,
                "decision_model": next((row.get("decision_model") for row in condition_rows if row.get("decision_model")), None),
                "eligible_examples": eligible_count,
                "completed_predictions": len(successful),
                "coverage": len(successful) / eligible_count if eligible_count else None,
                "accuracy": correct / len(successful) if successful else None,
                "shared_completion_examples": len(shared),
                "shared_completion_accuracy": sum(shared) / len(shared) if shared else None,
                "question_only_difference_pp": comparison["difference_pp"] if comparison else 0.0,
                "question_only_difference_ci_low_pp": comparison["ci_low_pp"] if comparison else 0.0,
                "question_only_difference_ci_high_pp": comparison["ci_high_pp"] if comparison else 0.0,
                **_condition_generation_token_fields(condition, token_summary),
            }
        )
    return rows, paired


def _token_latency_rows(
    decision_rows: list[dict[str, str]],
    normal_rows: list[dict[str, str]],
    poetic_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        subset = [row for row in decision_rows if row.get("condition") == condition]
        latency = numeric_summary(row.get("latency_ms") for row in subset)
        input_tokens = numeric_summary(row.get("decision_input_tokens") for row in subset)
        output_tokens = numeric_summary(row.get("decision_output_tokens") for row in subset)
        rows.append(
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
    for variant, source_rows, output_field in (
        ("normal", normal_rows, "cot_output_tokens"),
        ("poetic", poetic_rows, "poetic_cot_output_tokens"),
    ):
        latency = numeric_summary(row.get("latency_ms") for row in source_rows)
        output_tokens = numeric_summary(row.get(output_field) for row in source_rows)
        reasoning_words = numeric_summary(row.get("reasoning_word_count") for row in source_rows)
        rows.append(
            {
                "stage": "gemini",
                "variant": variant,
                "latency_count": latency["count"],
                "latency_mean_ms": latency["mean"],
                "latency_median_ms": latency["median"],
                "latency_p95_ms": latency["p95"],
                "output_token_count": output_tokens["count"],
                "output_token_mean": output_tokens["mean"],
                "reasoning_word_count": reasoning_words["count"],
                "reasoning_word_mean": reasoning_words["mean"],
            }
        )
    return rows


def _per_question_rows(
    records: Iterable[ArcRecord],
    decision_rows: list[dict[str, str]],
    normal_rows: list[dict[str, str]],
    poetic_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    decisions: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in decision_rows:
        decisions[str(row["example_id"])][str(row["condition"])] = row
    normal = _by_example(normal_rows)
    poetic = _by_example(poetic_rows)
    output: list[dict[str, Any]] = []
    for record in records:
        normal_row = normal.get(record.example_id, {})
        poetic_row = poetic.get(record.example_id, {})
        row: dict[str, Any] = {
            "example_id": record.example_id,
            "dataset_config": record.dataset_config,
            "split": record.split,
            "source_row_index": record.source_row_index,
            "question": record.question,
            "choices_json": [choice.to_dict() for choice in record.choices],
            "gold_answer_label": record.gold_answer_label,
            "gold_answer_text": record.gold_answer_text,
            "cot_reasoning": normal_row.get("cot_reasoning"),
            "normal_generation_status": normal_row.get("generation_status"),
            "normal_leakage_status": normal_row.get("leakage_status"),
            "normal_gemini_answer_label": normal_row.get("final_answer_label"),
            "poetic_cot_reasoning": poetic_row.get("poetic_cot_reasoning"),
            "poetic_generation_status": poetic_row.get("generation_status"),
            "poetic_leakage_status": poetic_row.get("leakage_status"),
            "poetic_gemini_answer_label": poetic_row.get("final_answer_label"),
        }
        for condition in CONDITIONS:
            prediction = decisions.get(record.example_id, {}).get(condition, {})
            prefix = condition
            row[f"{prefix}_status"] = prediction.get("request_status")
            row[f"{prefix}_prediction"] = prediction.get("predicted_answer_label")
            row[f"{prefix}_correct"] = prediction.get("is_correct")
            row[f"{prefix}_confidence"] = prediction.get("confidence")
            row[f"{prefix}_attempt_id"] = prediction.get("latest_attempt_id")
        output.append(row)
    return output


def _error_rows(store: RunStore) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stage in ("gemini", "decision"):
        for attempt in store.attempts(stage):
            if attempt.get("status") == "success":
                continue
            output.append(
                {
                    "stage": stage,
                    "attempt_id": attempt.get("attempt_id"),
                    "example_id": attempt.get("example_id"),
                    "condition": attempt.get("condition") or attempt.get("task"),
                    "status": attempt.get("status"),
                    "error_type": attempt.get("error_type"),
                    "error_message": attempt.get("error_message"),
                    "timestamp": attempt.get("timestamp"),
                }
            )
    return output


def analyze_run(store: RunStore) -> dict[str, Any]:
    """Write every derived analysis artifact with no model or network calls."""
    records = store.load_records()
    decision_rows = _read_csv(store.root / "decision_predictions.csv")
    normal_rows = _read_csv(store.root / "front_cot.csv")
    poetic_rows = _read_csv(store.root / "front_poetic.csv")
    summary_rows, paired = _summary_rows(decision_rows, len(records), normal_rows, poetic_rows)
    token_latency_rows = _token_latency_rows(decision_rows, normal_rows, poetic_rows)
    per_question_rows = _per_question_rows(records, decision_rows, normal_rows, poetic_rows)
    errors = _error_rows(store)
    write_csv_atomic(store.root / "summary.csv", summary_rows)
    write_csv_atomic(store.root / "token_latency_summary.csv", token_latency_rows)
    write_csv_atomic(store.root / "per_question_comparison.csv", per_question_rows)
    write_csv_atomic(store.root / "errors.csv", errors)
    analysis_dir = store.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        analysis_dir / "paired_comparisons.json",
        {"bootstrap_samples": 2000, "seed": 7, "comparisons": paired},
    )
    return {
        "summary_rows": summary_rows,
        "paired_comparisons": paired,
        "per_question_count": len(per_question_rows),
        "error_count": len(errors),
    }
