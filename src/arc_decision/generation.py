"""Resumable one-question-at-a-time Gemini rationale generation."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Iterable

from tqdm import tqdm

from .config import ExperimentConfig
from .gemini_client import GeminiClient, result_as_dict
from .leakage import detect_answer_leakage, validate_poem_form
from .models import ArcRecord, GeminiResult
from .prompts import GenerationTask, build_gemini_prompt
from .storage import RunStore, sanitized_error, stable_hash, write_csv_atomic


GENERATION_FIELDS = (
    "run_id",
    "example_id",
    "dataset_config",
    "split",
    "source_row_index",
    "question",
    "choices_json",
    "gold_answer_label",
    "gold_answer_text",
    "gemini_model",
    "prompt_hash",
    "cot_reasoning",
    "poetic_cot_reasoning",
    "poem_form_requested",
    "poem_form_returned",
    "poem_form_validation_status",
    "reasoning_char_count",
    "reasoning_word_count",
    "reasoning_text_token_count",
    "reasoning_token_count_method",
    "final_answer_label",
    "final_answer_text",
    "final_answer_correct",
    "leakage_status",
    "leakage_hard_reasons_json",
    "leakage_review_reasons_json",
    "generation_status",
    "attempt_count",
    "latency_ms",
    "gemini_prompt_tokens",
    "gemini_candidate_output_tokens",
    "gemini_total_tokens",
    "cot_output_tokens",
    "poetic_cot_output_tokens",
    "finish_reason",
    "gemini_request_input_json",
    "gemini_raw_response_json",
    "latest_attempt_id",
    "error_type",
    "error_message",
)


class GeminiQuotaExhaustedError(RuntimeError):
    """Raised after checkpointing a quota failure and stopping generation."""


def _is_quota_exhausted_error(error: BaseException) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    error_text = f"{code or ''} {error}".upper()
    try:
        is_429 = int(code) == 429
    except (TypeError, ValueError):
        is_429 = False
    if not is_429 and not re.search(r"\b429\b", error_text):
        return False
    quota_exhaustion_markers = (
        "QUOTA EXCEEDED",
        "QUOTA HAS BEEN EXCEEDED",
        "LIMIT EXCEEDED",
        "LIMIT HAS BEEN EXCEEDED",
        "EXCEEDED YOUR CURRENT QUOTA",
        "EXCEEDED YOUR CURRENT LIMIT",
        "REQUEST LIMIT REACHED",
        "DAILY LIMIT REACHED",
    )
    return any(marker in error_text for marker in quota_exhaustion_markers)


def _is_retryable_error(error: BaseException) -> bool:
    """Retry transient failures, but stop on permanent request and quota errors."""
    if _is_quota_exhausted_error(error):
        return False
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    try:
        if int(code) in {400, 401, 403, 404, 405, 410, 422}:
            return False
    except (TypeError, ValueError):
        pass
    error_text = f"{code or ''} {error}".upper()
    permanent_markers = (
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "MODEL_NOT_FOUND",
    )
    return not any(marker in error_text for marker in permanent_markers) and not re.search(
        r"\b(?:400|401|403|404|405|410|422)\b", error_text
    )


def _retry_delay(error: BaseException, attempt_number: int) -> float:
    """Prefer a provider-supplied retry delay, otherwise use bounded exponential backoff."""
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        if hasattr(retry_after, "total_seconds"):
            return max(0.0, float(retry_after.total_seconds()))
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

    match = re.search(r"\bretry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s\b", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return float(min(2 ** (attempt_number - 1), 30))


class _RequestRateLimiter:
    """Space Gemini attempts below the configured per-minute request budget."""

    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute + 0.1
        self.last_request_at: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_request_at is not None:
            remaining = self.interval - (now - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        self.last_request_at = now


def _cache_key(
    record: ArcRecord,
    *,
    task: GenerationTask,
    config: ExperimentConfig,
    prompt_hash: str,
) -> str:
    return stable_hash(
        {
            "stage": "gemini",
            "example_id": record.example_id,
            "task": task,
            "gemini_model": config.gemini_model,
            "poem_form": config.poem_form if task == "poetic" else None,
            "prompt_hash": prompt_hash,
            "temperature": config.temperature,
            "seed": config.seed,
        }
    )


def _base_projection(
    record: ArcRecord,
    *,
    task: GenerationTask,
    config: ExperimentConfig,
    prompt_hash: str,
    request_input: dict[str, Any],
    attempt_count: int,
    latency_ms: int | None,
    attempt_id: str | None,
) -> dict[str, Any]:
    return {
        "run_id": config.output_dir.name,
        "example_id": record.example_id,
        "dataset_config": record.dataset_config,
        "split": record.split,
        "source_row_index": record.source_row_index,
        "question": record.question,
        "choices_json": [choice.to_dict() for choice in record.choices],
        "gold_answer_label": record.gold_answer_label,
        "gold_answer_text": record.gold_answer_text,
        "gemini_model": config.gemini_model,
        "prompt_hash": prompt_hash,
        "cot_reasoning": None,
        "poetic_cot_reasoning": None,
        "poem_form_requested": config.poem_form if task == "poetic" else None,
        "poem_form_returned": None,
        "poem_form_validation_status": None,
        "reasoning_char_count": None,
        "reasoning_word_count": None,
        "reasoning_text_token_count": None,
        "reasoning_token_count_method": "unavailable",
        "final_answer_label": None,
        "final_answer_text": None,
        "final_answer_correct": None,
        "leakage_status": None,
        "leakage_hard_reasons_json": [],
        "leakage_review_reasons_json": [],
        "generation_status": "failed",
        "attempt_count": attempt_count,
        "latency_ms": latency_ms,
        "gemini_prompt_tokens": None,
        "gemini_candidate_output_tokens": None,
        "gemini_total_tokens": None,
        "cot_output_tokens": None,
        "poetic_cot_output_tokens": None,
        "finish_reason": None,
        "gemini_request_input_json": request_input,
        "gemini_raw_response_json": None,
        "latest_attempt_id": attempt_id,
        "error_type": None,
        "error_message": None,
    }


def _successful_projection(
    record: ArcRecord,
    *,
    task: GenerationTask,
    config: ExperimentConfig,
    prompt_hash: str,
    request_input: dict[str, Any],
    result: GeminiResult,
    attempt_count: int,
    latency_ms: int,
    attempt_id: str | None,
) -> dict[str, Any]:
    projection = _base_projection(
        record,
        task=task,
        config=config,
        prompt_hash=prompt_hash,
        request_input=request_input,
        attempt_count=attempt_count,
        latency_ms=latency_ms,
        attempt_id=attempt_id,
    )
    leakage = detect_answer_leakage(result.reasoning, record)
    projection.update(
        {
            "gemini_model": result.model,
            "final_answer_label": result.final_answer_label,
            "final_answer_text": result.final_answer_text,
            "final_answer_correct": result.final_answer_label == record.gold_answer_label,
            "reasoning_char_count": len(result.reasoning),
            "reasoning_word_count": len(result.reasoning.split()),
            "leakage_status": leakage.status,
            "leakage_hard_reasons_json": list(leakage.hard_reasons),
            "leakage_review_reasons_json": list(leakage.review_reasons),
            "generation_status": "flagged" if leakage.is_hard_flagged else "success",
            "gemini_prompt_tokens": result.usage.input_tokens,
            "gemini_candidate_output_tokens": result.usage.output_tokens,
            "gemini_total_tokens": result.usage.total_tokens,
            "finish_reason": result.finish_reason,
            "gemini_raw_response_json": result.raw_response,
        }
    )
    if task == "normal":
        projection["cot_reasoning"] = result.reasoning
        projection["cot_output_tokens"] = result.usage.output_tokens
    else:
        projection["poetic_cot_reasoning"] = result.reasoning
        projection["poetic_cot_output_tokens"] = result.usage.output_tokens
        projection["poem_form_returned"] = result.poem_form
        projection["poem_form_validation_status"] = validate_poem_form(
            config.poem_form,
            result.poem_form,
            result.reasoning,
        )
    return projection


def _write_generation_projections(store: RunStore) -> None:
    projections = store.latest_projections("gemini")
    normal_rows = sorted(
        (row for row in projections if row.get("task") == "normal"),
        key=lambda row: int(row["source_row_index"]),
    )
    poetic_rows = sorted(
        (row for row in projections if row.get("task") == "poetic"),
        key=lambda row: int(row["source_row_index"]),
    )
    write_csv_atomic(store.root / "front_cot.csv", normal_rows, GENERATION_FIELDS)
    write_csv_atomic(store.root / "front_poetic.csv", poetic_rows, GENERATION_FIELDS)


def generate_rationales(
    records: Iterable[ArcRecord],
    *,
    config: ExperimentConfig,
    store: RunStore,
    client: GeminiClient,
) -> None:
    """Generate normal and poetic rationales independently for every supplied record."""
    records = list(records)
    store.initialize(config)
    existing = store.completed_keys("gemini")
    if existing and not config.resume and not config.force_regenerate:
        raise FileExistsError(
            "Gemini attempt ledger already has completed records. Use --resume or --force-regenerate explicitly."
        )
    rate_limiter = _RequestRateLimiter(config.gemini_requests_per_minute)
    checkpoint_every = 1 if len(records) * 2 <= 100 else 25
    completed_tasks = 0
    progress = tqdm(
        total=len(records) * 2,
        desc="Gemini generation",
        unit="task",
        dynamic_ncols=True,
    )
    quota_exhaustion_detected = False
    try:
        for record in records:
            for task in ("normal", "poetic"):
                prompt = build_gemini_prompt(
                    record,
                    task=task,
                    poem_form=config.poem_form,
                    strict_separation=False,
                )
                prompt_hash = stable_hash(prompt)
                cache_key = _cache_key(record, task=task, config=config, prompt_hash=prompt_hash)
                if cache_key in existing and config.resume and not config.force_regenerate:
                    progress.set_postfix_str(f"{record.example_id} {task}: cached")
                    progress.update(1)
                    completed_tasks += 1
                    if completed_tasks % checkpoint_every == 0:
                        _write_generation_projections(store)
                    continue
                request_input = {
                    "task": task,
                    "poem_form": config.poem_form if task == "poetic" else None,
                    "prompt": prompt,
                }
                task_status = "failed"
                for attempt_number in range(1, config.max_retries + 2):
                    progress.set_postfix_str(f"{record.example_id} {task}: attempt {attempt_number}")
                    rate_limiter.wait()
                    started = time.monotonic()
                    attempt_id = str(uuid.uuid4())
                    retry_delay = None
                    try:
                        result = client.generate(
                            record,
                            task=task,
                            poem_form=config.poem_form,
                            strict_separation=attempt_number > 1,
                        )
                        latency_ms = int((time.monotonic() - started) * 1000)
                        provisional_projection = _successful_projection(
                            record,
                            task=task,
                            config=config,
                            prompt_hash=prompt_hash,
                            request_input=request_input,
                            result=result,
                            attempt_count=attempt_number,
                            latency_ms=latency_ms,
                            attempt_id=attempt_id,
                        )
                        hard_flagged = provisional_projection["leakage_status"] == "hard_flagged"
                        task_status = "hard_flagged" if hard_flagged else "success"
                        terminal = not hard_flagged or attempt_number == config.max_retries + 1
                        store.append_attempt(
                            "gemini",
                            {
                                "attempt_id": attempt_id,
                                "run_id": config.output_dir.name,
                                "example_id": record.example_id,
                                "cache_key": cache_key,
                                "task": task,
                                "attempt_number": attempt_number,
                                "terminal": terminal,
                                "status": "hard_flagged" if hard_flagged else "success",
                                "latency_ms": latency_ms,
                                "request_input": request_input,
                                "request_input_hash": stable_hash(request_input),
                                "result": result_as_dict(result),
                                "projection": {**provisional_projection, "task": task} if terminal else None,
                            },
                        )
                        if terminal:
                            break
                        retry_delay = float(min(2 ** (attempt_number - 1), 30))
                    except Exception as error:  # API, transport, or strict parsing error; all are retained.
                        latency_ms = int((time.monotonic() - started) * 1000)
                        quota_exhaustion_detected = _is_quota_exhausted_error(error)
                        retryable = _is_retryable_error(error)
                        terminal = attempt_number == config.max_retries + 1 or not retryable
                        projection = _base_projection(
                            record,
                            task=task,
                            config=config,
                            prompt_hash=prompt_hash,
                            request_input=request_input,
                            attempt_count=attempt_number,
                            latency_ms=latency_ms,
                            attempt_id=attempt_id,
                        )
                        projection.update(
                            {
                                "task": task,
                                "error_type": type(error).__name__,
                                "error_message": sanitized_error(error),
                            }
                        )
                        store.append_attempt(
                            "gemini",
                            {
                                "attempt_id": attempt_id,
                                "run_id": config.output_dir.name,
                                "example_id": record.example_id,
                                "cache_key": cache_key,
                                "task": task,
                                "attempt_number": attempt_number,
                                "terminal": terminal,
                                "status": "failed",
                                "latency_ms": latency_ms,
                                "request_input": request_input,
                                "request_input_hash": stable_hash(request_input),
                                "error_type": type(error).__name__,
                                "error_message": sanitized_error(error),
                                "projection": projection if terminal else None,
                            },
                        )
                        if terminal:
                            task_status = "failed"
                            break
                        retry_delay = _retry_delay(error, attempt_number)
                    if retry_delay is not None:
                        time.sleep(retry_delay)
                progress.set_postfix_str(f"{record.example_id} {task}: {task_status}")
                progress.update(1)
                completed_tasks += 1
                if completed_tasks % checkpoint_every == 0:
                    _write_generation_projections(store)
                if quota_exhaustion_detected:
                    break
            if quota_exhaustion_detected:
                break
    finally:
        progress.close()
    _write_generation_projections(store)
    if quota_exhaustion_detected:
        raise GeminiQuotaExhaustedError(
            "Gemini reported a 429 quota/limit exhaustion. The failed attempt was saved and no further "
            "generation requests were sent; resume after the provider quota resets or is increased."
        )


def rebuild_generation_csvs(store: RunStore) -> None:
    """Regenerate both required CSV files from saved Gemini attempt records only."""
    _write_generation_projections(store)
