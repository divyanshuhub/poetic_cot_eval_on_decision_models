"""ARC dataset loading, normalization, and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Iterable, Mapping

from .models import ArcRecord, Choice


DATASET_ID = "allenai/ai2_arc"


class DatasetValidationError(ValueError):
    """Raised when an ARC row cannot be safely normalized."""


@dataclass(frozen=True)
class SkippedRecord:
    source_row_index: int
    reason: str


@dataclass(frozen=True)
class DatasetLoadResult:
    records: tuple[ArcRecord, ...]
    skipped: tuple[SkippedRecord, ...]
    dataset_fingerprint: str | None
    datasets_version: str | None


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _extract_choices(raw_choices: Any) -> tuple[Choice, ...]:
    """Normalize the ARC choices mapping while preserving its source order."""
    if isinstance(raw_choices, Mapping):
        labels = raw_choices.get("label", raw_choices.get("labels"))
        texts = raw_choices.get("text", raw_choices.get("texts"))
        if not isinstance(labels, list) or not isinstance(texts, list):
            raise DatasetValidationError("choices must contain list-valued label and text fields.")
        pairs = zip(labels, texts, strict=True)
    elif isinstance(raw_choices, list):
        try:
            pairs = ((item["label"], item["text"]) for item in raw_choices)
        except (KeyError, TypeError) as error:
            raise DatasetValidationError("choice entries must contain label and text fields.") from error
    else:
        raise DatasetValidationError("choices must be a mapping or a list of mappings.")

    choices = tuple(Choice(_string(label, "choice label"), _string(text, "choice text")) for label, text in pairs)
    if len(choices) < 2:
        raise DatasetValidationError("at least two answer choices are required.")
    labels = [choice.label for choice in choices]
    if len(set(labels)) != len(labels):
        raise DatasetValidationError("choice labels must be unique.")
    return choices


def normalize_arc_row(
    row: Mapping[str, Any],
    *,
    dataset_config: str,
    split: str,
    source_row_index: int,
) -> ArcRecord:
    """Convert one ARC row into the stable record used by every pipeline stage."""
    question = row.get("question", row.get("question_stem"))
    question_text = _string(question, "question")
    choices = _extract_choices(row.get("choices"))
    answer_key = _string(row.get("answerKey", row.get("answer_key")), "answer key")
    choice_map = {choice.label: choice.text for choice in choices}
    if answer_key not in choice_map:
        raise DatasetValidationError(
            f"answer key {answer_key!r} is not one of the available labels: {sorted(choice_map)}."
        )
    config_slug = dataset_config.lower().replace("-", "_")
    return ArcRecord(
        example_id=f"arc_{config_slug}__{split}__{source_row_index:06d}",
        dataset_id=DATASET_ID,
        dataset_config=dataset_config,
        split=split,
        source_row_index=source_row_index,
        question=question_text,
        choices=choices,
        gold_answer_label=answer_key,
        gold_answer_text=choice_map[answer_key],
    )


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_config: str,
    split: str,
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Normalize in source order and retain malformed rows as observable skips."""
    records: list[ArcRecord] = []
    skipped: list[SkippedRecord] = []
    for source_row_index, row in enumerate(rows):
        if limit is not None and len(records) >= limit:
            break
        try:
            records.append(
                normalize_arc_row(
                    row,
                    dataset_config=dataset_config,
                    split=split,
                    source_row_index=source_row_index,
                )
            )
        except DatasetValidationError as error:
            skipped.append(SkippedRecord(source_row_index=source_row_index, reason=str(error)))
    if strict and skipped:
        sample = "; ".join(f"row {item.source_row_index}: {item.reason}" for item in skipped[:3])
        raise DatasetValidationError(f"Dataset validation failed for {len(skipped)} row(s): {sample}")
    return DatasetLoadResult(
        records=tuple(records),
        skipped=tuple(skipped),
        dataset_fingerprint=None,
        datasets_version=None,
    )


def load_arc_dataset(
    *,
    dataset_config: str,
    split: str,
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Load ARC lazily so offline tests do not require the datasets dependency."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "ARC loading requires the optional 'datasets' dependency. Install this package's dependencies first."
        ) from error

    dataset = load_dataset(DATASET_ID, dataset_config, split=split)
    result = normalize_rows(
        dataset,
        dataset_config=dataset_config,
        split=split,
        strict=strict,
        limit=limit,
    )
    return DatasetLoadResult(
        records=result.records,
        skipped=result.skipped,
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        datasets_version=version("datasets"),
    )


def load_typed_decisions_dataset(
    *,
    dataset_config: str = "all",
    split: str = "train",
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Load LocalLLaMA/typed-decisions, extracting choice-type questions."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "typed-decisions loading requires the 'datasets' package. Install dependencies first."
        ) from error

    dataset = load_dataset("LocalLLaMA/typed-decisions", dataset_config, split=split)
    records: list[ArcRecord] = []
    skipped: list[SkippedRecord] = []
    row_counter = 0

    for source_row_index, row in enumerate(dataset):
        if limit is not None and len(records) >= limit:
            break
        state = row.get("state") or ""
        questions_raw = row.get("questions") or {}
        gold_raw = row.get("gold") or {}
        if isinstance(questions_raw, str):
            try:
                questions = json.loads(questions_raw)
            except Exception as e:
                skipped.append(SkippedRecord(source_row_index, f"Invalid questions JSON: {e}"))
                continue
        else:
            questions = questions_raw
        if isinstance(gold_raw, str):
            try:
                gold = json.loads(gold_raw)
            except Exception:
                gold = {}
        else:
            gold = gold_raw

        for qname, qdata in questions.items():
            if limit is not None and len(records) >= limit:
                break
            criteria = qdata.get("criteria")
            if not criteria:
                continue
            if isinstance(criteria, dict):
                choices = tuple(Choice(str(lbl), str(txt)) for lbl, txt in criteria.items())
            elif isinstance(criteria, list):
                choices = tuple(Choice(str(i), str(txt)) for i, txt in enumerate(criteria))
            else:
                continue

            gold_info = gold.get(qname, {})
            gold_label = gold_info.get("label")
            if gold_label is None and "probabilities" in gold_info:
                probs = gold_info["probabilities"]
                if probs:
                    gold_label = max(probs.items(), key=lambda x: x[1])[0]
            if gold_label is None:
                continue
            gold_label = str(gold_label)
            choice_map = {c.label: c.text for c in choices}
            if gold_label not in choice_map:
                continue

            records.append(
                ArcRecord(
                    example_id=f"td_{dataset_config}__{split}__{source_row_index:06d}_{qname}",
                    dataset_id="LocalLLaMA/typed-decisions",
                    dataset_config=dataset_config,
                    split=split,
                    source_row_index=row_counter,
                    question=qdata.get("instructions") or qname,
                    choices=choices,
                    gold_answer_label=gold_label,
                    gold_answer_text=choice_map[gold_label],
                    metadata={"state": state, "question_name": qname},
                )
            )
            row_counter += 1

    return DatasetLoadResult(
        records=tuple(records),
        skipped=tuple(skipped),
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        datasets_version=version("datasets"),
    )


def load_proofwriter_dataset(
    *,
    dataset_config: str = "default",
    split: str = "train",
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Load tasksource/proofwriter with True/False/Unknown choices."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "ProofWriter loading requires the 'datasets' package. Install dependencies first."
        ) from error

    try:
        dataset = load_dataset("tasksource/proofwriter", split=split, streaming=True)
    except Exception:
        dataset = load_dataset("tasksource/proofwriter", split=split)
    records: list[ArcRecord] = []
    skipped: list[SkippedRecord] = []
    pw_choices = (Choice("True", "True"), Choice("False", "False"), Choice("Unknown", "Unknown"))
    choice_map = {c.label: c.text for c in pw_choices}

    for source_row_index, row in enumerate(dataset):
        if limit is not None and len(records) >= limit:
            break
        theory = str(row.get("theory") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        all_proofs = str(row.get("allProofs") or "").strip()
        if not question or not answer:
            skipped.append(SkippedRecord(source_row_index, "Missing question or answer"))
            continue
        if answer not in choice_map:
            skipped.append(SkippedRecord(source_row_index, f"Invalid answer {answer!r}"))
            continue

        records.append(
            ArcRecord(
                example_id=f"proofwriter_{dataset_config}__{split}__{source_row_index:06d}",
                dataset_id="tasksource/proofwriter",
                dataset_config=dataset_config,
                split=split,
                source_row_index=source_row_index,
                question=question,
                choices=pw_choices,
                gold_answer_label=answer,
                gold_answer_text=choice_map[answer],
                metadata={"theory": theory, "allProofs": all_proofs},
            )
        )

    return DatasetLoadResult(
        records=tuple(records),
        skipped=tuple(skipped),
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        datasets_version=version("datasets"),
    )


def load_logiqa_dataset(
    *,
    dataset_config: str = "default",
    split: str = "train",
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Load lucasmccabe/logiqa (with parquet fallback for modern datasets versions)."""
    records: list[ArcRecord] = []
    skipped: list[SkippedRecord] = []

    dataset_rows = None
    try:
        from datasets import load_dataset
        ds = load_dataset("lucasmccabe/logiqa", split=split)
        dataset_rows = ds
    except Exception:
        import pandas as pd
        url = f"https://huggingface.co/api/datasets/lucasmccabe/logiqa/parquet/default/{split}/0.parquet"
        df = pd.read_parquet(url)
        dataset_rows = df.to_dict(orient="records")

    for source_row_index, row in enumerate(dataset_rows):
        if limit is not None and len(records) >= limit:
            break
        context = str(row.get("context") or "").strip()
        query = str(row.get("query") or "").strip()
        options = row.get("options")
        correct_option = row.get("correct_option")
        if query == "" or options is None or correct_option is None:
            skipped.append(SkippedRecord(source_row_index, "Missing required logiqa field"))
            continue
        try:
            correct_idx = int(correct_option)
        except (ValueError, TypeError):
            skipped.append(SkippedRecord(source_row_index, f"Invalid correct_option {correct_option!r}"))
            continue
        choices = tuple(Choice(str(i), str(opt).strip()) for i, opt in enumerate(options))
        choice_map = {c.label: c.text for c in choices}
        if str(correct_idx) not in choice_map:
            skipped.append(SkippedRecord(source_row_index, f"correct_option {correct_idx} out of range"))
            continue

        records.append(
            ArcRecord(
                example_id=f"logiqa_{dataset_config}__{split}__{source_row_index:06d}",
                dataset_id="lucasmccabe/logiqa",
                dataset_config=dataset_config,
                split=split,
                source_row_index=source_row_index,
                question=query,
                choices=choices,
                gold_answer_label=str(correct_idx),
                gold_answer_text=choice_map[str(correct_idx)],
                metadata={"context": context},
            )
        )

    return DatasetLoadResult(
        records=tuple(records),
        skipped=tuple(skipped),
        dataset_fingerprint=None,
        datasets_version=None,
    )


def load_cot_dataset(
    dataset_name: str,
    *,
    dataset_config: str,
    split: str,
    strict: bool = False,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Unified loader dispatching across all supported CoT generation datasets."""
    normalized = dataset_name.lower().replace("-", "_")
    if normalized == "arc":
        return load_arc_dataset(dataset_config=dataset_config, split=split, strict=strict, limit=limit)
    if normalized in ("typed_decisions", "typed-decisions"):
        return load_typed_decisions_dataset(dataset_config=dataset_config, split=split, strict=strict, limit=limit)
    if normalized == "proofwriter":
        return load_proofwriter_dataset(dataset_config=dataset_config, split=split, strict=strict, limit=limit)
    if normalized == "logiqa":
        return load_logiqa_dataset(dataset_config=dataset_config, split=split, strict=strict, limit=limit)
    raise ValueError(
        f"Unsupported dataset {dataset_name!r}; choose one of: arc, typed_decisions, proofwriter, logiqa."
    )
