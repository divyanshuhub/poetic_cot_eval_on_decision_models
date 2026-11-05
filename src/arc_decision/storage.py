"""Append-only run storage with atomic derived artifact projections."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .config import ExperimentConfig
from .models import ArcRecord


_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|access[_-]?token|secret)$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact_secrets(value: Any) -> Any:
    """Recursively remove credential-like values before durable storage."""
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


def sanitized_error(error: BaseException | str) -> str:
    message = str(error)
    message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", message)
    return message[:2000]


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(redact_secrets(value), ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def write_yaml_atomic(path: Path, value: Any) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Writing YAML requires PyYAML. Install the project dependencies first.") from error
    _atomic_write(path, yaml.safe_dump(redact_secrets(value), sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    contents = "".join(canonical_json(redact_secrets(dict(row))) + "\n" for row in rows)
    _atomic_write(path, contents)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL in {path} at line {line_number}.") from error


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]], preferred_fields: Iterable[str] = ()) -> None:
    materialized = [redact_secrets(dict(row)) for row in rows]
    fields: list[str] = list(preferred_fields)
    seen = set(fields)
    for row in materialized:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow({field: _csv_value(row.get(field)) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class RunStore:
    """Owns one immutable output directory and its durable evidence."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.attempts_dir = self.root / "attempts"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def dataset_records_path(self) -> Path:
        return self.root / "dataset_records.jsonl"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.root / "dataset_manifest.json"

    def attempt_path(self, stage: str) -> Path:
        if stage not in {"gemini", "decision"}:
            raise ValueError(f"Unsupported attempt stage {stage!r}.")
        return self.attempts_dir / f"{stage}_attempts.jsonl"

    def initialize(self, config: ExperimentConfig) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(exist_ok=True)
        if not self.manifest_path.exists():
            write_json_atomic(
                self.manifest_path,
                {
                    "run_id": self.root.name,
                    "created_at": utc_now(),
                    "config": config.manifest_values(),
                    "schema_version": 1,
                },
            )

    def record_decision_model(self, config: ExperimentConfig) -> None:
        """Record the evaluation model once selected, without mixing models in one run."""
        self.initialize(config)
        manifest = read_json(self.manifest_path)
        manifest_config = manifest.setdefault("config", {})
        selected = {
            "decision_model": config.decision_model,
            "laya_finetuned_checkpoint": config.laya_finetuned_checkpoint,
        }
        if self.attempts("decision") and any(
            manifest_config.get(key) != value for key, value in selected.items()
        ):
            raise ValueError(
                "This run already has decision attempts for a different decision model or checkpoint; "
                "use a new --output-dir."
            )
        manifest_config.update(selected)
        write_json_atomic(self.manifest_path, manifest)

    def write_dataset(
        self,
        records: Iterable[ArcRecord],
        *,
        skipped: Iterable[Mapping[str, Any]] = (),
        dataset_fingerprint: str | None = None,
        datasets_version: str | None = None,
    ) -> None:
        materialized = list(records)
        serialized = [record.to_dict() for record in materialized]
        write_jsonl_atomic(self.dataset_records_path, serialized)
        write_json_atomic(
            self.dataset_manifest_path,
            {
                "created_at": utc_now(),
                "record_count": len(materialized),
                "skipped": list(skipped),
                "dataset_fingerprint": dataset_fingerprint,
                "datasets_version": datasets_version,
                "records_sha256": stable_hash(serialized),
            },
        )

    def load_records(self) -> list[ArcRecord]:
        if not self.dataset_records_path.exists():
            raise FileNotFoundError(
                f"No normalized dataset found at {self.dataset_records_path}. Run the prepare stage first."
            )
        return [ArcRecord.from_dict(row) for row in read_jsonl(self.dataset_records_path)]

    def append_attempt(self, stage: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """Durably append a redacted attempt record before updating a projection."""
        path = self.attempt_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = dict(record)
        value.setdefault("attempt_id", str(uuid.uuid4()))
        value.setdefault("timestamp", utc_now())
        value.setdefault("stage", stage)
        serialized = canonical_json(redact_secrets(value)) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        return value

    def attempts(self, stage: str) -> list[dict[str, Any]]:
        return list(read_jsonl(self.attempt_path(stage)))

    def completed_keys(self, stage: str) -> set[str]:
        return {
            str(item["cache_key"])
            for item in self.attempts(stage)
            if item.get("terminal") is True
            and item.get("cache_key")
            and item.get("status") != "failed"
        }

    def latest_projections(self, stage: str) -> list[dict[str, Any]]:
        """Return last terminal projection per cache key, preserving ledger order."""
        by_key: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in self.attempts(stage):
            cache_key = item.get("cache_key")
            projection = item.get("projection")
            if not item.get("terminal") or not cache_key or not isinstance(projection, Mapping):
                continue
            key = str(cache_key)
            if key not in by_key:
                order.append(key)
            by_key[key] = dict(projection)
        return [by_key[key] for key in order]
