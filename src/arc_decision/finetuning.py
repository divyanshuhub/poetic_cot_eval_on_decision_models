"""Prepare ARC System One data and delegate fine-tuning to each native model tool."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .dataset import DATASET_ID, DatasetLoadResult, load_arc_dataset, load_cot_dataset
from .models import ArcRecord
from .storage import write_json_atomic, write_jsonl_atomic, write_yaml_atomic


FINE_TUNE_MODELS = ("laya", "kev", "clef")
DIFFICULTY_CONFIGS = {"easy": "ARC-Easy", "challenge": "ARC-Challenge"}
SUPPORTED_DATASETS = ("arc", "typed_decisions", "proofwriter", "logiqa")
DATASET_DEFAULT_CONFIGS = {
    "arc": "easy",
    "typed_decisions": "all",
    "proofwriter": "default",
    "logiqa": "default",
}
DATASET_CONFIG_ALIASES = {
    "arc": {
        "easy": "ARC-Easy",
        "challenge": "ARC-Challenge",
        "arc-easy": "ARC-Easy",
        "arc_easy": "ARC-Easy",
        "arc-challenge": "ARC-Challenge",
        "arc_challenge": "ARC-Challenge",
    },
    "typed_decisions": {
        "all": "all",
        "customer_service": "customer_service",
        "customer-service": "customer_service",
        "invoice_processing": "invoice_processing",
        "invoice-processing": "invoice_processing",
        "security_incidents": "security_incidents",
        "security-incidents": "security_incidents",
        "agent_trace_observability": "agent_trace_observability",
        "agent-trace-observability": "agent_trace_observability",
    },
    "proofwriter": {"default": "default"},
    "logiqa": {"default": "default"},
}
DATASET_CHOICE_INSTRUCTIONS = {
    "arc": "Select the correct answer for this ARC science question.",
    "logiqa": "Select the correct answer for this logical reasoning question.",
    "proofwriter": "Determine whether the statement is True, False, or Unknown based on the theory.",
    "typed_decisions": "Select the best decision option based on the scenario state.",
}
DATASET_IDS = {
    "arc": "allenai/ai2_arc",
    "typed_decisions": "LocalLLaMA/typed-decisions",
    "proofwriter": "tasksource/proofwriter",
    "logiqa": "lucasmccabe/logiqa",
}
ARC_QUESTION_ID = "arc_answer"
ARC_CHOICE_INSTRUCTIONS = "Select the correct answer for this ARC science question."
DEFAULT_BASE_MODELS = {
    "laya": "convaiinnovations/laya",
    "kev": "jaredpalmer/kev-4b",
    "clef": "Cloudflare/clef-flash",
}
DEFAULT_KEV_BACKBONES = {"jaredpalmer/kev-4b": "Qwen/Qwen3.5-4B-Base"}


@dataclass(frozen=True)
class FineTuneSplits:
    """Normalized records used for native train and validation artifacts."""

    difficulty: str
    dataset_config: str
    train: tuple[ArcRecord, ...]
    validation: tuple[ArcRecord, ...]
    validation_source: str
    train_skipped: int
    validation_skipped: int
    dataset_name: str = "arc"


@dataclass(frozen=True)
class FineTunePaths:
    """All artifacts for one model and dataset configuration."""

    root: Path
    data_dir: Path
    train_jsonl: Path
    validation_jsonl: Path
    checkpoint_dir: Path
    monitor_dir: Path
    manifest_path: Path
    native_commands_path: Path
    clef_config_path: Path


@dataclass(frozen=True)
class NativeCommand:
    """A native training or monitoring command and its working directory."""

    name: str
    argv: tuple[str, ...]
    cwd: Path | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_finetune_dataset_config(
    dataset_name: str,
    dataset_config: str | None = None,
    difficulty: str | None = None,
) -> str:
    """Resolve and validate the canonical configuration for any supported dataset."""
    norm_dataset = dataset_name.strip().lower().replace("-", "_")
    if norm_dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(SUPPORTED_DATASETS)}.")

    if norm_dataset == "arc":
        raw = dataset_config or difficulty or DATASET_DEFAULT_CONFIGS["arc"]
        norm_val = raw.strip().lower()
        aliases = DATASET_CONFIG_ALIASES["arc"]
        if norm_val in aliases:
            return aliases[norm_val]
        raise ValueError("difficulty must be one of: easy, challenge.")

    valid_configs = DATASET_CONFIG_ALIASES.get(norm_dataset, {})
    if dataset_config is not None:
        norm_val = dataset_config.strip().lower()
        if norm_val in valid_configs:
            return valid_configs[norm_val]
        available = sorted(list(set(valid_configs.values())))
        raise ValueError(
            f"BuilderConfig {dataset_config!r} not found for {norm_dataset}. "
            f"Available: {available}"
        )

    return DATASET_DEFAULT_CONFIGS[norm_dataset]


def dataset_config_for_difficulty(difficulty: str) -> str:
    """Map the public fine-tuning flag to the canonical Hugging Face ARC config."""
    return resolve_finetune_dataset_config("arc", difficulty=difficulty)


def _validation_unavailable(error: BaseException) -> bool:
    text = str(error).lower()
    return "validation" in text and any(
        marker in text for marker in ("unknown", "not found", "not available", "does not exist", "not contain")
    )


def _hold_out_validation(
    records: tuple[ArcRecord, ...],
    *,
    fraction: float,
    seed: int,
) -> tuple[tuple[ArcRecord, ...], tuple[ArcRecord, ...]]:
    if len(records) < 2:
        raise ValueError("Train data needs at least two valid records to create a validation holdout.")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    count = max(1, min(len(records) - 1, round(len(records) * fraction)))
    indexes = list(range(len(records)))
    random.Random(seed).shuffle(indexes)
    validation_indexes = set(indexes[:count])
    train = tuple(record for index, record in enumerate(records) if index not in validation_indexes)
    validation = tuple(record for index, record in enumerate(records) if index in validation_indexes)
    return train, validation


def load_finetune_splits(
    dataset_name: str = "arc",
    dataset_config: str | None = None,
    difficulty: str | None = None,
    *,
    limit: int | None = None,
    validation_limit: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 7,
    loader: Callable[..., DatasetLoadResult] | None = None,
) -> FineTuneSplits:
    """Load train split plus validation monitoring, falling back to a deterministic train holdout."""
    norm_dataset = dataset_name.strip().lower().replace("-", "_")
    resolved_config = resolve_finetune_dataset_config(norm_dataset, dataset_config, difficulty)
    public_difficulty = (
        ("easy" if "easy" in resolved_config.lower() else "challenge")
        if norm_dataset == "arc"
        else resolved_config
    )

    load_func = loader if loader is not None else (
        (lambda **kw: load_cot_dataset(norm_dataset, **kw))
        if norm_dataset != "arc"
        else load_arc_dataset
    )

    train_result = load_func(dataset_config=resolved_config, split="train", strict=True, limit=limit)
    if not train_result.records:
        raise ValueError(f"{dataset_name} train split has no valid records for fine-tuning.")

    try:
        validation_result = load_func(
            dataset_config=resolved_config,
            split="validation",
            strict=True,
            limit=validation_limit,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        if not _validation_unavailable(error):
            raise
        train_records, validation_records = _hold_out_validation(
            train_result.records,
            fraction=validation_fraction,
            seed=seed,
        )
        return FineTuneSplits(
            dataset_name=norm_dataset,
            difficulty=public_difficulty,
            dataset_config=resolved_config,
            train=train_records,
            validation=validation_records,
            validation_source="train_holdout",
            train_skipped=len(train_result.skipped),
            validation_skipped=0,
        )

    if validation_result.records:
        return FineTuneSplits(
            dataset_name=norm_dataset,
            difficulty=public_difficulty,
            dataset_config=resolved_config,
            train=train_result.records,
            validation=validation_result.records,
            validation_source="validation",
            train_skipped=len(train_result.skipped),
            validation_skipped=len(validation_result.skipped),
        )

    train_records, validation_records = _hold_out_validation(
        train_result.records,
        fraction=validation_fraction,
        seed=seed,
    )
    return FineTuneSplits(
        dataset_name=norm_dataset,
        difficulty=public_difficulty,
        dataset_config=resolved_config,
        train=train_records,
        validation=validation_records,
        validation_source="train_holdout",
        train_skipped=len(train_result.skipped),
        validation_skipped=len(validation_result.skipped),
    )


def load_arc_finetune_splits(
    difficulty: str,
    *,
    limit: int | None = None,
    validation_limit: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 7,
    loader: Callable[..., DatasetLoadResult] = load_arc_dataset,
) -> FineTuneSplits:
    """Load ARC train plus validation, falling back to a deterministic train holdout."""
    return load_finetune_splits(
        dataset_name="arc",
        difficulty=difficulty,
        limit=limit,
        validation_limit=validation_limit,
        validation_fraction=validation_fraction,
        seed=seed,
        loader=loader,
    )


def systemone_state_for_record(record: ArcRecord) -> dict[str, Any]:
    norm_id = (record.dataset_id or "").lower()
    if "proofwriter" in norm_id:
        theory = str(record.metadata.get("theory") or "").strip()
        return {"theory": theory, "question": record.question}
    if "logiqa" in norm_id:
        context = str(record.metadata.get("context") or "").strip()
        return {"context": context, "question": record.question}
    if "typed" in norm_id or "decisions" in norm_id:
        st = record.metadata.get("state")
        if isinstance(st, dict):
            return {**st, "question": record.question}
        if isinstance(st, str) and st.strip():
            return {"state": st, "question": record.question}
        return {"question": record.question}
    return {"question": record.question}


def _arc_state(record: ArcRecord) -> dict[str, Any]:
    return systemone_state_for_record(record)


def _choice_question(record: ArcRecord) -> dict[str, Any]:
    norm_id = (record.dataset_id or "").lower()
    instructions = ARC_CHOICE_INSTRUCTIONS
    if "proofwriter" in norm_id:
        instructions = DATASET_CHOICE_INSTRUCTIONS["proofwriter"]
    elif "logiqa" in norm_id:
        instructions = DATASET_CHOICE_INSTRUCTIONS["logiqa"]
    elif "typed" in norm_id or "decisions" in norm_id:
        instructions = DATASET_CHOICE_INSTRUCTIONS["typed_decisions"]
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {choice.label: choice.text for choice in record.choices},
    }


def laya_training_row(record: ArcRecord) -> dict[str, Any]:
    """Create Laya's native `{state, questions, expected}` fine-tuning row."""
    return {
        "state": _arc_state(record),
        "questions": {ARC_QUESTION_ID: _choice_question(record)},
        "expected": {ARC_QUESTION_ID: record.gold_answer_label},
    }


def kev_training_row(record: ArcRecord) -> dict[str, Any]:
    """Create Kev's labelled System One request consumed by `kev.train --data`."""
    question = _choice_question(record)
    question["label"] = record.gold_answer_label
    return {"state": _arc_state(record), "questions": {ARC_QUESTION_ID: question}}


def clef_training_row(record: ArcRecord) -> dict[str, Any]:
    """Create Clef's native request plus its expected answer under `labels`."""
    return {
        "id": record.example_id,
        "state": _arc_state(record),
        "questions": {ARC_QUESTION_ID: _choice_question(record)},
        "labels": {ARC_QUESTION_ID: record.gold_answer_label},
    }


def training_row(model: str, record: ArcRecord) -> dict[str, Any]:
    """Render one normalized ARC record in the selected native training format."""
    normalized = model.lower()
    if normalized == "laya":
        return laya_training_row(record)
    if normalized == "kev":
        return kev_training_row(record)
    if normalized == "clef":
        return clef_training_row(record)
    raise ValueError(f"Unsupported model {model!r}; choose one of: {', '.join(FINE_TUNE_MODELS)}.")


def fine_tune_paths(
    model: str,
    difficulty: str | None = None,
    output_dir: str | Path | None = None,
    *,
    dataset: str = "arc",
    dataset_config: str | None = None,
) -> FineTunePaths:
    """Choose a distinct run directory per model, dataset, and difficulty/config by default."""
    normalized_model = model.lower()
    if normalized_model not in FINE_TUNE_MODELS:
        raise ValueError(f"Unsupported model {model!r}; choose one of: {', '.join(FINE_TUNE_MODELS)}.")
    norm_dataset = dataset.lower().replace("-", "_")
    if norm_dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose one of: {', '.join(SUPPORTED_DATASETS)}.")

    project_root = _project_root()
    if output_dir is not None:
        requested = Path(output_dir)
    elif norm_dataset == "arc":
        resolved_config = resolve_finetune_dataset_config(norm_dataset, dataset_config, difficulty)
        slug = "easy" if "easy" in resolved_config.lower() else "challenge"
        requested = Path("outputs") / "finetunes" / f"{normalized_model}_arc_{slug}"
    else:
        resolved_config = resolve_finetune_dataset_config(norm_dataset, dataset_config, difficulty)
        config_slug = resolved_config.lower().replace("-", "_")
        requested = Path("outputs") / "finetunes" / f"{normalized_model}_{norm_dataset}_{config_slug}"

    root = (requested if requested.is_absolute() else project_root / requested).resolve()
    data_dir = root / "data"
    return FineTunePaths(
        root=root,
        data_dir=data_dir,
        train_jsonl=data_dir / "train.jsonl",
        validation_jsonl=data_dir / "validation.jsonl",
        checkpoint_dir=root / "checkpoint",
        monitor_dir=root / "validation_monitoring",
        manifest_path=root / "dataset_manifest.json",
        native_commands_path=root / "native_commands.json",
        clef_config_path=root / "clef_finetune.yaml",
    )


def clean_run_directory(paths: FineTunePaths) -> None:
    """Clean existing checkpoint, data, and config artifacts when --replace is specified."""
    import shutil

    for path in (paths.checkpoint_dir, paths.data_dir, paths.monitor_dir):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for file_path in (paths.manifest_path, paths.native_commands_path, paths.clef_config_path):
        if file_path.exists():
            file_path.unlink()


def write_native_training_data(model: str, splits: FineTuneSplits, paths: FineTunePaths) -> None:
    """Write train/validation JSONL in the selected model's native System One shape."""
    normalized_model = model.lower()
    write_jsonl_atomic(paths.train_jsonl, (training_row(normalized_model, record) for record in splits.train))
    write_jsonl_atomic(paths.validation_jsonl, (training_row(normalized_model, record) for record in splits.validation))
    dataset_name = getattr(splits, "dataset_name", "arc")
    write_json_atomic(
        paths.manifest_path,
        {
            "dataset_name": dataset_name,
            "dataset_id": DATASET_IDS.get(dataset_name, DATASET_ID),
            "dataset_config": splits.dataset_config,
            "difficulty": splits.difficulty,
            "model": normalized_model,
            "train_split": "train",
            "validation_source": splits.validation_source,
            "train_records": len(splits.train),
            "validation_records": len(splits.validation),
            "train_skipped": splits.train_skipped,
            "validation_skipped": splits.validation_skipped,
            "question_id": ARC_QUESTION_ID,
            "question_instructions": (
                DATASET_CHOICE_INSTRUCTIONS.get(dataset_name, ARC_CHOICE_INSTRUCTIONS)
            ),
            "train_jsonl": str(paths.train_jsonl),
            "validation_jsonl": str(paths.validation_jsonl),
        },
    )


def write_clef_config(
    paths: FineTunePaths,
    *,
    base_model: str,
    seed: int,
    max_steps: int,
) -> None:
    """Write a YAML-compatible JSON config for the native `clef-finetune train` CLI."""
    if max_steps < 1:
        raise ValueError("clef_max_steps must be at least one.")
    write_yaml_atomic(
        paths.clef_config_path,
        {
            "model": {"path": base_model, "revision": None, "dtype": "bfloat16"},
            "data": {
                "train": str(paths.train_jsonl),
                "eval": str(paths.validation_jsonl),
                "max_length": 8192,
            },
            "augment": {
                "shuffle_questions": True,
                "question_subset_prob": 0.0,
                "paraphrase_prob": 0.0,
                "rename_option_prob": 0.0,
                "criteria_dropout_prob": 0.0,
            },
            "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
            "train": {
                "seed": seed,
                "max_steps": max_steps,
                "grad_accum": 16,
                "lr": 1.0e-4,
                "head_lr": 5.0e-5,
                "warmup_steps": min(20, max_steps),
                "label_smoothing": 0.05,
                "brier_weight": 1.0,
                "ordinal_weight": 0.0,
                "train_head": True,
                "head_dtype": "float32",
                "gradient_checkpointing": True,
                "save_every": min(100, max_steps),
            },
            "output_dir": str(paths.checkpoint_dir),
        },
    )


def _kev_prefix(kev_python: str | None, kev_uv_command: str) -> list[str]:
    return [kev_python] if kev_python else [kev_uv_command, "run", "python"]


def build_native_commands(
    model: str,
    paths: FineTunePaths,
    *,
    base_model: str,
    device: str,
    native_args: Sequence[str] = (),
    laya_command: str = "laya-train",
    kev_python: str | None = None,
    kev_uv_command: str = "uv",
    kev_repo_dir: Path | None = None,
    kev_base_model: str | None = None,
    clef_command: str = "clef-finetune",
    clef_max_steps: int = 500,
) -> list[NativeCommand]:
    """Build native training and validation-monitoring commands without running them."""
    normalized = model.lower()
    extra = list(native_args)
    if normalized == "laya":
        return [
            NativeCommand(
                "laya-train",
                tuple(
                    [
                        laya_command,
                        "--data",
                        str(paths.train_jsonl),
                        "--eval",
                        str(paths.validation_jsonl),
                        "--base",
                        base_model,
                        "--out",
                        str(paths.checkpoint_dir),
                        "--device",
                        device,
                        *extra,
                    ]
                ),
            )
        ]
    if normalized == "kev":
        prefix = _kev_prefix(kev_python, kev_uv_command)
        kev_backbone = kev_base_model or DEFAULT_KEV_BACKBONES.get(base_model)
        if not kev_backbone:
            raise ValueError(
                f"Could not infer Kev's backbone for init checkpoint {base_model!r}; "
                "pass --kev-base-model explicitly."
            )
        return [
            NativeCommand(
                "kev.train",
                tuple(
                    [
                        *prefix,
                        "-m",
                        "kev.train",
                        "--base",
                        kev_backbone,
                        "--data",
                        str(paths.train_jsonl),
                        "--init_from",
                        base_model,
                        "--device",
                        device,
                        "--out",
                        str(paths.checkpoint_dir),
                        *( ["--dtype", "bf16"] if device == "cuda" else [] ),
                        *extra,
                    ]
                ),
                cwd=kev_repo_dir,
            ),
            NativeCommand(
                "kev.benchmark",
                tuple(
                    [
                        *prefix,
                        "-m",
                        "kev.benchmark",
                        "--run",
                        str(paths.checkpoint_dir),
                        "--data",
                        str(paths.validation_jsonl),
                        "--out",
                        str(paths.monitor_dir),
                    ]
                ),
                cwd=kev_repo_dir,
            ),
        ]
    if normalized == "clef":
        checkpoint = paths.checkpoint_dir / f"checkpoint-{clef_max_steps}"
        return [
            NativeCommand(
                "clef-finetune train",
                tuple(
                    [
                        clef_command,
                        "train",
                        str(paths.clef_config_path),
                        "--device",
                        device,
                        "--output-dir",
                        str(paths.checkpoint_dir),
                        *extra,
                    ]
                ),
            ),
            NativeCommand(
                "clef-finetune eval",
                tuple(
                    [
                        clef_command,
                        "eval",
                        "--model",
                        base_model,
                        "--data",
                        str(paths.validation_jsonl),
                        "--checkpoint",
                        str(checkpoint),
                        "--compare-base",
                        "--output-dir",
                        str(paths.monitor_dir),
                        "--device",
                        device,
                    ]
                ),
            ),
        ]
    raise ValueError(f"Unsupported model {model!r}; choose one of: {', '.join(FINE_TUNE_MODELS)}.")


def _command_record(command: NativeCommand) -> dict[str, Any]:
    return {"name": command.name, "argv": list(command.argv), "cwd": str(command.cwd) if command.cwd else None}


def run_native_commands(commands: Sequence[NativeCommand], *, dry_run: bool) -> int:
    """Run native tools in sequence; monitoring is skipped if training fails."""
    import shutil
    for command in commands:
        argv = list(command.argv)
        if argv and not shutil.which(argv[0]):
            candidate = Path(sys.executable).parent / argv[0]
            if candidate.is_file():
                argv[0] = str(candidate)
        print(f"[{command.name}] {shlex.join(argv)}", flush=True)
        if dry_run:
            continue
        try:
            result = subprocess.run(argv, cwd=command.cwd, check=False)
        except FileNotFoundError as error:
            print(f"Could not start {command.name}: {error}", file=sys.stderr)
            return 127
        if result.returncode:
            return result.returncode
    return 0


def _default_kev_repo_dir() -> Path:
    configured = os.environ.get("KEV_REPO_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return _project_root() / "kev"


def _build_parser(fixed_model: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Laya, Kev, or Clef decision models on reasoning datasets in each model's native format."
    )
    if fixed_model is None:
        parser.add_argument("--model", choices=FINE_TUNE_MODELS, required=True)
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset",
        choices=SUPPORTED_DATASETS,
        default="arc",
        help="Target reasoning dataset: arc, typed_decisions, proofwriter, or logiqa (default: arc).",
    )
    parser.add_argument(
        "--difficulty",
        choices=tuple(DIFFICULTY_CONFIGS),
        help="ARC difficulty setting: easy or challenge. (ARC-specific; ignored for non-ARC datasets unless mapped).",
    )
    parser.add_argument(
        "--dataset-config",
        "--dataset-type",
        dest="dataset_config",
        help="Dataset configuration / subset (defaults based on dataset, e.g. 'all' for typed_decisions, 'default' for logiqa).",
    )
    parser.add_argument(
        "--replace",
        "--overwrite",
        action="store_true",
        dest="replace",
        help="Replace existing fine-tuned model checkpoint / output directory.",
    )
    parser.add_argument("--output-dir", help="Run root; defaults to outputs/finetunes/<model>_<dataset>_<config>.")
    parser.add_argument(
        "--base-model",
        help="Native model/checkpoint ID to initialize/warm-start from (replaces the base model with this checkpoint).",
    )
    parser.add_argument("--limit", type=int, help="Optional cap on train records for a smoke run.")
    parser.add_argument("--validation-limit", type=int, help="Optional cap on validation records.")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Holdout fraction if validation is absent.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="Write native JSONL/config files without invoking a trainer.")
    parser.add_argument("--dry-run", action="store_true", help="Write artifacts and print native commands without invoking them.")
    parser.add_argument(
        "--native-arg",
        action="append",
        default=[],
        help="Extra argument for the selected native training command; repeat once per argument.",
    )
    parser.add_argument("--laya-command", default="laya-train")
    parser.add_argument("--kev-repo-dir", type=Path, default=_default_kev_repo_dir())
    parser.add_argument("--kev-python", help="Python executable with Kev installed; otherwise use `uv run python`.")
    parser.add_argument("--kev-uv-command", default="uv")
    parser.add_argument(
        "--kev-base-model",
        help="Underlying backbone passed to Kev's --base; inferred for the default Kev-4B checkpoint.",
    )
    parser.add_argument("--clef-command", default="clef-finetune")
    parser.add_argument("--clef-max-steps", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None, *, fixed_model: str | None = None) -> int:
    """Prepare native artifacts and delegate to one selected native trainer."""
    parser = _build_parser(fixed_model)
    args = parser.parse_args(argv)
    model = fixed_model or args.model
    assert model is not None
    dataset_name = args.dataset.lower().replace("-", "_")
    if dataset_name == "arc" and not args.difficulty and not args.dataset_config:
        parser.error("--difficulty (easy or challenge) or --dataset-config is required when --dataset is arc.")

    difficulty = args.difficulty
    dataset_config = args.dataset_config
    if dataset_name == "arc" and difficulty is None and dataset_config in DIFFICULTY_CONFIGS:
        difficulty = dataset_config

    if dataset_name != "arc" and difficulty is not None:
        resolved_preview = resolve_finetune_dataset_config(dataset_name, dataset_config, difficulty)
        print(
            f"[INFO] --difficulty ({difficulty}) is an ARC-specific option and does not apply to {dataset_name}. "
            f"Using dataset configuration {resolved_preview!r} (override with --dataset-config).",
            file=sys.stderr,
        )

    try:
        paths = fine_tune_paths(
            model,
            difficulty=difficulty,
            output_dir=args.output_dir,
            dataset=dataset_name,
            dataset_config=dataset_config,
        )
        if args.replace:
            clean_run_directory(paths)

        splits = load_finetune_splits(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            difficulty=difficulty,
            limit=args.limit,
            validation_limit=args.validation_limit,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        write_native_training_data(model, splits, paths)
        base_model = args.base_model or DEFAULT_BASE_MODELS[model]
        kev_base_model = args.kev_base_model
        if model == "kev" and not kev_base_model:
            kev_base_model = DEFAULT_KEV_BACKBONES.get(base_model)
        if model == "clef":
            write_clef_config(paths, base_model=base_model, seed=args.seed, max_steps=args.clef_max_steps)
        kev_repo_dir = args.kev_repo_dir.expanduser().resolve()
        if model == "kev" and not (args.prepare_only or args.dry_run) and not kev_repo_dir.is_dir():
            raise FileNotFoundError(
                f"Kev repository not found: {kev_repo_dir}. Clone jaredpalmer/kev there or pass --kev-repo-dir."
            )
        commands = build_native_commands(
            model,
            paths,
            base_model=base_model,
            device=args.device,
            native_args=args.native_arg,
            laya_command=args.laya_command,
            kev_python=args.kev_python,
            kev_uv_command=args.kev_uv_command,
            kev_repo_dir=kev_repo_dir if model == "kev" else None,
            kev_base_model=kev_base_model,
            clef_command=args.clef_command,
            clef_max_steps=args.clef_max_steps,
        )
        write_json_atomic(
            paths.native_commands_path,
            {
                "model": model,
                "dataset": dataset_name,
                "difficulty": splits.difficulty,
                "dataset_config": splits.dataset_config,
                "base_model": base_model,
                "kev_base_model": kev_base_model,
                "commands": [_command_record(command) for command in commands],
            },
        )
        print(
            json.dumps(
                {
                    "model": model,
                    "dataset": dataset_name,
                    "difficulty": splits.difficulty,
                    "dataset_config": splits.dataset_config,
                    "train_records": len(splits.train),
                    "validation_records": len(splits.validation),
                    "validation_source": splits.validation_source,
                    "run_dir": str(paths.root),
                    "checkpoint_dir": str(paths.checkpoint_dir),
                    "train_jsonl": str(paths.train_jsonl),
                    "validation_jsonl": str(paths.validation_jsonl),
                },
                indent=2,
            )
        )
        if args.prepare_only:
            return 0
        return run_native_commands(commands, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())