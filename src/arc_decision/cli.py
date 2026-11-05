"""Command-line entry point for the ARC reasoning-style transfer experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .analysis import analyze_run
from .config import DECISION_MODELS, POEM_FORMS, SPLITS, ExperimentConfig
from .dataset import load_arc_dataset
from .decision_client import create_decision_client
from .evaluation import evaluate_decisions
from .gemini_client import GoogleGenAIClient
from .generation import generate_rationales
from .storage import RunStore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_dir(value: str | Path) -> Path:
    project_root = _project_root().resolve()
    requested = Path(value)
    output_dir = (requested if requested.is_absolute() else project_root / requested).resolve()
    try:
        output_dir.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"Output directory must be inside the project folder: {project_root}.") from error
    if output_dir == project_root:
        raise ValueError(f"Output directory must be a subdirectory of the project folder: {project_root}.")
    return output_dir


def _resolve_laya_checkpoint(value: str | Path | None) -> str | None:
    if value is None:
        return None
    checkpoint = str(value).strip()
    if not checkpoint:
        raise ValueError("--laya-finetuned-checkpoint cannot be empty.")
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    return str(path.resolve())


def _load_defaults(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("The CLI requires PyYAML. Install the project dependencies first.") from error
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping.")
    return data


def _value(args: argparse.Namespace, defaults: dict[str, Any], name: str, fallback: Any) -> Any:
    parsed = getattr(args, name, None)
    return parsed if parsed is not None else defaults.get(name, fallback)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    defaults = _load_defaults(Path(args.config))
    output_dir = _value(args, defaults, "output_dir", "outputs/arc_easy_test")
    if output_dir is None:
        raise ValueError("Set output_dir in the YAML config or pass --output-dir.")
    return ExperimentConfig(
        output_dir=_resolve_output_dir(output_dir),
        dataset_config=_value(args, defaults, "dataset_config", "easy"),
        split=_value(args, defaults, "split", "test"),
        gemini_model=_value(args, defaults, "gemini_model", "gemini-3.8-flash"),
        gemini_api_key_env=_value(args, defaults, "gemini_api_key_env", "GEMINI_API_KEY"),
        decision_model=_value(args, defaults, "decision_model", "laya"),
        laya_finetuned_checkpoint=_resolve_laya_checkpoint(
            _value(args, defaults, "laya_finetuned_checkpoint", None)
        ),
        poem_form=_value(args, defaults, "poem_form", "tanka"),
        temperature=float(_value(args, defaults, "temperature", 0.0)),
        seed=_value(args, defaults, "seed", 7),
        gemini_requests_per_minute=float(_value(args, defaults, "gemini_requests_per_minute", 4.0)),
        limit=getattr(args, "limit", None),
        resume=bool(getattr(args, "resume", False)),
        strict=bool(getattr(args, "strict", False)),
        max_retries=int(_value(args, defaults, "max_retries", 2)),
        request_timeout=float(_value(args, defaults, "request_timeout", 60.0)),
        include_flagged_reasoning=bool(getattr(args, "include_flagged_reasoning", False)),
        force_regenerate=bool(getattr(args, "force_regenerate", False)),
    )


def _prepared_records(config: ExperimentConfig, store: RunStore) -> list:
    records = store.load_records()
    return records[: config.limit] if config.limit is not None else records


def _prepare(config: ExperimentConfig, store: RunStore) -> dict[str, Any]:
    if store.dataset_records_path.exists() and not config.resume and not config.force_regenerate:
        raise FileExistsError(
            "A normalized dataset already exists in this output directory. Use --resume or --force-regenerate explicitly."
        )
    if store.dataset_records_path.exists() and config.resume and not config.force_regenerate:
        records = store.load_records()
        return {"prepared": False, "record_count": len(records), "reason": "reused_existing_dataset"}
    result = load_arc_dataset(
        dataset_config=config.dataset_config,
        split=config.split,
        strict=config.strict,
        limit=config.limit,
    )
    store.initialize(config)
    store.write_dataset(
        result.records,
        skipped=[{"source_row_index": item.source_row_index, "reason": item.reason} for item in result.skipped],
        dataset_fingerprint=result.dataset_fingerprint,
        datasets_version=result.datasets_version,
    )
    return {"prepared": True, "record_count": len(result.records), "skipped_count": len(result.skipped)}


def _generate(config: ExperimentConfig, store: RunStore) -> dict[str, Any]:
    records = _prepared_records(config, store)
    client = GoogleGenAIClient(
        model=config.gemini_model,
        temperature=config.temperature,
        seed=config.seed,
        api_key_env=config.gemini_api_key_env,
    )
    generate_rationales(records, config=config, store=store, client=client)
    return {"generated_examples": len(records), "output_dir": str(store.root)}


def _evaluate(config: ExperimentConfig, store: RunStore) -> dict[str, Any]:
    records = _prepared_records(config, store)
    store.record_decision_model(config)
    client = create_decision_client(
        config.decision_model,
        timeout=config.request_timeout,
        laya_finetuned_checkpoint=config.laya_finetuned_checkpoint,
    )
    evaluate_decisions(records, config=config, store=store, client=client)
    return {"evaluated_examples": len(records), "decision_model": config.decision_model}


def _analyze(store: RunStore) -> dict[str, Any]:
    return analyze_run(store)


def _run(config: ExperimentConfig, store: RunStore) -> dict[str, Any]:
    preparation = _prepare(config, store)
    generation = _generate(config, store)
    evaluation = _evaluate(config, store)
    analysis = _analyze(store)
    return {"prepare": preparation, "generate": generation, "evaluate": evaluation, "analyze": analysis}


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_generation: bool, include_decision: bool) -> None:
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory inside arc_decision_reasoning; defaults to output_dir in the YAML config.",
    )
    parser.add_argument(
        "--config",
        default=str(_project_root() / "configs" / "experiment.yaml"),
        help="YAML defaults; explicit flags take precedence.",
    )
    parser.add_argument("--dataset-config", choices=("easy", "challenge", "normal"), default=None)
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Use a prefix for a low-cost smoke test.")
    parser.add_argument("--resume", action="store_true", help="Reuse exact terminal attempt keys from this run directory.")
    parser.add_argument("--strict", action="store_true", help="Fail preparation when any ARC row is invalid.")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--force-regenerate", action="store_true", help="Bypass completed cache keys deliberately.")
    if include_generation:
        parser.add_argument("--gemini-model", default=None)
        parser.add_argument("--poem-form", choices=POEM_FORMS, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--gemini-requests-per-minute", type=float, default=None)
    if include_decision:
        parser.add_argument("--decision-model", choices=DECISION_MODELS, default=None)
        parser.add_argument(
            "--laya-finetuned-checkpoint",
            default=None,
            help="Local Laya fine-tune output directory; relative paths resolve from arc_decision_reasoning/.",
        )
        parser.add_argument(
            "--include-flagged-reasoning",
            action="store_true",
            help="Opt in to evaluating hard-flagged Gemini rationales for a sensitivity analysis.",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC Gemini rationale and decision-model experiment.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, generate, evaluate in (
        ("prepare", False, False),
        ("generate", True, False),
        ("evaluate", False, True),
        ("analyze", False, False),
        ("run", True, True),
    ):
        command = subcommands.add_parser(name)
        _add_common_arguments(command, include_generation=generate, include_decision=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        store = RunStore(config.output_dir)
        handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "prepare": lambda: _prepare(config, store),
            "generate": lambda: _generate(config, store),
            "evaluate": lambda: _evaluate(config, store),
            "analyze": lambda: _analyze(store),
            "run": lambda: _run(config, store),
        }
        print(json.dumps(handlers[args.command](), ensure_ascii=True, indent=2, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
