#!/usr/bin/env python3
"""Combine evaluation summaries across decision models for each dataset.

This script aggregates per-run `summary.csv` (and optionally `token_latency_summary.csv`)
files from CoT decision model evaluations into unified dataset-level files and a master summary.

Examples:
    # Auto-discover all datasets and combine all decision models:
    python scripts/combine_cot_summaries.py

    # Combine only LogiQA decision models:
    python scripts/combine_cot_summaries.py --dataset logiqa

    # Combine specific summary.csv files into a custom output:
    python scripts/combine_cot_summaries.py \\
        --inputs outputs/cot_evaluations/logiqa_jev/summary.csv \\
                 outputs/cot_evaluations/logiqa_kev_ft/summary.csv \\
                 outputs/cot_evaluations/logiqa_laya_ft/summary.csv \\
        --output-file outputs/cot_evaluations/combined_summaries/logiqa_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Pattern to extract dataset and model name from directory names like logiqa_kev_ft or arc_easy_jev
MODEL_SUFFIX_PATTERN = re.compile(
    r"^(.*?)[_-](jev|kev|laya|clef)(_ft)?$", re.IGNORECASE
)

# Standard ordering for reasoning conditions
CONDITION_ORDER = [
    "question_only",
    "standard_reasoning",
    "poisoned_reasoning",
    "poetic_reasoning",
]


def resolve_base_paths() -> Tuple[Path, Path]:
    """Find default evaluations directory and default output directory."""
    script_path = Path(__file__).resolve()
    # If placed in arc_decision_reasoning/scripts/
    if script_path.parent.name == "scripts":
        arc_root = script_path.parent.parent
    else:
        arc_root = script_path.parent

    default_eval_dir = arc_root / "outputs" / "cot_evaluations"
    default_output_dir = default_eval_dir / "combined_summaries"
    return default_eval_dir, default_output_dir


def extract_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Extract dataset name, model name, and test status from run directory and manifest."""
    folder_name = run_dir.name
    is_test = folder_name.startswith(("smoke_", "test_"))

    dataset_name: Optional[str] = None
    decision_model: Optional[str] = None
    model_run: str = folder_name

    # Check manifest.json if present
    manifest_file = run_dir / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            cot_info = manifest_data.get("cot_csv_evaluation", {})
            config_info = manifest_data.get("config", {})

            dataset_name = cot_info.get("dataset_name")
            if not dataset_name:
                config_dataset = cot_info.get("dataset_config") or config_info.get("dataset_config")
                if config_dataset:
                    if config_dataset.lower() in ("arc-easy", "easy"):
                        dataset_name = "arc_easy"
                    elif config_dataset.lower() in ("arc-challenge", "challenge"):
                        dataset_name = "arc_challenge"
                    else:
                        dataset_name = str(config_dataset).lower()

            decision_model = cot_info.get("decision_model") or config_info.get("decision_model")
            limit_val = cot_info.get("limit") or config_info.get("limit")
            selected_rows = cot_info.get("selected_rows")
            if limit_val is not None and int(limit_val) <= 10:
                is_test = True
            elif selected_rows is not None and int(selected_rows) <= 10:
                is_test = True
        except Exception:
            pass

    # Fall back to folder name regex if dataset_name is not yet resolved
    m = MODEL_SUFFIX_PATTERN.match(folder_name)
    if m:
        if not dataset_name:
            dataset_name = m.group(1).lower()
        model_part = m.group(2).lower() + (m.group(3) or "").lower()
        model_run = model_part
        if not decision_model:
            decision_model = m.group(2).lower()
    else:
        if not dataset_name:
            dataset_name = folder_name

    return {
        "run_dir": run_dir,
        "run_name": folder_name,
        "model_run": model_run,
        "dataset_name": dataset_name,
        "decision_model": decision_model,
        "is_test": is_test,
    }


def find_dataset_runs(
    eval_dir: Path, include_tests: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """Discover and group all valid evaluation runs by dataset."""
    groups: Dict[str, List[Dict[str, Any]]] = {}

    if not eval_dir.exists() or not eval_dir.is_dir():
        print(f"Evaluation directory not found: {eval_dir}", file=sys.stderr)
        return groups

    for child in sorted(eval_dir.iterdir()):
        if not child.is_dir():
            continue
        summary_file = child / "summary.csv"
        if not summary_file.exists():
            continue

        meta = extract_run_metadata(child)
        if meta["is_test"] and not include_tests:
            continue

        dname = meta["dataset_name"] or "unknown_dataset"
        groups.setdefault(dname, []).append(meta)

    # Sort runs within each group: jev, kev, laya, clef, then others
    model_priority = {"jev": 1, "kev": 2, "kev_ft": 2, "laya": 3, "laya_ft": 3, "clef": 4, "clef_ft": 4}

    for dname, runs in groups.items():
        runs.sort(
            key=lambda r: (
                model_priority.get(r["model_run"], 99),
                r["model_run"],
                r["run_name"],
            )
        )

    return groups


def combine_summaries(
    runs: List[Dict[str, Any]],
    add_metadata: bool = True,
    summary_filename: str = "summary.csv",
) -> pd.DataFrame:
    """Read and concatenate summary CSVs from a list of run metadata dictionaries."""
    dfs: List[pd.DataFrame] = []
    for r in runs:
        csv_file = r["run_dir"] / summary_filename
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if add_metadata:
            # Prepend dataset and run_name
            cols_to_insert = [
                ("dataset", r.get("dataset_name", "")),
                ("run_name", r.get("run_name", "")),
                ("model_run", r.get("model_run", "")),
            ]
            for col_idx, (col_name, val) in enumerate(cols_to_insert):
                if col_name not in df.columns:
                    df.insert(col_idx, col_name, val)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_comparison_matrix(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build a condition x decision model matrix of accuracy and difference vs question_only."""
    if combined_df.empty or "condition" not in combined_df.columns:
        return pd.DataFrame()

    model_col = "model_run" if "model_run" in combined_df.columns else "decision_model"
    acc_col = "shared_completion_accuracy" if "shared_completion_accuracy" in combined_df.columns else "accuracy"
    diff_col = "question_only_difference_pp"

    piv_acc = combined_df.pivot(index="condition", columns=model_col, values=acc_col)
    existing_conds = [c for c in CONDITION_ORDER if c in piv_acc.index]
    other_conds = [c for c in piv_acc.index if c not in CONDITION_ORDER]
    ordered_conds = existing_conds + other_conds
    piv_acc = piv_acc.reindex(ordered_conds)

    piv_diff = None
    if diff_col in combined_df.columns:
        piv_diff = combined_df.pivot(index="condition", columns=model_col, values=diff_col)
        piv_diff = piv_diff.reindex(ordered_conds)

    matrix_df = pd.DataFrame(index=ordered_conds)
    matrix_df.index.name = "condition"

    for col in piv_acc.columns:
        matrix_df[f"{col}_accuracy_%"] = (piv_acc[col] * 100).round(2)
        if piv_diff is not None and col in piv_diff.columns:
            matrix_df[f"{col}_diff_pp"] = piv_diff[col].round(2)

    return matrix_df.reset_index()


def print_ascii_table(title: str, df: pd.DataFrame) -> None:
    """Print a clean Markdown table to console."""
    if df.empty:
        return
    print(f"\n### {title}\n")
    print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))


def main() -> int:
    default_eval_dir, default_output_dir = resolve_base_paths()

    parser = argparse.ArgumentParser(
        description="Combine evaluation summaries across decision models for each dataset."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=default_eval_dir,
        help=f"Directory containing evaluation runs (default: {default_eval_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to save combined summary files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        help="Optional dataset name(s) to process (e.g. logiqa, arc_easy, typed_decisions)",
    )
    parser.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        help="Explicit list of summary.csv files to combine (combines directly into output-file or stdout)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        help="Custom output file path (used with --inputs or when single dataset is combined)",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include smoke test / test run folders (e.g. test_*)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not add metadata columns (dataset, run_name, model_run) to combined CSV",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Do not generate condition x model comparison matrix files",
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="Do not generate token latency combined summaries",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Do not print comparison tables to stdout",
    )

    args = parser.parse_args()

    # Case 1: Explicit files passed via --inputs
    if args.inputs:
        runs = []
        for input_path_str in args.inputs:
            input_path = Path(input_path_str).resolve()
            if not input_path.exists():
                print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
                return 1
            parent_dir = input_path.parent
            meta = extract_run_metadata(parent_dir)
            runs.append(meta)

        combined_df = combine_summaries(runs, add_metadata=not args.raw, summary_filename=input_path.name)
        if combined_df.empty:
            print("No data loaded from inputs.", file=sys.stderr)
            return 1

        out_path = args.output_file or (args.output_dir / "combined_summary.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path, index=False)
        print(f"Wrote combined summary ({len(combined_df)} rows) to: {out_path}")

        if not args.no_matrix and "condition" in combined_df.columns:
            matrix_df = build_comparison_matrix(combined_df)
            matrix_path = out_path.with_name(out_path.stem + "_comparison_matrix.csv")
            matrix_df.to_csv(matrix_path, index=False)
            print(f"Wrote comparison matrix to: {matrix_path}")
            if not args.quiet:
                print_ascii_table(f"Comparison Matrix: {out_path.stem}", matrix_df)

        return 0

    # Case 2: Auto-discover datasets from evaluation directory
    groups = find_dataset_runs(args.eval_dir, include_tests=args.include_tests)
    if not groups:
        print(f"No valid evaluation runs with summary.csv found under {args.eval_dir}", file=sys.stderr)
        return 1

    # Filter datasets if specified
    if args.dataset:
        wanted = {d.lower().strip() for d in args.dataset}
        groups = {k: v for k, v in groups.items() if k.lower() in wanted}
        if not groups:
            print(f"None of the requested datasets {args.dataset} were found.", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_dfs: List[pd.DataFrame] = []
    print(f"Found {len(groups)} dataset(s): {', '.join(sorted(groups.keys()))}\n")

    for dataset_name, runs in sorted(groups.items()):
        print(f"Processing dataset: '{dataset_name}' ({len(runs)} models: {', '.join(r['model_run'] for r in runs)})")
        combined_summary = combine_summaries(runs, add_metadata=not args.raw, summary_filename="summary.csv")

        if combined_summary.empty:
            print(f"  Warning: No summary data for {dataset_name}", file=sys.stderr)
            continue

        all_dataset_dfs.append(combined_summary)

        # Output file path
        if args.output_file and len(groups) == 1:
            dataset_out_csv = args.output_file
        else:
            dataset_out_csv = args.output_dir / f"{dataset_name}_summary.csv"

        combined_summary.to_csv(dataset_out_csv, index=False)
        print(f"  -> Saved combined summary: {dataset_out_csv}")

        # Comparison matrix
        if not args.no_matrix:
            matrix_df = build_comparison_matrix(combined_summary)
            if not matrix_df.empty:
                matrix_out_csv = args.output_dir / f"{dataset_name}_comparison_matrix.csv"
                matrix_df.to_csv(matrix_out_csv, index=False)
                print(f"  -> Saved comparison matrix: {matrix_out_csv}")
                if not args.quiet:
                    print_ascii_table(f"Dataset: {dataset_name} (Accuracy & Delta vs Question-Only)", matrix_df)

        # Token latency summary
        if not args.no_latency:
            combined_latency = combine_summaries(
                runs, add_metadata=not args.raw, summary_filename="token_latency_summary.csv"
            )
            if not combined_latency.empty:
                latency_out_csv = args.output_dir / f"{dataset_name}_token_latency_summary.csv"
                combined_latency.to_csv(latency_out_csv, index=False)
                print(f"  -> Saved combined latency summary: {latency_out_csv}")

        print()

    # Master combined file across all datasets
    if len(all_dataset_dfs) > 1:
        master_df = pd.concat(all_dataset_dfs, ignore_index=True)
        master_out_csv = args.output_dir / "all_datasets_summary.csv"
        master_df.to_csv(master_out_csv, index=False)
        print(f"Saved master summary across ALL datasets ({len(master_df)} rows): {master_out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
