import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from combine_cot_summaries import (
    extract_run_metadata,
    find_dataset_runs,
    combine_summaries,
    build_comparison_matrix,
)


class TestCombineCotSummaries(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_summary(self, dir_path: Path, model: str, base_acc: float):
        dir_path.mkdir(parents=True, exist_ok=True)
        data = {
            "condition": [
                "question_only",
                "standard_reasoning",
                "poisoned_reasoning",
                "poetic_reasoning",
            ],
            "decision_model": [model] * 4,
            "decision_model_checkpoint": [f"ckpt_{model}"] * 4,
            "eligible_examples": [100] * 4,
            "completed_predictions": [100] * 4,
            "coverage": [1.0] * 4,
            "accuracy": [base_acc, base_acc + 0.1, base_acc - 0.2, base_acc + 0.05],
            "shared_completion_examples": [100] * 4,
            "shared_completion_accuracy": [base_acc, base_acc + 0.1, base_acc - 0.2, base_acc + 0.05],
            "question_only_difference_pp": [0.0, 10.0, -20.0, 5.0],
            "question_only_difference_ci_low_pp": [0.0, 8.0, -22.0, 3.0],
            "question_only_difference_ci_high_pp": [0.0, 12.0, -18.0, 7.0],
            "cot_tokens_reported_examples": [100] * 4,
            "cot_tokens_total": [50000] * 4,
            "cot_tokens_mean": [500.0] * 4,
            "cot_tokens_std": [50.0] * 4,
            "cot_tokens_delta_percent_vs_question_only": [0.0, 15.0, 15.0, 5.0],
        }
        df = pd.DataFrame(data)
        df.to_csv(dir_path / "summary.csv", index=False)
        return df

    def test_extract_metadata(self):
        run_dir = self.root / "logiqa_kev_ft"
        run_dir.mkdir(parents=True)
        manifest = {
            "cot_csv_evaluation": {
                "dataset_name": "logiqa",
                "decision_model": "kev",
            }
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        meta = extract_run_metadata(run_dir)
        self.assertEqual(meta["dataset_name"], "logiqa")
        self.assertEqual(meta["model_run"], "kev_ft")
        self.assertEqual(meta["decision_model"], "kev")
        self.assertFalse(meta["is_test"])

    def test_find_dataset_runs_and_skip_test(self):
        d1 = self.root / "benchmark_jev"
        d2 = self.root / "benchmark_kev_ft"
        smoke = self.root / "smoke_run"

        self._create_mock_summary(d1, "jev", 0.7)
        self._create_mock_summary(d2, "kev", 0.8)
        self._create_mock_summary(smoke, "laya", 0.5)

        groups = find_dataset_runs(self.root, include_tests=False)
        self.assertIn("benchmark", groups)
        self.assertEqual(len(groups["benchmark"]), 2)

        # Smoke run should have been skipped
        self.assertNotIn("smoke_run", groups)

    def test_combine_summaries_and_matrix(self):
        d1 = self.root / "logiqa_jev"
        d2 = self.root / "logiqa_kev_ft"

        self._create_mock_summary(d1, "jev", 0.7)
        self._create_mock_summary(d2, "kev", 0.8)

        meta1 = extract_run_metadata(d1)
        meta2 = extract_run_metadata(d2)

        combined = combine_summaries([meta1, meta2], add_metadata=True)
        self.assertEqual(len(combined), 8)
        self.assertIn("dataset", combined.columns)
        self.assertIn("run_name", combined.columns)

        matrix = build_comparison_matrix(combined)
        self.assertEqual(len(matrix), 4)
        self.assertIn("jev_accuracy_%", matrix.columns)
        self.assertIn("kev_ft_accuracy_%", matrix.columns)

    def test_cli_execution(self):
        d1 = self.root / "evals" / "mydata_jev"
        d2 = self.root / "evals" / "mydata_laya_ft"
        out_dir = self.root / "out"

        self._create_mock_summary(d1, "jev", 0.6)
        self._create_mock_summary(d2, "laya", 0.4)

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "combine_cot_summaries.py"),
            "--eval-dir",
            str(self.root / "evals"),
            "--output-dir",
            str(out_dir),
            "--quiet",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        combined_csv = out_dir / "mydata_summary.csv"
        self.assertTrue(combined_csv.exists())
        df = pd.read_csv(combined_csv)
        self.assertEqual(len(df), 8)


if __name__ == "__main__":
    unittest.main()
