import csv
import json
import tempfile
import unittest
from pathlib import Path

from arc_decision.cot_evaluation import (
    COT_CONDITIONS,
    _paired_cot_input_token_summary,
    detect_dataset_from_csv,
    main,
    run_cot_csv_evaluation,
)
from arc_decision.models import DecisionResult, Usage


class FakeDecisionClient:
    def __init__(self):
        self.calls = []

    def decide(self, state, record):
        self.calls.append((state, record.example_id))
        labels = list(record.choice_map.keys())
        label = labels[0] if "poisoned_rationale" in state else labels[-1]
        rationale_key = next((k for k in ("standard_rationale", "poisoned_rationale", "poetic_rationale") if k in state), "question_only")
        token_count = {
            "question_only": 100,
            "standard_rationale": 110,
            "poisoned_rationale": 130,
            "poetic_rationale": 120,
        }[rationale_key]
        probs = {lbl: (0.8 if lbl == label else 0.2 / max(1, len(labels) - 1)) for lbl in labels}
        return DecisionResult(
            predicted_answer_label=label,
            confidence=0.8,
            probabilities=probs,
            model="fake-laya-finetuned",
            usage=Usage(input_tokens=token_count, output_tokens=1, total_tokens=token_count + 1),
            request_id="request-1",
            raw_response={"answers": {"arc_answer": {"choice": label}}},
        )


def write_cot_csv(path: Path, *, blank_poisoned: bool = False):
    rows = [
        {
            "question": "Which process transfers heat through direct contact?",
            "choices": json.dumps(
                [{"label": "A", "text": "Radiation"}, {"label": "B", "text": "Conduction"}]
            ),
            "answerKey": "B",
            "standard_cot": "Heat moves between touching particles.\nThe final answer is ####B",
            "poisoned_cot": "Light waves transfer heat between touching particles.",
            "poetic_cot": "Metal meets\nWarmth crosses through\nEnergy travels",
        }
    ]
    if blank_poisoned:
        rows[0]["poisoned_cot"] = ""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CotEvaluationTests(unittest.TestCase):
    def test_runs_four_conditions_strips_standard_answer_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "ARC_easy_test_complete_rows.csv"
            output_dir = root / "evaluation"
            write_cot_csv(input_csv)
            client = FakeDecisionClient()

            result = run_cot_csv_evaluation(
                input_csv=input_csv,
                dataset_type="easy",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-checkpoint",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-checkpoint",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 4)
            self.assertEqual(COT_CONDITIONS, tuple(row["condition"] for row in result["summary_rows"]))
            states_by_key = {next(iter(state.keys() - {"question", "choices"}), "question_only"): state for state, _ in client.calls}
            self.assertEqual(states_by_key["question_only"].keys(), {"question", "choices"})
            self.assertEqual(states_by_key["standard_rationale"], {"question": "Which process transfers heat through direct contact?", "choices": [{"label": "A", "text": "Radiation"}, {"label": "B", "text": "Conduction"}], "standard_rationale": "Heat moves between touching particles."})
            self.assertNotIn("answerKey", states_by_key["standard_rationale"])

            with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual([row["condition"] for row in summary], list(COT_CONDITIONS))
            self.assertEqual(summary[0]["completed_predictions"], "1")
            self.assertEqual(summary[0]["accuracy"], "1.0")
            self.assertEqual(summary[2]["accuracy"], "0.0")
            self.assertEqual(summary[0]["cot_tokens_total"], "100")
            self.assertEqual(summary[1]["cot_tokens_total"], "110")
            self.assertEqual(summary[1]["cot_tokens_delta_percent_vs_question_only"], "10.0")
            self.assertTrue((output_dir / "decision_predictions.csv").is_file())
            self.assertTrue((output_dir / "analysis" / "paired_comparisons.json").is_file())

    def test_token_statistics_use_paired_reported_usage_and_question_only_percent(self):
        rows = []
        counts = {
            "example-1": {"question_only": 100, "standard_reasoning": 110, "poisoned_reasoning": 130, "poetic_reasoning": 120},
            "example-2": {"question_only": 200, "standard_reasoning": 220, "poisoned_reasoning": 230, "poetic_reasoning": 260},
            "example-3": {"question_only": 300, "standard_reasoning": 310, "poisoned_reasoning": 330, "poetic_reasoning": None},
        }
        for example_id, condition_counts in counts.items():
            for condition in COT_CONDITIONS:
                token_count = condition_counts[condition]
                rows.append(
                    {
                        "example_id": example_id,
                        "condition": condition,
                        "request_status": "success",
                        "decision_input_tokens": token_count,
                    }
                )

        summary = _paired_cot_input_token_summary(rows)

        self.assertEqual(summary["question_only"]["cot_tokens_reported_examples"], 2)
        self.assertEqual(summary["question_only"]["cot_tokens_total"], 300)
        self.assertEqual(summary["question_only"]["cot_tokens_mean"], 150)
        self.assertEqual(summary["question_only"]["cot_tokens_std"], 50)
        self.assertEqual(summary["standard_reasoning"]["cot_tokens_total"], 330)
        self.assertEqual(summary["standard_reasoning"]["cot_tokens_mean"], 165)
        self.assertEqual(summary["standard_reasoning"]["cot_tokens_std"], 55)
        self.assertEqual(summary["standard_reasoning"]["cot_tokens_delta_percent_vs_question_only"], 10)
        self.assertAlmostEqual(summary["poetic_reasoning"]["cot_tokens_delta_percent_vs_question_only"], 40 / 150 * 100)

    def test_missing_rationale_is_skipped_without_blocking_other_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "ARC_easy_test.csv"
            output_dir = root / "evaluation"
            write_cot_csv(input_csv, blank_poisoned=True)
            client = FakeDecisionClient()

            result = run_cot_csv_evaluation(
                input_csv=input_csv,
                dataset_type="easy",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-checkpoint",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-checkpoint",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 3)
            skipped = [row for row in result["summary_rows"] if row["condition"] == "poisoned_reasoning"][0]
            self.assertEqual(skipped["completed_predictions"], 0)
            self.assertEqual(skipped["coverage"], 0.0)
            self.assertEqual(result["error_count"], 1)

    def test_resume_skips_successful_matching_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "ARC_easy_test.csv"
            output_dir = root / "evaluation"
            write_cot_csv(input_csv)
            first_client = FakeDecisionClient()
            arguments = {
                "input_csv": input_csv,
                "dataset_type": "easy",
                "split": "test",
                "decision_model": "laya",
                "model_identity": "fake-laya-checkpoint",
                "output_dir": output_dir,
                "laya_finetuned_checkpoint": "fake-laya-checkpoint",
                "max_retries": 0,
            }
            run_cot_csv_evaluation(**arguments, client=first_client)
            second_client = FakeDecisionClient()
            run_cot_csv_evaluation(**arguments, resume=True, client=second_client)

            self.assertEqual(len(first_client.calls), 4)
            self.assertEqual(second_client.calls, [])

    def test_multi_dataset_cot_csv_evaluation_with_finetuned_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logiqa_csv = root / "LogiQA_default_test.csv"
            rows = [
                {
                    "context": "All birds have feathers. Sparrows are birds.",
                    "query": "What do sparrows have?",
                    "options": json.dumps(["Feathers", "Scales", "Fur", "Gills"]),
                    "correct_option": "0",
                    "standard_cot": "Sparrows are birds so they have feathers.\nThe final answer is ####0",
                    "poisoned_cot": "Sparrows live near water and might have scales.",
                    "poetic_cot": "Feathers in the sky\nSparrows fly so very high\nWings of grace",
                }
            ]
            with logiqa_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            client = FakeDecisionClient()
            output_dir = root / "eval_logiqa"
            result = run_cot_csv_evaluation(
                input_csv=logiqa_csv,
                dataset_name="logiqa",
                dataset_type="default",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-ft-logiqa",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-ft-logiqa",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 4)
            for state, _ in client.calls:
                self.assertEqual(state.get("context"), "All birds have feathers. Sparrows are birds.")

    def test_cot_csv_evaluation_typed_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            td_csv = root / "TypedDecisions_all_test.csv"
            rows = [
                {
                    "state": json.dumps({"agent_id": "agent-42", "memory_usage": "120MB"}),
                    "question": "Should the system scale up resources?",
                    "choices": json.dumps([{"label": "scale_up", "text": "Scale up"}, {"label": "maintain", "text": "Maintain"}]),
                    "answerKey": "scale_up",
                    "standard_cot": "Memory usage is high so scale up.\nThe final answer is ####scale_up",
                    "poisoned_cot": "Memory usage is normal so maintain.",
                    "poetic_cot": "Loads rise high\nServers need more room to fly\nScale to the sky",
                }
            ]
            with td_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            class TdDecisionClient:
                def __init__(self):
                    self.calls = []

                def decide(self, state, record):
                    self.calls.append((state, record))
                    return DecisionResult(
                        predicted_answer_label="scale_up",
                        confidence=0.9,
                        probabilities={"scale_up": 0.9, "maintain": 0.1},
                        model="fake-td-model",
                        usage=Usage(input_tokens=50, output_tokens=1, total_tokens=51),
                        raw_response={"answers": {"arc_answer": {"choice": "scale_up"}}},
                    )

            client = TdDecisionClient()
            output_dir = root / "eval_td"
            result = run_cot_csv_evaluation(
                input_csv=td_csv,
                dataset_name="typed_decisions",
                dataset_type="all",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-ft-td",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-ft-td",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 4)
            for state, _ in client.calls:
                self.assertEqual(state.get("agent_id"), "agent-42")
                self.assertEqual(state.get("memory_usage"), "120MB")

            with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual([row["condition"] for row in summary], list(COT_CONDITIONS))
            self.assertEqual(summary[0]["coverage"], "1.0")

    def test_cot_csv_evaluation_proofwriter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pw_csv = root / "ProofWriter_default_test.csv"
            rows = [
                {
                    "theory": "All carnivores eat meat. Lions are carnivores.",
                    "question": "Lions eat meat.",
                    "answer": "True",
                    "allProofs": "Lions -> carnivores -> eat meat",
                    "standard_cot": "Since lions are carnivores they eat meat.\nThe final answer is ####True",
                    "poisoned_cot": "Lions do not eat meat.",
                    "poetic_cot": "Lions roar loud\nEating meat and standing proud\nTrue without cloud",
                }
            ]
            with pw_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            class PwDecisionClient:
                def __init__(self):
                    self.calls = []

                def decide(self, state, record):
                    self.calls.append((state, record))
                    return DecisionResult(
                        predicted_answer_label="True",
                        confidence=0.95,
                        probabilities={"True": 0.95, "False": 0.03, "Unknown": 0.02},
                        model="fake-pw-model",
                        usage=Usage(input_tokens=40, output_tokens=1, total_tokens=41),
                        raw_response={"answers": {"arc_answer": {"choice": "True"}}},
                    )

            client = PwDecisionClient()
            output_dir = root / "eval_pw"
            result = run_cot_csv_evaluation(
                input_csv=pw_csv,
                dataset_name="proofwriter",
                dataset_type="default",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-ft-pw",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-ft-pw",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 4)
            for state, _ in client.calls:
                self.assertEqual(state.get("theory"), "All carnivores eat meat. Lions are carnivores.")
            self.assertTrue((output_dir / "summary.csv").is_file())

    def test_dataset_auto_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            # Test filename-based detection
            ds, cfg, sp = detect_dataset_from_csv(root / "LogiQA_default_test.csv")
            self.assertEqual((ds, cfg, sp), ("logiqa", "default", "test"))

            ds, cfg, sp = detect_dataset_from_csv(root / "TypedDecisions_all_test.csv")
            self.assertEqual((ds, cfg, sp), ("typed_decisions", "all", "test"))

            ds, cfg, sp = detect_dataset_from_csv(root / "ProofWriter_default_train.csv")
            self.assertEqual((ds, cfg, sp), ("proofwriter", "default", "train"))

            ds, cfg, sp = detect_dataset_from_csv(root / "ARC_challenge_test.csv")
            self.assertEqual((ds, cfg, sp), ("arc", "challenge", "test"))

            # Test header-based detection for generic filename
            generic_csv = root / "unnamed_records.csv"
            with generic_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["context", "query", "options", "correct_option", "standard_cot", "poisoned_cot", "poetic_cot"])
            ds, cfg, sp = detect_dataset_from_csv(generic_csv)
            self.assertEqual(ds, "logiqa")

    def test_cli_main_with_auto_detection_and_custom_flags(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logiqa_csv = root / "LogiQA_default_test.csv"
            rows = [
                {
                    "context": "Solar cells convert sunlight into electricity.",
                    "query": "What do solar cells do?",
                    "options": json.dumps(["Convert sunlight", "Burn fuel", "Store water", "Create wind"]),
                    "correct_option": "0",
                    "standard_cot": "Solar cells convert sunlight to electricity.\nThe final answer is ####0",
                    "poisoned_cot": "Solar cells store water.",
                    "poetic_cot": "Sunlight from high\nPower flows without a sigh\nClean energy sparks",
                }
            ]
            with logiqa_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "out_cli"
            with patch("arc_decision.cot_evaluation._project_root", return_value=root), \
                 patch("arc_decision.cot_evaluation.create_decision_client", return_value=FakeDecisionClient()), \
                 patch("arc_decision.cot_evaluation.resolve_finetuned_model_identity", return_value=("fake-id", "fake-ckpt")):
                exit_code = main([
                    "--input-csv", str(logiqa_csv),
                    "--decision-model", "laya",
                    "--laya-finetuned-checkpoint", "fake-ckpt",
                    "--output-dir", str(output_dir),
                    "--max-retries", "0",
                ])
                self.assertEqual(exit_code, 0)
                self.assertTrue((output_dir / "summary.csv").is_file())
                with (output_dir / "summary.csv").open(encoding="utf-8") as h:
                    summary = list(csv.DictReader(h))
                self.assertEqual(len(summary), 4)
                for s_row in summary:
                    self.assertEqual(s_row["completed_predictions"], "1")

    def test_multi_dataset_cot_csv_evaluation_with_finetuned_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logiqa_csv = root / "LogiQA_default_test.csv"
            rows = [
                {
                    "context": "All birds have feathers. Sparrows are birds.",
                    "query": "What do sparrows have?",
                    "options": json.dumps(["Feathers", "Scales", "Fur", "Gills"]),
                    "correct_option": "0",
                    "standard_cot": "Sparrows are birds so they have feathers.\nThe final answer is ####0",
                    "poisoned_cot": "Sparrows live near water and might have scales.",
                    "poetic_cot": "Feathers in the sky\nSparrows fly so very high\nWings of grace",
                }
            ]
            with logiqa_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            client = FakeDecisionClient()
            output_dir = root / "eval_logiqa"
            result = run_cot_csv_evaluation(
                input_csv=logiqa_csv,
                dataset_name="logiqa",
                dataset_type="default",
                split="test",
                decision_model="laya",
                model_identity="fake-laya-ft-logiqa",
                output_dir=output_dir,
                laya_finetuned_checkpoint="fake-laya-ft-logiqa",
                max_retries=0,
                client=client,
            )

            self.assertEqual(len(client.calls), 4)
            for state, _ in client.calls:
                self.assertEqual(state.get("context"), "All birds have feathers. Sparrows are birds.")
