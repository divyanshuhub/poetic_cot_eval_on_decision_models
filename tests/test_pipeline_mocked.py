import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arc_decision.analysis import _summary_rows, analyze_run
from arc_decision.config import ExperimentConfig
from arc_decision.evaluation import _cache_key, evaluate_decisions
from arc_decision.generation import (
    GeminiQuotaExhaustedError,
    _is_retryable_error,
    _retry_delay,
    generate_rationales,
)
from arc_decision.models import ArcRecord, Choice, DecisionResult, GeminiResult, Usage
from arc_decision.storage import RunStore


class StubGemini:
    def generate(self, record, *, task, poem_form, strict_separation=False):
        reasoning = (
            "Thermal energy passes between nearby particles that touch."
            if task == "normal"
            else "Metal meets a hand\nWarmth crosses through contact\nEnergy travels"
        )
        return GeminiResult(
            reasoning=reasoning,
            final_answer_label="A",
            final_answer_text="conduction",
            poem_form=poem_form if task == "poetic" else None,
            model="stub-gemini",
            usage=Usage(input_tokens=10, output_tokens=12, total_tokens=22),
            raw_response={"ok": True},
        )


class MissingModelGemini:
    def __init__(self):
        self.calls = 0

    def generate(self, record, *, task, poem_form, strict_separation=False):
        self.calls += 1
        raise RuntimeError("404 NOT_FOUND: requested model is unavailable")


class QuotaExhaustedGemini:
    def __init__(self):
        self.calls = []

    def generate(self, record, *, task, poem_form, strict_separation=False):
        self.calls.append((record.example_id, task))
        raise RuntimeError("429 RESOURCE_EXHAUSTED: request limit exceeded for this project")


class RateLimitedOnceGemini:
    def __init__(self):
        self.calls = 0

    def generate(self, record, *, task, poem_form, strict_separation=False):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: please retry in 0s")
        return StubGemini().generate(
            record,
            task=task,
            poem_form=poem_form,
            strict_separation=strict_separation,
        )


class CountingGemini(StubGemini):
    def __init__(self):
        self.calls = 0

    def generate(self, record, *, task, poem_form, strict_separation=False):
        self.calls += 1
        return super().generate(
            record,
            task=task,
            poem_form=poem_form,
            strict_separation=strict_separation,
        )


class CsvCheckpointGemini(StubGemini):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.normal_rows_before_poetic = None

    def generate(self, record, *, task, poem_form, strict_separation=False):
        if task == "poetic":
            with (self.output_dir / "front_cot.csv").open(newline="", encoding="utf-8") as handle:
                self.normal_rows_before_poetic = list(csv.DictReader(handle))
        return super().generate(
            record,
            task=task,
            poem_form=poem_form,
            strict_separation=strict_separation,
        )


class StubDecision:
    def decide(self, state, record):
        return DecisionResult(
            predicted_answer_label="A",
            confidence=0.9,
            probabilities={"A": 0.9, "B": 0.1},
            model="stub-laya",
            usage=Usage(input_tokens=11, output_tokens=1, total_tokens=12),
            raw_response={"answers": {"arc_answer": {"choice": "A"}}},
        )


def record(source_row_index=0):
    return ArcRecord(
        example_id=f"arc_arc_easy__test__{source_row_index:06d}", dataset_id="allenai/ai2_arc", dataset_config="ARC-Easy", split="test", source_row_index=source_row_index,
        question="How does heat move through a metal spoon?", choices=(Choice("A", "conduction"), Choice("B", "evaporation")),
        gold_answer_label="A", gold_answer_text="conduction",
    )


class SummaryTokenAggregationTests(unittest.TestCase):
    def test_summary_uses_paired_reported_tokens_and_keeps_accuracy(self):
        conditions = ("question_only", "normal_reasoning", "poetic_reasoning")
        decision_rows = [
            {
                "example_id": example_id,
                "condition": condition,
                "decision_model": "stub-laya",
                "request_status": "success",
                "is_correct": "true",
            }
            for example_id in ("paired-a", "paired-b")
            for condition in conditions
        ]
        normal_rows = [
            {"example_id": "paired-a", "generation_status": "success", "cot_output_tokens": "10"},
            {"example_id": "paired-b", "generation_status": "flagged", "cot_output_tokens": "20"},
            {"example_id": "normal-only", "generation_status": "success", "cot_output_tokens": "500"},
            {"example_id": "missing-token", "generation_status": "success", "cot_output_tokens": ""},
            {"example_id": "failed-pair", "generation_status": "success", "cot_output_tokens": "100"},
        ]
        poetic_rows = [
            {"example_id": "paired-a", "generation_status": "success", "poetic_cot_output_tokens": "20"},
            {"example_id": "paired-b", "generation_status": "flagged", "poetic_cot_output_tokens": "30"},
            {"example_id": "poetic-only", "generation_status": "success", "poetic_cot_output_tokens": "900"},
            {"example_id": "missing-token", "generation_status": "success", "poetic_cot_output_tokens": "40"},
            {"example_id": "failed-pair", "generation_status": "failed", "poetic_cot_output_tokens": "200"},
        ]

        rows, _ = _summary_rows(decision_rows, 2, normal_rows, poetic_rows)
        by_condition = {row["condition"]: row for row in rows}

        self.assertEqual(by_condition["question_only"]["gemini_output_tokens_total"], None)
        self.assertEqual(by_condition["normal_reasoning"]["gemini_token_pair_examples"], 2)
        self.assertEqual(by_condition["normal_reasoning"]["gemini_output_tokens_total"], 30)
        self.assertEqual(by_condition["normal_reasoning"]["gemini_output_tokens_mean"], 15.0)
        self.assertEqual(by_condition["normal_reasoning"]["gemini_output_tokens_total_delta"], 0)
        self.assertEqual(by_condition["poetic_reasoning"]["gemini_output_tokens_total"], 50)
        self.assertEqual(by_condition["poetic_reasoning"]["gemini_output_tokens_mean"], 25.0)
        self.assertEqual(by_condition["poetic_reasoning"]["gemini_output_tokens_total_delta"], 20)
        self.assertEqual(by_condition["poetic_reasoning"]["gemini_output_tokens_mean_delta"], 10.0)
        self.assertAlmostEqual(by_condition["poetic_reasoning"]["gemini_output_tokens_delta_percent"], 200 / 3)
        self.assertEqual(by_condition["normal_reasoning"]["accuracy"], 1.0)

    def test_percentage_delta_is_null_when_normal_token_total_is_zero(self):
        decision_rows: list[dict[str, str]] = []
        normal_rows = [{"example_id": "paired", "generation_status": "success", "cot_output_tokens": "0"}]
        poetic_rows = [
            {"example_id": "paired", "generation_status": "success", "poetic_cot_output_tokens": "10"}
        ]

        rows, _ = _summary_rows(decision_rows, 1, normal_rows, poetic_rows)
        poetic = next(row for row in rows if row["condition"] == "poetic_reasoning")

        self.assertEqual(poetic["gemini_output_tokens_total_delta"], 10)
        self.assertIsNone(poetic["gemini_output_tokens_delta_percent"])


class MockedPipelineTests(unittest.TestCase):
    def test_finetuned_checkpoint_changes_decision_cache_key(self):
        base_config = ExperimentConfig(output_dir=Path("run"))
        finetuned_config = ExperimentConfig(
            output_dir=Path("run"),
            laya_finetuned_checkpoint="/models/laya-arc-easy",
        )
        key_args = {
            "record": record(),
            "condition": "question_only",
            "state_hash": "same-input",
        }

        self.assertNotEqual(
            _cache_key(config=base_config, **key_args),
            _cache_key(config=finetuned_config, **key_args),
        )

    def test_single_progress_bar_tracks_tasks_and_resume_skips(self):
        class RecordingProgress:
            def __init__(self, **kwargs):
                self.total = kwargs["total"]
                self.completed = 0
                self.postfixes = []

            def update(self, count=1):
                self.completed += count

            def set_postfix_str(self, value):
                self.postfixes.append(value)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(output_dir=root, resume=True, gemini_requests_per_minute=1000)
            store = RunStore(root)
            client = CountingGemini()
            bars = []

            def create_bar(**kwargs):
                bar = RecordingProgress(**kwargs)
                bars.append(bar)
                return bar

            with patch("arc_decision.generation.tqdm", side_effect=create_bar):
                generate_rationales([record()], config=config, store=store, client=client)
                generate_rationales([record()], config=config, store=store, client=client)

            self.assertEqual(client.calls, 2)
            self.assertEqual(len(bars), 2)
            self.assertTrue(all(bar.total == 2 and bar.completed == 2 for bar in bars))
            self.assertTrue(any("cached" in postfix for postfix in bars[1].postfixes))

    def test_normal_csv_is_checkpointed_before_poetic_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(output_dir=root, resume=True, gemini_requests_per_minute=1000)
            client = CsvCheckpointGemini(root)

            with patch("arc_decision.generation.time.sleep"):
                generate_rationales([record()], config=config, store=RunStore(root), client=client)

            self.assertEqual(len(client.normal_rows_before_poetic), 1)
            self.assertEqual(client.normal_rows_before_poetic[0]["generation_status"], "success")

    def test_provider_retry_hint_is_honored(self):
        self.assertEqual(_retry_delay(RuntimeError("Please retry in 43.7s."), 1), 43.7)

    def test_exhausted_gemini_quota_is_not_retried(self):
        error = RuntimeError(
            "429 RESOURCE_EXHAUSTED: You exceeded your current quota. "
            "Quota exceeded for metric: generate_content_free_tier_requests. "
            "Please retry in 1h37m."
        )
        self.assertFalse(_is_retryable_error(error))

    def test_quota_exhaustion_stops_generation_and_checkpoints_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(
                output_dir=root,
                resume=True,
                max_retries=2,
                gemini_requests_per_minute=1000,
            )
            store = RunStore(root)
            client = QuotaExhaustedGemini()

            with self.assertRaises(GeminiQuotaExhaustedError):
                generate_rationales([record(), record(1)], config=config, store=store, client=client)

            self.assertEqual(client.calls, [(record().example_id, "normal")])
            attempts = store.attempts("gemini")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["status"], "failed")
            with (root / "front_cot.csv").open(newline="", encoding="utf-8") as handle:
                normal_rows = list(csv.DictReader(handle))
            with (root / "front_poetic.csv").open(newline="", encoding="utf-8") as handle:
                poetic_rows = list(csv.DictReader(handle))
            self.assertEqual(normal_rows[0]["generation_status"], "failed")
            self.assertEqual(poetic_rows, [])

    def test_retryable_rate_limit_can_retry_to_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(
                output_dir=root,
                resume=True,
                max_retries=1,
                gemini_requests_per_minute=1000,
            )
            store = RunStore(root)
            client = RateLimitedOnceGemini()

            with patch("arc_decision.generation.time.sleep") as sleep:
                generate_rationales([record()], config=config, store=store, client=client)

            self.assertEqual(client.calls, 3)
            self.assertEqual(len(store.completed_keys("gemini")), 2)
            sleep.assert_any_call(0.0)

    def test_permanent_gemini_404_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(
                output_dir=root,
                resume=True,
                max_retries=2,
                gemini_requests_per_minute=1000,
            )
            store = RunStore(root)
            client = MissingModelGemini()

            generate_rationales([record()], config=config, store=store, client=client)

            self.assertEqual(client.calls, 2)
            attempts = store.attempts("gemini")
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all(attempt["terminal"] for attempt in attempts))
            self.assertTrue(all(attempt["attempt_number"] == 1 for attempt in attempts))

    def test_resume_retries_terminal_failures_but_keeps_successes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(
                output_dir=root,
                resume=True,
                max_retries=0,
                gemini_requests_per_minute=1000,
            )
            store = RunStore(root)
            failed_client = MissingModelGemini()
            generate_rationales([record()], config=config, store=store, client=failed_client)

            self.assertEqual(len(store.completed_keys("gemini")), 0)

            successful_client = StubGemini()
            generate_rationales([record()], config=config, store=store, client=successful_client)

            self.assertEqual(len(store.completed_keys("gemini")), 2)
            self.assertEqual(len(store.latest_projections("gemini")), 2)

    def test_full_pipeline_is_rebuildable_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            config = ExperimentConfig(
                output_dir=root,
                resume=True,
                decision_model="laya",
                gemini_requests_per_minute=1000,
            )
            store = RunStore(root)
            store.initialize(config)
            store.write_dataset([record()])
            generate_rationales([record()], config=config, store=store, client=StubGemini())
            evaluate_decisions([record()], config=config, store=store, client=StubDecision())
            report = analyze_run(store)

            with (root / "decision_predictions.csv").open(newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([row["condition"] for row in predictions], ["question_only", "normal_reasoning", "poetic_reasoning"])
            self.assertTrue(all(row["request_status"] == "success" for row in predictions))
            for row in predictions:
                state = json.loads(row["decision_input_state_json"])
                self.assertNotIn("gold_answer_label", state)
                self.assertNotIn("final_answer_label", state)
            self.assertEqual(report["per_question_count"], 1)
            self.assertTrue((root / "front_cot.csv").exists())
            self.assertTrue((root / "front_poetic.csv").exists())
            self.assertTrue((root / "analysis" / "paired_comparisons.json").exists())
            with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = {row["condition"]: row for row in csv.DictReader(handle)}
            self.assertEqual(summary["normal_reasoning"]["gemini_output_tokens_total"], "12")
            self.assertEqual(summary["poetic_reasoning"]["gemini_output_tokens_total_delta"], "0")
