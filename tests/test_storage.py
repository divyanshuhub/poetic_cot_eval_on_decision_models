import tempfile
import unittest
from pathlib import Path

from arc_decision.config import ExperimentConfig
from arc_decision.models import ArcRecord, Choice
from arc_decision.storage import RunStore, read_json, redact_secrets


def record():
    return ArcRecord(
        example_id="example-1", dataset_id="allenai/ai2_arc", dataset_config="ARC-Easy", split="test", source_row_index=0,
        question="Question?", choices=(Choice("A", "first"), Choice("B", "second")), gold_answer_label="A", gold_answer_text="first",
    )


class StorageTests(unittest.TestCase):
    def test_record_decision_model_updates_manifest_and_locks_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            store = RunStore(root)
            store.initialize(ExperimentConfig(output_dir=root))
            finetuned_config = ExperimentConfig(
                output_dir=root,
                laya_finetuned_checkpoint="/models/laya-arc-easy",
            )

            store.record_decision_model(finetuned_config)

            manifest_config = read_json(store.manifest_path)["config"]
            self.assertEqual(manifest_config["decision_model"], "laya")
            self.assertEqual(
                manifest_config["laya_finetuned_checkpoint"],
                "/models/laya-arc-easy",
            )

            store.append_attempt(
                "decision",
                {"cache_key": "finetuned", "terminal": True, "status": "success"},
            )
            with self.assertRaisesRegex(ValueError, "different decision model or checkpoint"):
                store.record_decision_model(ExperimentConfig(output_dir=root))

    def test_dataset_round_trip_and_terminal_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            store = RunStore(root)
            config = ExperimentConfig(output_dir=root)
            store.initialize(config)
            store.write_dataset([record()])
            self.assertEqual(store.load_records(), [record()])
            store.append_attempt(
                "gemini",
                {
                    "cache_key": "key", "terminal": True,
                    "projection": {"example_id": "example-1", "task": "normal", "source_row_index": 0},
                },
            )
            self.assertEqual(store.completed_keys("gemini"), {"key"})
            self.assertEqual(store.latest_projections("gemini")[0]["task"], "normal")

    def test_redacts_nested_credentials(self):
        redacted = redact_secrets({"api_key": "secret", "nested": {"Authorization": "Bearer secret"}})
        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["Authorization"], "[REDACTED]")

    def test_failed_terminal_attempt_is_not_a_completed_cache_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "run")
            store.append_attempt("gemini", {"cache_key": "failed", "terminal": True, "status": "failed"})
            store.append_attempt("gemini", {"cache_key": "success", "terminal": True, "status": "success"})
            self.assertEqual(store.completed_keys("gemini"), {"success"})
