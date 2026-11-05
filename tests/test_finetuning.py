import json
import tempfile
import unittest
from pathlib import Path

import yaml

from arc_decision.dataset import DatasetLoadResult
from arc_decision.finetuning import (
    ARC_QUESTION_ID,
    FineTuneSplits,
    build_native_commands,
    clean_run_directory,
    clef_training_row,
    dataset_config_for_difficulty,
    fine_tune_paths,
    kev_training_row,
    laya_training_row,
    load_arc_finetune_splits,
    load_finetune_splits,
    resolve_finetune_dataset_config,
    systemone_state_for_record,
    write_clef_config,
    write_native_training_data,
)
from arc_decision.models import ArcRecord, Choice


def record(index=0):
    return ArcRecord(
        example_id=f"arc_arc_easy__train__{index:06d}",
        dataset_id="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="train",
        source_row_index=index,
        question="Which process transfers heat through direct contact?",
        choices=(Choice("A", "Radiation"), Choice("B", "Conduction")),
        gold_answer_label="B",
        gold_answer_text="Conduction",
    )


class FineTuningDataTests(unittest.TestCase):
    def test_difficulty_maps_to_canonical_arc_configuration(self):
        self.assertEqual(dataset_config_for_difficulty("easy"), "ARC-Easy")
        self.assertEqual(dataset_config_for_difficulty("challenge"), "ARC-Challenge")

    def test_each_model_uses_its_native_system_one_label_shape(self):
        source = record()
        laya = laya_training_row(source)
        kev = kev_training_row(source)
        clef = clef_training_row(source)

        self.assertEqual(laya["state"], {"question": source.question})
        self.assertEqual(laya["expected"], {ARC_QUESTION_ID: "B"})
        self.assertNotIn("label", laya["questions"][ARC_QUESTION_ID])
        self.assertEqual(kev["questions"][ARC_QUESTION_ID]["label"], "B")
        self.assertNotIn("expected", kev)
        self.assertEqual(clef["labels"], {ARC_QUESTION_ID: "B"})
        self.assertEqual(clef["questions"][ARC_QUESTION_ID]["criteria"], {"A": "Radiation", "B": "Conduction"})
        self.assertEqual(clef["state"], {"question": source.question})

    def test_uses_official_validation_split_when_available(self):
        calls = []

        def loader(**kwargs):
            calls.append(kwargs)
            source = record(0 if kwargs["split"] == "train" else 1)
            return DatasetLoadResult((source,), (), "fingerprint", "version")

        splits = load_arc_finetune_splits("easy", loader=loader)

        self.assertEqual(splits.dataset_config, "ARC-Easy")
        self.assertEqual(splits.validation_source, "validation")
        self.assertEqual([call["split"] for call in calls], ["train", "validation"])

    def test_holds_out_train_only_when_validation_split_is_unavailable(self):
        source_records = tuple(record(index) for index in range(10))

        def loader(**kwargs):
            if kwargs["split"] == "validation":
                raise ValueError("Unknown split validation")
            return DatasetLoadResult(source_records, (), "fingerprint", "version")

        splits = load_arc_finetune_splits("challenge", loader=loader, validation_fraction=0.2, seed=3)

        self.assertEqual(splits.validation_source, "train_holdout")
        self.assertEqual(len(splits.train), 8)
        self.assertEqual(len(splits.validation), 2)
        self.assertEqual({row.example_id for row in splits.train} & {row.example_id for row in splits.validation}, set())

    def test_multi_dataset_config_resolution(self):
        self.assertEqual(resolve_finetune_dataset_config("arc", "easy"), "ARC-Easy")
        self.assertEqual(resolve_finetune_dataset_config("arc", "challenge"), "ARC-Challenge")
        self.assertEqual(resolve_finetune_dataset_config("typed_decisions"), "all")
        self.assertEqual(resolve_finetune_dataset_config("typed_decisions", difficulty="challenge"), "all")
        self.assertEqual(resolve_finetune_dataset_config("typed_decisions", difficulty="easy"), "all")
        self.assertEqual(resolve_finetune_dataset_config("typed_decisions", dataset_config="customer_service"), "customer_service")
        self.assertEqual(resolve_finetune_dataset_config("typed_decisions", dataset_config="customer-service"), "customer_service")
        self.assertEqual(resolve_finetune_dataset_config("proofwriter"), "default")
        self.assertEqual(resolve_finetune_dataset_config("proofwriter", difficulty="challenge"), "default")
        self.assertEqual(resolve_finetune_dataset_config("logiqa"), "default")
        self.assertEqual(resolve_finetune_dataset_config("logiqa", difficulty="challenge"), "default")
        with self.assertRaises(ValueError):
            resolve_finetune_dataset_config("typed_decisions", dataset_config="invalid_config")

    def test_systemone_formats_for_all_reasoning_datasets(self):
        arc_rec = record(0)
        logiqa_rec = ArcRecord(
            example_id="logiqa_default__train__000001",
            dataset_id="lucasmccabe/logiqa",
            dataset_config="default",
            split="train",
            source_row_index=1,
            question="Which deduction follows?",
            choices=(Choice("0", "Opt A"), Choice("1", "Opt B")),
            gold_answer_label="0",
            gold_answer_text="Opt A",
            metadata={"context": "All humans are mortal. Socrates is human."},
        )
        pw_rec = ArcRecord(
            example_id="proofwriter_default__train__000002",
            dataset_id="tasksource/proofwriter",
            dataset_config="default",
            split="train",
            source_row_index=2,
            question="Socrates is mortal.",
            choices=(Choice("True", "True"), Choice("False", "False"), Choice("Unknown", "Unknown")),
            gold_answer_label="True",
            gold_answer_text="True",
            metadata={"theory": "Theory: Humans are mortal.", "allProofs": ""},
        )
        td_rec = ArcRecord(
            example_id="td_all__train__000003",
            dataset_id="LocalLLaMA/typed-decisions",
            dataset_config="all",
            split="train",
            source_row_index=3,
            question="What is the next action?",
            choices=(Choice("A", "Retry"), Choice("B", "Abort")),
            gold_answer_label="A",
            gold_answer_text="Retry",
            metadata={"state": {"user_id": "u123", "error_code": 500}},
        )

        # Test state representations
        self.assertEqual(systemone_state_for_record(arc_rec), {"question": arc_rec.question})
        self.assertEqual(systemone_state_for_record(logiqa_rec), {"context": "All humans are mortal. Socrates is human.", "question": logiqa_rec.question})
        self.assertEqual(systemone_state_for_record(pw_rec), {"theory": "Theory: Humans are mortal.", "question": pw_rec.question})
        self.assertEqual(systemone_state_for_record(td_rec), {"user_id": "u123", "error_code": 500, "question": td_rec.question})

        # Test model transformations
        laya_logiqa = laya_training_row(logiqa_rec)
        self.assertEqual(laya_logiqa["expected"], {ARC_QUESTION_ID: "0"})
        self.assertIn("context", laya_logiqa["state"])

        kev_pw = kev_training_row(pw_rec)
        self.assertEqual(kev_pw["questions"][ARC_QUESTION_ID]["label"], "True")
        self.assertIn("theory", kev_pw["state"])

        clef_td = clef_training_row(td_rec)
        self.assertEqual(clef_td["labels"], {ARC_QUESTION_ID: "A"})
        self.assertEqual(clef_td["state"]["user_id"], "u123")

    def test_multi_dataset_distinct_paths_and_clean_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arc_paths = fine_tune_paths("laya", "easy")
            self.assertIn("laya_arc_easy", str(arc_paths.root))

            logiqa_paths = fine_tune_paths("kev", dataset="logiqa")
            self.assertIn("kev_logiqa_default", str(logiqa_paths.root))

            td_paths = fine_tune_paths("clef", dataset="typed_decisions", dataset_config="customer_service")
            self.assertIn("clef_typed_decisions_customer_service", str(td_paths.root))

            # Test clean_run_directory for replacement
            temp_paths = fine_tune_paths("laya", output_dir=root / "test_run")
            temp_paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (temp_paths.checkpoint_dir / "model.bin").write_text("old_model")
            self.assertTrue(temp_paths.checkpoint_dir.exists())

            clean_run_directory(temp_paths)
            self.assertFalse(temp_paths.checkpoint_dir.exists())

    def test_writes_distinct_data_and_clef_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = fine_tune_paths("clef", "easy", root / "clef_easy")
            splits = FineTuneSplits("easy", "ARC-Easy", (record(),), (record(1),), "validation", 0, 0)

            write_native_training_data("clef", splits, paths)
            write_clef_config(paths, base_model="Cloudflare/clef-flash", seed=7, max_steps=25)

            train_row = json.loads(paths.train_jsonl.read_text(encoding="utf-8").strip())
            config = yaml.safe_load(paths.clef_config_path.read_text(encoding="utf-8"))
            self.assertEqual(train_row["labels"][ARC_QUESTION_ID], "B")
            self.assertEqual(config["data"]["train"], str(paths.train_jsonl))
            self.assertEqual(config["data"]["eval"], str(paths.validation_jsonl))
            self.assertEqual(config["output_dir"], str(paths.checkpoint_dir))
            self.assertEqual(config["train"]["lr"], 1.0e-4)
            self.assertEqual(config["train"]["head_lr"], 5.0e-5)
            self.assertIsInstance(config["train"]["head_lr"], float)


class NativeCommandTests(unittest.TestCase):
    def test_each_model_delegates_to_the_expected_native_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = fine_tune_paths("laya", "easy", root / "laya_easy")
            laya = build_native_commands("laya", paths, base_model="base", device="cuda")
            kev = build_native_commands(
                "kev",
                paths,
                base_model="jaredpalmer/kev-4b",
                device="cuda",
                kev_repo_dir=root,
            )
            clef = build_native_commands("clef", paths, base_model="base", device="cuda", clef_max_steps=25)

        self.assertEqual(laya[0].argv[0], "laya-train")
        self.assertIn("--eval", laya[0].argv)
        self.assertEqual(kev[0].argv[:5], ("uv", "run", "python", "-m", "kev.train"))
        self.assertEqual(kev[0].argv[kev[0].argv.index("--base") + 1], "Qwen/Qwen3.5-4B-Base")
        self.assertEqual(kev[0].argv[kev[0].argv.index("--init_from") + 1], "jaredpalmer/kev-4b")
        self.assertEqual(kev[1].argv[:5], ("uv", "run", "python", "-m", "kev.benchmark"))
        self.assertEqual(clef[0].argv[:2], ("clef-finetune", "train"))
        self.assertEqual(clef[1].argv[:2], ("clef-finetune", "eval"))
        self.assertIn("checkpoint-25", " ".join(clef[1].argv))

    def test_base_model_replacement_in_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = fine_tune_paths("laya", dataset="logiqa", output_dir=root / "laya_ft")
            custom_checkpoint = "/path/to/previous/checkpoint"
            laya = build_native_commands("laya", paths, base_model=custom_checkpoint, device="cuda")
            kev = build_native_commands(
                "kev",
                paths,
                base_model=custom_checkpoint,
                device="cuda",
                kev_repo_dir=root,
                kev_base_model="Qwen/Qwen3.5-4B-Base",
            )
            clef = build_native_commands("clef", paths, base_model=custom_checkpoint, device="cuda", clef_max_steps=50)

        self.assertIn(custom_checkpoint, laya[0].argv)
        self.assertEqual(kev[0].argv[kev[0].argv.index("--init_from") + 1], custom_checkpoint)