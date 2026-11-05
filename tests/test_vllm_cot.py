import csv
import json
import tempfile
import unittest
from pathlib import Path

from arc_decision.models import ArcRecord, Choice
from arc_decision.vllm_cot import (
    COT_COLUMNS,
    CSV_COLUMNS,
    DATASET_PREFIXES,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    build_cot_prompt,
    build_poisoned_cot_judge_prompt,
    build_poetic_cot_judge_prompt,
    build_cot_generation_report,
    csv_columns_for_dataset,
    initial_csv_rows,
    load_or_initialize_csv_rows,
    make_gemini_batch_generator,
    output_csv_path,
    output_report_path,
    populate_cot_columns,
    resolve_gemini_keys,
    run_cot_pipeline,
    split_standard_cot,
    write_csv_atomic,
    _GeminiKeyPool,
    _RequestRateLimiter,
    _build_parser,
    main,
)


def record(index: int) -> ArcRecord:
    return ArcRecord(
        example_id=f"arc_arc_easy__test__{index:06d}",
        dataset_id="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="test",
        source_row_index=index,
        question=f"Which process transfers heat through direct contact? ({index})",
        choices=(Choice("A", "Radiation"), Choice("B", "Conduction")),
        gold_answer_label="B",
        gold_answer_text="Conduction",
    )


class VllmCotPipelineTests(unittest.TestCase):
    def test_generates_exact_schema_and_resumes_completed_cells(self):
        records = (record(0), record(1))
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            rows = load_or_initialize_csv_rows(output_path, records)
            calls: list[list[str]] = []

            def generate(prompts: list[str]) -> list[str]:
                calls.append(prompts)
                completions = []
                for prompt in prompts:
                    if "Explanatory poem:" in prompt:
                        completions.append("B")
                    elif "Supported choice label:" in prompt:
                        completions.append("A")
                    elif "Corrupted Explanation" in prompt:
                        completions.append("Heat moves through light waves instead of touching particles.")
                    elif "Poetic Explanation" in prompt:
                        completions.append("Particles meet\nWarmth crosses by touch\nMetal carries heat\nEnergy walks neighbor to neighbor\nA quiet thermal path")
                    else:
                        completions.append(
                            "Thermal energy moves between particles that touch and collide.\n"
                            "The final answer is ####B"
                        )
                return completions

            populate_cot_columns(
                records,
                rows,
                batch_generator=generate,
                batch_size=2,
                checkpoint=lambda updated_rows: write_csv_atomic(output_path, updated_rows),
                show_progress=False,
            )

            with output_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                saved_rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), CSV_COLUMNS)

            self.assertEqual(len(calls), 5)
            self.assertEqual(len(saved_rows), 2)
            self.assertEqual(saved_rows[0]["answerKey"], "B")
            self.assertEqual(json.loads(saved_rows[0]["choices"])[1]["text"], "Conduction")
            for column in COT_COLUMNS:
                self.assertTrue(saved_rows[0][column])
            self.assertIn("The final answer is ####B", saved_rows[0]["standard_cot"])
            self.assertNotIn("The final answer is", saved_rows[0]["poisoned_cot"])
            self.assertIn("Original Explanation to Corrupt:", calls[1][0])
            self.assertNotIn("The final answer is ####B", calls[1][0])
            self.assertIn("Original Explanation to Rewrite:", calls[3][0])
            self.assertNotIn("The final answer is ####B", calls[3][0])

            populate_cot_columns(
                records,
                saved_rows,
                batch_generator=generate,
                batch_size=2,
                show_progress=False,
            )
            self.assertEqual(len(calls), 5)

    def test_rejects_existing_output_for_different_source_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            write_csv_atomic(output_path, load_or_initialize_csv_rows(output_path, (record(0),)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_or_initialize_csv_rows(output_path, (record(1),), resume=True)

    def test_existing_output_requires_explicit_resume_or_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            write_csv_atomic(output_path, load_or_initialize_csv_rows(output_path, (record(0),)))

            with self.assertRaisesRegex(FileExistsError, "--resume"):
                load_or_initialize_csv_rows(output_path, (record(0),))

            self.assertEqual(
                load_or_initialize_csv_rows(output_path, (record(0),), resume=True)[0]["question"],
                record(0).question,
            )
            self.assertEqual(
                load_or_initialize_csv_rows(output_path, (record(0),), force=True)[0]["answerKey"],
                "B",
            )

    def test_builds_requested_filename(self):
        path = output_csv_path("outputs/vllm_cot", "challenge", "train")
        self.assertEqual(path.name, "ARC_challenge_train.csv")

    def test_report_counts_blank_cells_and_row_completion(self):
        rows = [
            {"standard_cot": "standard", "poisoned_cot": "poisoned", "poetic_cot": "poetic"},
            {"standard_cot": "standard", "poisoned_cot": "", "poetic_cot": ""},
            {"standard_cot": "", "poisoned_cot": "", "poetic_cot": ""},
        ]
        output_csv = Path("outputs/vllm_cot/ARC_easy_test.csv")
        report = build_cot_generation_report(
            rows,
            dataset_type="easy",
            split="test",
            model="google/gemma-3-27b-it",
            output_csv=output_csv,
            max_attempts=5,
            run_status="completed",
        )

        self.assertEqual(output_report_path(output_csv).name, "ARC_easy_test_report.json")
        self.assertEqual(report["content_status"], "partial")
        self.assertEqual(report["row_counts"], {"complete": 1, "partial": 1, "empty": 1})
        self.assertEqual(report["rationale_cells"]["standard_cot"], {"filled": 2, "blank": 1})
        self.assertEqual(report["rationale_cells"]["poisoned_cot"], {"filled": 1, "blank": 2})
        self.assertEqual(report["rationale_cells"]["poetic_cot"], {"filled": 1, "blank": 2})
        self.assertEqual(report["rationale_cells"]["total"], {"filled": 4, "blank": 5})

    def test_prompts_are_named_and_do_not_include_gold_answer(self):
        source = record(0)
        clean_input = "Thermal energy moves by direct contact."
        for column in COT_COLUMNS:
            prompt = build_cot_prompt(
                source,
                column,
                clean_input=None if column == "standard_cot" else clean_input,
            )
            self.assertIn(source.question, prompt)
            self.assertIn("A. Radiation", prompt)
            self.assertNotIn('"answerKey"', prompt)
            if column != "standard_cot":
                self.assertIn(clean_input, prompt)
                self.assertNotIn("The final answer is ####B", prompt)

        poisoned_prompt = build_cot_prompt(source, "poisoned_cot", clean_input=clean_input)
        self.assertIn("Assigned target distractor", poisoned_prompt)
        self.assertIn("A. Radiation", poisoned_prompt)
        self.assertNotIn("B. Conduction", poisoned_prompt.split("Assigned target distractor", 1)[1])

        standard_prompt = build_cot_prompt(source, "standard_cot")
        self.assertIn("Question: Which process transfers thermal energy through direct contact between particles?", standard_prompt)
        self.assertIn("The final answer is ####B", standard_prompt)

    def test_standard_cot_separates_its_explanation_from_final_answer(self):
        source = record(0)
        standard = "Heat flows between touching particles.\nThe final answer is ####B"
        stored, clean_input = split_standard_cot(standard, source)

        self.assertEqual(stored, standard)
        self.assertEqual(clean_input, "Heat flows between touching particles.")
        self.assertIn("Supported choice label:", build_poisoned_cot_judge_prompt(source, "Wrong science."))

    def test_poetic_judge_prompt_uses_question_choices_and_poem(self):
        source = record(0)
        prompt = build_poetic_cot_judge_prompt(source, "Heat flows\nThrough touching metal")

        self.assertIn(source.question, prompt)
        self.assertIn("A. Radiation", prompt)
        self.assertIn("Explanatory poem:\nHeat flows\nThrough touching metal", prompt)
        self.assertIn("output UNKNOWN", prompt)

    def test_poetic_judge_retries_wrong_and_unknown_then_resume_fills_blank(self):
        source = record(0)
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            rows = load_or_initialize_csv_rows(output_path, (source,))
            rows[0]["standard_cot"] = "Heat flows through touching particles.\nThe final answer is ####B"
            rows[0]["poisoned_cot"] = "A wrong explanation that passed its judge."
            write_csv_atomic(output_path, rows)
            calls = {"poem": 0, "judge": 0}

            def reject_poem(prompts: list[str]) -> list[str]:
                if "Poetic Explanation (poem only):" in prompts[0]:
                    calls["poem"] += 1
                    return [f"Poem attempt {calls['poem']}"]
                if "Explanatory poem:" in prompts[0]:
                    calls["judge"] += 1
                    return ["A" if calls["judge"] == 1 else "UNKNOWN"]
                self.fail(f"Unexpected prompt: {prompts[0]}")

            populate_cot_columns(
                (source,),
                rows,
                batch_generator=reject_poem,
                batch_size=1,
                max_attempts=2,
                checkpoint=lambda updated_rows: write_csv_atomic(output_path, updated_rows),
                show_progress=False,
            )

            self.assertEqual(calls, {"poem": 2, "judge": 2})
            self.assertEqual(rows[0]["poetic_cot"], "")

            def accept_poem(prompts: list[str]) -> list[str]:
                if "Poetic Explanation (poem only):" in prompts[0]:
                    return ["Sunlight becomes food\nEnergy moves through green leaves\nLife builds on its source\nFood webs carry stored sunlight\nThe cycle flows onward"]
                if "Explanatory poem:" in prompts[0]:
                    return ["B"]
                self.fail(f"Unexpected prompt: {prompts[0]}")

            populate_cot_columns(
                (source,),
                rows,
                batch_generator=accept_poem,
                batch_size=1,
                max_attempts=2,
                checkpoint=lambda updated_rows: write_csv_atomic(output_path, updated_rows),
                show_progress=False,
            )

            resumed_rows = load_or_initialize_csv_rows(output_path, (source,), resume=True)
            self.assertIn("Sunlight becomes food", resumed_rows[0]["poetic_cot"])

    def test_failed_poison_rows_stay_blank_and_poetic_generation_continues(self):
        records = (record(0), record(1))
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            rows = load_or_initialize_csv_rows(output_path, records)
            for row in rows:
                row["standard_cot"] = "Heat flows between touching particles.\nThe final answer is ####B"

            def generate(prompts: list[str]) -> list[str]:
                if "Explanatory poem:" in prompts[0]:
                    return ["B" for _ in prompts]
                if "Supported choice label:" in prompts[0]:
                    return ["A" if "(0)" in prompt else "B" for prompt in prompts]
                if "Corrupted Explanation" in prompts[0]:
                    return [f"Corruption {attempt}" for attempt in range(len(prompts))]
                if "Poetic Explanation" in prompts[0]:
                    return ["Five short\nLines shape the verse\nHeat travels\nAcross each touched particle\nWarmth flows onward" for _ in prompts]
                self.fail(f"Unexpected prompt: {prompts[0]}")

            populate_cot_columns(
                records,
                rows,
                batch_generator=generate,
                batch_size=2,
                max_attempts=2,
                checkpoint=lambda updated_rows: write_csv_atomic(output_path, updated_rows),
                show_progress=False,
            )

            self.assertEqual(rows[0]["poisoned_cot"], "Corruption 0")
            self.assertEqual(rows[1]["poisoned_cot"], "")
            self.assertTrue(rows[0]["poetic_cot"])
            self.assertTrue(rows[1]["poetic_cot"])
            persisted_rows = load_or_initialize_csv_rows(output_path, records, resume=True)
            self.assertEqual(persisted_rows[0]["poisoned_cot"], "Corruption 0")
            self.assertEqual(persisted_rows[1]["poisoned_cot"], "")
            self.assertTrue(all(row["poetic_cot"] for row in persisted_rows))

    def test_invalid_standard_row_does_not_discard_other_rows_or_stop_poetry(self):
        records = (record(0), record(1))
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ARC_easy_test.csv"
            rows = load_or_initialize_csv_rows(output_path, records)

            def generate(prompts: list[str]) -> list[str]:
                if "Now, solve this problem:" in prompts[0]:
                    return [
                        "Direct contact transfers heat.\nThe final answer is ####B"
                        if "(0)" in prompt
                        else "Direct contact transfers heat.\nThe final answer is ####A"
                        for prompt in prompts
                    ]
                if "Corrupted Explanation (explanation only):" in prompts[0]:
                    return ["Light waves transfer heat through direct contact." for _ in prompts]
                if "Explanatory poem:" in prompts[0]:
                    return ["B" for _ in prompts]
                if "Supported choice label:" in prompts[0]:
                    return ["A" for _ in prompts]
                if "Poetic Explanation (poem only):" in prompts[0]:
                    return ["Heat flows\nParticles touch\nWarmth moves\nFrom one to the next\nEnergy travels" for _ in prompts]
                self.fail(f"Unexpected prompt: {prompts[0]}")

            populate_cot_columns(
                records,
                rows,
                batch_generator=generate,
                batch_size=2,
                max_attempts=2,
                checkpoint=lambda updated_rows: write_csv_atomic(output_path, updated_rows),
                show_progress=False,
            )

            self.assertIn("####B", rows[0]["standard_cot"])
            self.assertTrue(rows[0]["poisoned_cot"])
            self.assertTrue(rows[0]["poetic_cot"])
            self.assertEqual(rows[1]["standard_cot"], "")
            self.assertEqual(rows[1]["poisoned_cot"], "")
            self.assertEqual(rows[1]["poetic_cot"], "")
            persisted_rows = load_or_initialize_csv_rows(output_path, records, resume=True)
            self.assertEqual(persisted_rows[0]["standard_cot"], rows[0]["standard_cot"])
            self.assertEqual(persisted_rows[1]["standard_cot"], "")

    def test_multi_dataset_prompts_and_judge_prompts(self):
        typed_rec = ArcRecord(
            example_id="typed_1",
            dataset_id="LocalLLaMA/typed-decisions",
            dataset_config="all",
            split="train",
            source_row_index=0,
            question="You find a key on the floor.",
            choices=(Choice("A", "Pick it up"), Choice("B", "Leave it")),
            gold_answer_label="A",
            gold_answer_text="Pick it up",
            metadata={"state": "You are in a hallway."},
        )
        pw_rec = ArcRecord(
            example_id="pw_1",
            dataset_id="tasksource/proofwriter",
            dataset_config="default",
            split="train",
            source_row_index=0,
            question="The bear is quiet.",
            choices=(Choice("A", "True"), Choice("B", "False")),
            gold_answer_label="A",
            gold_answer_text="True",
            metadata={"theory": "The bear visits the squirrel.", "allProofs": "proof1"},
        )
        logiqa_rec = ArcRecord(
            example_id="logiqa_1",
            dataset_id="lucasmccabe/logiqa",
            dataset_config="default",
            split="train",
            source_row_index=0,
            question="Which follows logically?",
            choices=(
                Choice("A", "Option A"),
                Choice("B", "Option B"),
                Choice("C", "Option C"),
                Choice("D", "Option D"),
            ),
            gold_answer_label="A",
            gold_answer_text="Option A",
            metadata={"context": "All premise statements."},
        )

        for name, rec in (
            ("typed_decisions", typed_rec),
            ("proofwriter", pw_rec),
            ("logiqa", logiqa_rec),
        ):
            std = build_cot_prompt(rec, "standard_cot", dataset_name=name)
            self.assertIn(rec.question, std)
            pois = build_cot_prompt(rec, "poisoned_cot", dataset_name=name, clean_input="Original explanation")
            self.assertIn("Original explanation", pois)
            poet = build_cot_prompt(rec, "poetic_cot", dataset_name=name, clean_input="Original explanation")
            self.assertIn("Original explanation", poet)

            pois_judge = build_poisoned_cot_judge_prompt(rec, "Corrupted rationale", dataset_name=name, clean_input="Original explanation")
            poet_judge = build_poetic_cot_judge_prompt(rec, "Line 1\nLine 2", dataset_name=name, clean_input="Original explanation")
            self.assertTrue(len(pois_judge) > 0)
            self.assertTrue(len(poet_judge) > 0)

        # Check JSON judge specification for ProofWriter & LogiQA
        pw_judge = build_poisoned_cot_judge_prompt(pw_rec, "Wrong proof", dataset_name="proofwriter")
        self.assertIn('"valid_poison"', pw_judge)
        logiqa_judge = build_poetic_cot_judge_prompt(logiqa_rec, "A poem", dataset_name="logiqa")
        self.assertIn('"valid_rewrite"', logiqa_judge)

    def test_multi_dataset_output_paths_and_schemas(self):
        self.assertEqual(
            output_csv_path("outputs/vllm_cot", "all", "train", dataset_name="typed_decisions").name,
            "TypedDecisions_all_train.csv",
        )
        self.assertEqual(
            output_csv_path("outputs/vllm_cot", "default", "validation", dataset_name="proofwriter").name,
            "ProofWriter_default_validation.csv",
        )
        self.assertEqual(
            output_csv_path("outputs/vllm_cot", "default", "test", dataset_name="logiqa").name,
            "LogiQA_default_test.csv",
        )

        pw_rec = ArcRecord(
            example_id="pw_1",
            dataset_id="tasksource/proofwriter",
            dataset_config="default",
            split="train",
            source_row_index=0,
            question="The bear is quiet.",
            choices=(Choice("A", "True"), Choice("B", "False")),
            gold_answer_label="A",
            gold_answer_text="True",
            metadata={"theory": "The bear visits the squirrel.", "allProofs": "proof1"},
        )
        pw_rows = initial_csv_rows([pw_rec], dataset_name="proofwriter")
        self.assertEqual(
            tuple(pw_rows[0].keys()),
            csv_columns_for_dataset("proofwriter"),
        )
        self.assertEqual(pw_rows[0]["theory"], "The bear visits the squirrel.")
        self.assertEqual(pw_rows[0]["answer"], "A")

    def test_multi_dataset_populate_with_json_judge(self):
        pw_rec = ArcRecord(
            example_id="pw_1",
            dataset_id="tasksource/proofwriter",
            dataset_config="default",
            split="train",
            source_row_index=0,
            question="The bear is quiet.",
            choices=(Choice("A", "True"), Choice("B", "False")),
            gold_answer_label="A",
            gold_answer_text="True",
            metadata={"theory": "The bear visits the squirrel.", "allProofs": "proof1"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "ProofWriter_default_train.csv"
            rows = load_or_initialize_csv_rows(output_path, [pw_rec], dataset_name="proofwriter")

            def mock_generate(prompts: list[str]) -> list[str]:
                prompt = prompts[0]
                if "Write a concise, checkable explanation for a ProofWriter" in prompt:
                    return ["Proof: The bear is quiet.\nThe final answer is ####A"]
                if "Create a deliberately incorrect explanation" in prompt:
                    return ["Corrupted proof: The bear is loud."]
                if "Evaluate a candidate poisoned explanation" in prompt:
                    return ['{"valid_poison": true, "reason": "contradicts original"}']
                if "Rewrite the supplied correct ProofWriter explanation in poetic language" in prompt:
                    return ["Silent brown bear\nWalks through quiet green pine trees\nTruth remains unbroken\nStillness covers all the hills\nAnswers rest in calm"]
                if "Evaluate whether a poetic rewrite preserves a correct ProofWriter explanation" in prompt:
                    return ['{"valid_rewrite": true, "reason": "preserves meaning"}']
                return ["unknown"]

            populate_cot_columns(
                [pw_rec],
                rows,
                dataset_name="proofwriter",
                batch_generator=mock_generate,
                batch_size=1,
                show_progress=False,
            )

            self.assertIn("The final answer is ####A", rows[0]["standard_cot"])
            self.assertEqual(rows[0]["poisoned_cot"], "Corrupted proof: The bear is loud.")
            self.assertIn("Silent brown bear", rows[0]["poetic_cot"])

    def test_cli_parser_defaults_and_validation(self):
        parser = _build_parser()
        # Default is arc
        args = parser.parse_args(["--type", "easy", "--split", "test"])
        self.assertEqual(args.dataset, "arc")
        self.assertEqual(args.dataset_type, "easy")
        self.assertEqual(args.split, "test")

        # Missing --type for arc triggers error in main
        with self.assertRaises(SystemExit):
            main(["--split", "train"])

        # Other datasets don't require --type
        args_td = parser.parse_args(["--dataset", "typed_decisions", "--split", "train"])
        self.assertEqual(args_td.dataset, "typed_decisions")
        self.assertIsNone(args_td.dataset_type)

        args_pw = parser.parse_args(["--dataset", "proofwriter", "--split", "validation"])
        self.assertEqual(args_pw.dataset, "proofwriter")
        self.assertEqual(args_pw.split, "validation")

    def test_cli_backend_options(self):
        parser = _build_parser()
        args = parser.parse_args(["--backend", "gemini", "--dataset", "arc", "--type", "easy", "--split", "test"])
        self.assertEqual(args.backend, "gemini")
        self.assertIsNone(args.model)
        self.assertIsNone(args.api_key_env)
        self.assertEqual(args.requests_per_minute, 12.0)

        args_vllm = parser.parse_args(["--backend", "vllm", "--dataset", "arc", "--type", "easy", "--split", "test"])
        self.assertEqual(args_vllm.backend, "vllm")

    def test_resolve_gemini_keys_prioritizes_1_and_2(self):
        import os
        from unittest.mock import patch

        # Neither key set
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY_1 nor GEMINI_API_KEY_2"):
                resolve_gemini_keys()

        # GEMINI_API_KEY_1 only
        with patch.dict(os.environ, {"GEMINI_API_KEY_1": "key_one"}, clear=True):
            keys = resolve_gemini_keys()
            self.assertEqual(keys, [("GEMINI_API_KEY_1", "key_one")])

        # GEMINI_API_KEY_2 only
        with patch.dict(os.environ, {"GEMINI_API_KEY_2": "key_two"}, clear=True):
            keys = resolve_gemini_keys()
            self.assertEqual(keys, [("GEMINI_API_KEY_2", "key_two")])

        # Both exported
        with patch.dict(os.environ, {"GEMINI_API_KEY_1": "key_one", "GEMINI_API_KEY_2": "key_two"}, clear=True):
            keys = resolve_gemini_keys()
            self.assertEqual(keys, [("GEMINI_API_KEY_1", "key_one"), ("GEMINI_API_KEY_2", "key_two")])

        # Explicit api_key_env
        with patch.dict(os.environ, {"CUSTOM_KEY": "custom_val"}, clear=True):
            keys = resolve_gemini_keys(api_key_env="CUSTOM_KEY")
            self.assertEqual(keys, [("CUSTOM_KEY", "custom_val")])

    def test_gemini_key_pool_round_robin_and_failover(self):
        client1 = "mock_client_1"
        client2 = "mock_client_2"
        limiter1 = _RequestRateLimiter(0.0)
        limiter2 = _RequestRateLimiter(0.0)

        pool = _GeminiKeyPool([
            ("GEMINI_API_KEY_1", client1, limiter1),
            ("GEMINI_API_KEY_2", client2, limiter2),
        ])

        # Round robin alternating
        name1, c1, _ = pool.get_client()
        name2, c2, _ = pool.get_client()
        name3, c3, _ = pool.get_client()
        self.assertEqual(name1, "GEMINI_API_KEY_1")
        self.assertEqual(name2, "GEMINI_API_KEY_2")
        self.assertEqual(name3, "GEMINI_API_KEY_1")

        # Key 1 quota exhausted -> failover to Key 2
        self.assertTrue(pool.has_alternative("GEMINI_API_KEY_1"))
        pool.mark_exhausted("GEMINI_API_KEY_1")
        self.assertFalse(pool.has_alternative("GEMINI_API_KEY_2"))

        next_name, next_c, _ = pool.get_client()
        self.assertEqual(next_name, "GEMINI_API_KEY_2")
        next_name2, next_c2, _ = pool.get_client()
        self.assertEqual(next_name2, "GEMINI_API_KEY_2")

    def test_gemini_batch_generator_with_mock_client(self):
        class MockResponse:
            def __init__(self, text: str):
                self.text = text

        class MockModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, *, model, contents, config):
                self.calls.append((model, contents, config))
                return MockResponse(f"Generated for: {contents[:20]}")

        class MockClient:
            def __init__(self):
                self.models = MockModels()

        mock_client = MockClient()
        gen = make_gemini_batch_generator(
            model="gemini-3.5-flash-lite",
            client=mock_client,
            requests_per_minute=0.0,
        )
        results = gen(["Prompt 1", "Prompt 2"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "Generated for: Prompt 1")
        self.assertEqual(results[1], "Generated for: Prompt 2")

    def test_gemini_batch_generator_retries_transient_error(self):
        class MockResponse:
            def __init__(self, text: str):
                self.text = text

        class MockModels:
            def __init__(self):
                self.attempts = 0

            def generate_content(self, *, model, contents, config):
                self.attempts += 1
                if self.attempts == 1:
                    err = RuntimeError("Internal server error 500")
                    err.status_code = 500
                    raise err
                return MockResponse("Recovered completion")

        class MockClient:
            def __init__(self):
                self.models = MockModels()

        mock_client = MockClient()
        gen = make_gemini_batch_generator(
            model="gemini-3.5-flash-lite",
            client=mock_client,
            requests_per_minute=0.0,
        )
        results = gen(["Test prompt"])
        self.assertEqual(results, ["Recovered completion"])
        self.assertEqual(mock_client.models.attempts, 2)

    def test_run_cot_pipeline_with_gemini_backend(self):
        class MockResponse:
            def __init__(self, text: str):
                self.text = text

        class MockModels:
            def generate_content(self, *, model, contents, config):
                if "Explanatory poem:" in contents:
                    return MockResponse("A")
                if "Supported choice label:" in contents:
                    return MockResponse("B")
                if "Original Explanation to Corrupt:" in contents:
                    return MockResponse("Most ecosystems are on land where sun reaches.")
                if "Original Explanation to Rewrite:" in contents:
                    return MockResponse(
                        "Sunlight feeds the trees\nEnergy flows to all life\nGreen leaves take the light\nFood webs rise from sun-kissed soil\nLife begins with light"
                    )
                return MockResponse(
                    "Sunlight provides the primary energy input for producers.\nThe final answer is ####A"
                )

        class MockClient:
            def __init__(self):
                self.models = MockModels()

        with tempfile.TemporaryDirectory() as temporary:
            output_path = run_cot_pipeline(
                backend="gemini",
                dataset_name="arc",
                dataset_type="easy",
                split="test",
                output_dir=temporary,
                limit=1,
                gemini_client=MockClient(),
                requests_per_minute=0.0,
            )
            self.assertTrue(output_path.exists())
            report_path = output_report_path(output_path)
            self.assertTrue(report_path.exists())

            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["backend"], "gemini")
            self.assertEqual(report["model"], "gemini-3.5-flash-lite")
            self.assertEqual(report["total_rows"], 1)

            with output_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertIn("The final answer is ####A", rows[0]["standard_cot"])
            self.assertTrue(rows[0]["poisoned_cot"])
            self.assertTrue(rows[0]["poetic_cot"])
