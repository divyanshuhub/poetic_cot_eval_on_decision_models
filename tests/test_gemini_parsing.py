import os
import unittest
from unittest.mock import patch

from arc_decision.gemini_client import GoogleGenAIClient, parse_gemini_payload
from arc_decision.models import ArcRecord, Choice, Usage


def record():
    return ArcRecord(
        example_id="example-1",
        dataset_id="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="test",
        source_row_index=0,
        question="How does heat travel?",
        choices=(Choice("A", "radiation"), Choice("B", "conduction")),
        gold_answer_label="B",
        gold_answer_text="conduction",
    )


class GeminiParsingTests(unittest.TestCase):
    def test_accepts_matching_structured_output(self):
        result = parse_gemini_payload(
            {"reasoning": "Heat moves through contact.", "final_answer_label": "B", "final_answer_text": "conduction"},
            record=record(), task="normal", model="stub", usage=Usage(1, 2, 3), finish_reason="STOP", raw_response={},
        )
        self.assertEqual(result.final_answer_label, "B")

    def test_rejects_invalid_choice_label(self):
        with self.assertRaises(ValueError):
            parse_gemini_payload(
                {"reasoning": "Heat moves.", "final_answer_label": "Z"},
                record=record(), task="normal", model="stub", usage=Usage(), finish_reason=None, raw_response={},
            )


class GeminiApiKeySelectionTests(unittest.TestCase):
    def test_reads_the_configured_environment_variable(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY_2": "test-key"}, clear=True):
            client = GoogleGenAIClient(
                model="stub", temperature=0.0, seed=7, api_key_env="GEMINI_API_KEY_2"
            )

        self.assertEqual(client.api_key, "test-key")

    def test_missing_selected_environment_variable_names_the_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY_3"):
                GoogleGenAIClient(
                    model="stub", temperature=0.0, seed=7, api_key_env="GEMINI_API_KEY_3"
                )

    def test_rejects_mismatched_choice_text(self):
        with self.assertRaises(ValueError):
            parse_gemini_payload(
                {"reasoning": "Heat moves.", "final_answer_label": "B", "final_answer_text": "radiation"},
                record=record(), task="normal", model="stub", usage=Usage(), finish_reason=None, raw_response={},
            )
