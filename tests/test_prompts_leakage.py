import json
import unittest

from arc_decision.leakage import detect_answer_leakage, validate_poem_form
from arc_decision.models import ArcRecord, Choice
from arc_decision.prompts import build_decision_state, build_gemini_prompt, public_question_payload


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


class PromptAndLeakageTests(unittest.TestCase):
    def test_poetic_prompt_requires_form_specific_verse(self):
        requirements = {
            "haiku": ("exactly three non-empty verse lines", "5-7-5 syllable pattern"),
            "tanka": ("exactly five non-empty verse lines", "5-7-5-7-7 syllable pattern"),
            "limerick": ("exactly five non-empty verse lines", "AABBA end-rhyme scheme"),
        }
        for poem_form, expected_phrases in requirements.items():
            with self.subTest(poem_form=poem_form):
                prompt = build_gemini_prompt(record(), task="poetic", poem_form=poem_form)
                for phrase in expected_phrases:
                    self.assertIn(phrase, prompt)
                self.assertIn("not prose broken into lines or disconnected brief statements", prompt)
                self.assertIn("Do not put an answer label", prompt)

    def test_public_state_excludes_gold_and_final_fields(self):
        state = build_decision_state(record(), condition="normal_reasoning", rationale="Heat transfers by contact.")
        serialized = json.dumps(state)
        self.assertNotIn("gold_answer_label", serialized)
        self.assertNotIn("final_answer_label", serialized)
        self.assertNotIn("final_answer_text", serialized)
        self.assertNotIn("gold_answer_text", json.dumps(public_question_payload(record())))

    def test_flags_direct_answer_revelation(self):
        report = detect_answer_leakage("The correct answer is B.", record())
        self.assertEqual(report.status, "hard_flagged")
        self.assertIn("final_answer_marker", report.hard_reasons)

    def test_marks_exact_choice_overlap_for_review(self):
        report = detect_answer_leakage("A process called conduction transfers thermal energy.", record())
        self.assertEqual(report.status, "review_flagged")
        self.assertIn("contains_exact_choice_text:B", report.review_reasons)

    def test_lightweight_poem_check_does_not_rewrite_content(self):
        status = validate_poem_form("tanka", "tanka", "one\ntwo\nthree\nfour\nfive")
        self.assertEqual(status, "reported_form_matches_lightweight_check")
