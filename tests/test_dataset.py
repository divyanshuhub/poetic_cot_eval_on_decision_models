import unittest

from arc_decision.dataset import DatasetValidationError, normalize_arc_row, normalize_rows


def sample_row(answer_key="B"):
    return {
        "question": "Which process moves heat by direct contact?",
        "choices": {"label": ["A", "B"], "text": ["radiation", "conduction"]},
        "answerKey": answer_key,
    }


class DatasetNormalizationTests(unittest.TestCase):
    def test_normalizes_preserving_choice_order(self):
        record = normalize_arc_row(sample_row(), dataset_config="ARC-Easy", split="test", source_row_index=4)
        self.assertEqual(record.example_id, "arc_arc_easy__test__000004")
        self.assertEqual([choice.label for choice in record.choices], ["A", "B"])
        self.assertEqual(record.gold_answer_text, "conduction")

    def test_rejects_unmapped_answer_key(self):
        with self.assertRaises(DatasetValidationError):
            normalize_arc_row(sample_row("Z"), dataset_config="ARC-Easy", split="test", source_row_index=0)

    def test_records_invalid_rows_without_strict_failure(self):
        result = normalize_rows([sample_row(), sample_row("Z")], dataset_config="ARC-Easy", split="test")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.skipped), 1)
        with self.assertRaises(DatasetValidationError):
            normalize_rows([sample_row(), sample_row("Z")], dataset_config="ARC-Easy", split="test", strict=True)
