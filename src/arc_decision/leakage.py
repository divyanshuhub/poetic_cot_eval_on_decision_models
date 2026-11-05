"""Transparent diagnostics for rationale answer leakage and poetic-form metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ArcRecord


@dataclass(frozen=True)
class LeakageReport:
    status: str
    hard_reasons: tuple[str, ...]
    review_reasons: tuple[str, ...]

    @property
    def is_hard_flagged(self) -> bool:
        return self.status == "hard_flagged"


def _normalized(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def detect_answer_leakage(reasoning: str, record: ArcRecord) -> LeakageReport:
    """Flag obvious answer leakage without pretending to solve semantic leakage perfectly."""
    text = reasoning.strip()
    labels = "|".join(re.escape(choice.label) for choice in record.choices)
    hard_patterns = {
        "final_answer_marker": r"\b(?:final|correct|right)\s+(?:answer|choice|option)\b",
        "option_reference": rf"\b(?:option|choice)\s*(?:number\s*)?(?:{labels})\b",
        "answer_label": rf"\banswer\s*(?:is|:|=)\s*\(?\s*(?:{labels})\s*\)?\b",
        "label_correctness": rf"\b(?:{labels})\s+(?:is|was)\s+(?:the\s+)?(?:correct|right)\b",
    }
    hard_reasons = tuple(name for name, pattern in hard_patterns.items() if re.search(pattern, text, re.IGNORECASE))
    normalized_reasoning = _normalized(text)
    review_reasons: list[str] = []
    for choice in record.choices:
        normalized_choice = _normalized(choice.text)
        if len(normalized_choice) >= 4 and normalized_choice in normalized_reasoning:
            review_reasons.append(f"contains_exact_choice_text:{choice.label}")
    status = "hard_flagged" if hard_reasons else "review_flagged" if review_reasons else "clean"
    return LeakageReport(status=status, hard_reasons=hard_reasons, review_reasons=tuple(review_reasons))


def validate_poem_form(requested_form: str, returned_form: str | None, reasoning: str) -> str:
    """Provide a lightweight, non-destructive poetic-form diagnostic."""
    if returned_form is None or returned_form.strip().lower() != requested_form.lower():
        return "reported_form_mismatch"
    expected_lines = {"haiku": 3, "tanka": 5, "limerick": 5}.get(requested_form.lower())
    line_count = len([line for line in reasoning.splitlines() if line.strip()])
    if expected_lines is not None and line_count != expected_lines:
        return f"reported_form_line_count_{line_count}_expected_{expected_lines}"
    return "reported_form_matches_lightweight_check"
