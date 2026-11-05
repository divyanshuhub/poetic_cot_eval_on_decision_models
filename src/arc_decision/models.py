"""Typed records exchanged between the dataset, client, storage, and analysis layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Choice:
    label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArcRecord:
    example_id: str
    dataset_id: str
    dataset_config: str
    split: str
    source_row_index: int
    question: str
    choices: tuple[Choice, ...]
    gold_answer_label: str
    gold_answer_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["choices"] = [choice.to_dict() for choice in self.choices]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArcRecord":
        return cls(
            example_id=str(value["example_id"]),
            dataset_id=str(value["dataset_id"]),
            dataset_config=str(value["dataset_config"]),
            split=str(value["split"]),
            source_row_index=int(value["source_row_index"]),
            question=str(value["question"]),
            choices=tuple(Choice(str(item["label"]), str(item["text"])) for item in value["choices"]),
            gold_answer_label=str(value["gold_answer_label"]),
            gold_answer_text=str(value["gold_answer_text"]),
            metadata=dict(value.get("metadata") or {}),
        )

    @property
    def choice_map(self) -> dict[str, str]:
        return {choice.label: choice.text for choice in self.choices}


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class GeminiResult:
    reasoning: str
    final_answer_label: str
    final_answer_text: str | None
    poem_form: str | None
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    predicted_answer_label: str
    confidence: float | None
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
