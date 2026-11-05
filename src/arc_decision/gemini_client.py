"""Gemini structured-generation client, imported lazily to keep offline tests dependency-free."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Protocol

from .models import ArcRecord, GeminiResult, Usage
from .prompts import GenerationTask, build_gemini_prompt, gemini_response_schema


class GeminiClient(Protocol):
    def generate(
        self,
        record: ArcRecord,
        *,
        task: GenerationTask,
        poem_form: str,
        strict_separation: bool = False,
    ) -> GeminiResult:
        """Generate one structured rationale for one ARC record."""


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_json"):
        return json.loads(value.to_json())
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": str(value)}


def _usage_from_metadata(value: Any) -> Usage:
    mapping = _as_mapping(value)
    return Usage(
        input_tokens=mapping.get("prompt_token_count", mapping.get("input_tokens")),
        output_tokens=mapping.get("candidates_token_count", mapping.get("output_tokens")),
        total_tokens=mapping.get("total_token_count", mapping.get("total_tokens")),
    )


def parse_gemini_payload(
    payload: dict[str, Any],
    *,
    record: ArcRecord,
    task: GenerationTask,
    model: str,
    usage: Usage,
    finish_reason: str | None,
    raw_response: dict[str, Any],
) -> GeminiResult:
    """Validate structured output against the per-example answer label set."""
    reasoning = payload.get("reasoning")
    label = payload.get("final_answer_label")
    answer_text = payload.get("final_answer_text")
    poem_form = payload.get("poem_form") if task == "poetic" else None
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Gemini response is missing a non-empty reasoning field.")
    if not isinstance(label, str) or label.strip() not in record.choice_map:
        raise ValueError("Gemini response has a final_answer_label outside the source choices.")
    label = label.strip()
    if answer_text is not None:
        if not isinstance(answer_text, str):
            raise ValueError("Gemini final_answer_text must be a string when present.")
        if answer_text.strip() != record.choice_map[label].strip():
            raise ValueError("Gemini final_answer_text does not match final_answer_label's source choice.")
    if task == "poetic" and (not isinstance(poem_form, str) or not poem_form.strip()):
        raise ValueError("Poetic Gemini response is missing poem_form.")
    return GeminiResult(
        reasoning=reasoning.strip(),
        final_answer_label=label,
        final_answer_text=answer_text.strip() if isinstance(answer_text, str) else None,
        poem_form=poem_form.strip() if isinstance(poem_form, str) else None,
        model=model,
        usage=usage,
        finish_reason=finish_reason,
        raw_response=raw_response,
    )


class GoogleGenAIClient:
    """Official Google Gen AI SDK adapter for one question at a time."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        seed: int | None,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.api_key_env = api_key_env
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is required for Gemini generation (selected by gemini_api_key_env)."
            )

    def generate(
        self,
        record: ArcRecord,
        *,
        task: GenerationTask,
        poem_form: str,
        strict_separation: bool = False,
    ) -> GeminiResult:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("Gemini generation requires the 'google-genai' package.") from error

        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": gemini_response_schema(task),
        }
        if self.seed is not None:
            config_kwargs["seed"] = self.seed
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=build_gemini_prompt(
                record,
                task=task,
                poem_form=poem_form,
                strict_separation=strict_separation,
            ),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemini returned no structured response text.")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("Gemini returned invalid JSON despite structured output being requested.") from error
        candidates = getattr(response, "candidates", None) or []
        finish_reason = str(getattr(candidates[0], "finish_reason", "")) or None if candidates else None
        raw_response = _as_mapping(response)
        return parse_gemini_payload(
            payload,
            record=record,
            task=task,
            model=str(getattr(response, "model_version", None) or self.model),
            usage=_usage_from_metadata(getattr(response, "usage_metadata", None)),
            finish_reason=finish_reason,
            raw_response=raw_response,
        )


def result_as_dict(result: GeminiResult) -> dict[str, Any]:
    """Return a JSON-safe normalized Gemini result for the durable attempt ledger."""
    value = asdict(result)
    value["usage"] = result.usage.to_dict()
    return value
