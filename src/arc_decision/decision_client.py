"""Local open-weight adapters for Laya, Kev, and Clef decision-model runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DECISION_MODELS
from .models import ArcRecord, DecisionResult, Usage
from .prompts import build_choice_question
from .storage import sanitized_error


class DecisionClient(Protocol):
    def decide(self, state: dict[str, Any], record: ArcRecord) -> DecisionResult:
        """Choose exactly one label from the record's bounded choice set."""


@dataclass(frozen=True)
class LocalSystemOneConfig:
    """Connection settings for a locally hosted Kev or Clef SystemOne server."""

    decision_model: str
    model_id: str
    api_url: str
    api_key: str | None
    timeout: float


_LOCAL_SERVER_DEFAULTS = {
    "kev": {
        "model_id": "jaredpalmer/kev-4b",
        "api_url": "http://127.0.0.1:8008/v1/systemone",
    },
    "clef": {
        "model_id": "Cloudflare/clef-flash",
        "api_url": "http://127.0.0.1:30000/v1/systemone",
    },
}


def local_systemone_config(decision_model: str, *, timeout: float) -> LocalSystemOneConfig:
    """Resolve a localhost server configuration; credentials are optional, never required."""
    normalized = decision_model.lower()
    if normalized not in _LOCAL_SERVER_DEFAULTS:
        raise ValueError(f"{decision_model!r} does not use the local SystemOne server adapter.")
    defaults = _LOCAL_SERVER_DEFAULTS[normalized]
    prefix = normalized.upper()
    return LocalSystemOneConfig(
        decision_model=normalized,
        model_id=os.environ.get(f"{prefix}_MODEL_ID", defaults["model_id"]),
        api_url=os.environ.get(f"{prefix}_API_URL", defaults["api_url"]),
        api_key=os.environ.get(f"{prefix}_API_KEY") or None,
        timeout=timeout,
    )


def _usage(value: Any) -> Usage:
    mapping = value if isinstance(value, Mapping) else {}
    return Usage(
        input_tokens=mapping.get("input_tokens"),
        output_tokens=mapping.get("output_tokens"),
        total_tokens=mapping.get("total_tokens"),
    )


def _decision_result_from_response(
    raw_response: Mapping[str, Any],
    *,
    record: ArcRecord,
    fallback_model: str,
    request_id: str | None = None,
) -> DecisionResult:
    answer = raw_response.get("answers", {}).get("arc_answer")
    if not isinstance(answer, Mapping):
        raise ValueError("Local decision response has no arc_answer Choice result.")
    label = answer.get("choice")
    if not isinstance(label, str) or label not in record.choice_map:
        raise ValueError("Local decision response returned a label outside the source choice set.")
    probabilities = answer.get("probabilities") or {}
    if not isinstance(probabilities, Mapping):
        raise ValueError("Local decision probabilities must be a mapping when present.")
    return DecisionResult(
        predicted_answer_label=label,
        confidence=float(answer["confidence"]) if answer.get("confidence") is not None else None,
        probabilities={str(key): float(value) for key, value in probabilities.items()},
        model=str(raw_response.get("model") or fallback_model),
        usage=_usage(raw_response.get("usage")),
        request_id=request_id or raw_response.get("request_id"),
        raw_response=dict(raw_response),
    )


class LayaLocalClient:
    """Direct local Laya SDK adapter; the first construction downloads its open weights."""

    FINETUNED_MODEL_NAME = "arc_finetuned"

    def __init__(
        self,
        *,
        device: str | None = None,
        checkpoint: str | None = None,
        finetuned_checkpoint: str | None = None,
        preload: bool = False,
        router: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.finetuned_checkpoint = finetuned_checkpoint
        if router is not None:
            self.router = router
            return
        try:
            from laya import Router
        except ImportError as error:
            raise RuntimeError("Local Laya inference requires `pip install 'arc-decision-reasoning[laya]'`.") from error
        router_kwargs: dict[str, Any] = {"preload": preload}
        if device:
            router_kwargs["device"] = device
        if finetuned_checkpoint:
            router_kwargs["models"] = {self.FINETUNED_MODEL_NAME: finetuned_checkpoint}
        self.router = Router(**router_kwargs)

    @classmethod
    def from_environment(cls, *, finetuned_checkpoint: str | None = None) -> "LayaLocalClient":
        preload = os.environ.get("LAYA_PRELOAD", "0").strip().lower() in {"1", "true", "yes"}
        return cls(
            device=os.environ.get("LAYA_DEVICE") or None,
            checkpoint=os.environ.get("LAYA_CHECKPOINT") or None,
            finetuned_checkpoint=finetuned_checkpoint,
            preload=preload,
        )

    def decide(self, state: dict[str, Any], record: ArcRecord) -> DecisionResult:
        predict_kwargs: dict[str, Any] = {}
        if self.finetuned_checkpoint:
            predict_kwargs["model"] = self.FINETUNED_MODEL_NAME
        elif self.checkpoint:
            predict_kwargs["model"] = self.checkpoint
        response = self.router.predict(
            state,
            {"arc_answer": build_choice_question(record)},
            **predict_kwargs,
        )
        if not isinstance(response, Mapping):
            raise ValueError("Local Laya returned an invalid response.")
        routing = response.get("routing")
        routing_mapping = routing if isinstance(routing, Mapping) else {}
        model = (
            self.finetuned_checkpoint
            or routing_mapping.get("repo")
            or routing_mapping.get("model")
            or self.checkpoint
            or "convaiinnovations/laya"
        )
        return _decision_result_from_response(response, record=record, fallback_model=str(model))


class LocalSystemOneClient:
    """HTTP client for a self-hosted Kev or Clef `/v1/systemone` server."""

    def __init__(self, config: LocalSystemOneConfig) -> None:
        self.config = config

    def decide(self, state: dict[str, Any], record: ArcRecord) -> DecisionResult:
        payload = {
            "state": state,
            "model": self.config.model_id,
            "questions": {"arc_answer": build_choice_question(record)},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            self.config.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw_bytes = response.read()
                raw_response = json.loads(raw_bytes.decode("utf-8"))
                request_id = response.headers.get("X-Request-ID") or response.headers.get("Request-ID")
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")[:2000]
            # raise RuntimeError(f"Local {self.config.decision_model} server returned HTTP {error.code}: {sanitized_error(details)}") from error
            # In the HTTPError handler:
            raise RuntimeError(
                f"{self.config.decision_model} SystemOne API returned HTTP "
                f"{error.code}: {sanitized_error(details)}"
            ) from error
            

        except URLError as error:
            # raise RuntimeError(
            #     f"Could not reach local {self.config.decision_model} server at {self.config.api_url}: "
            #     f"{sanitized_error(error)}"
            # ) from error
            # In the URLError handler:
            raise RuntimeError(
                f"Could not reach {self.config.decision_model} SystemOne API "
                f"at {self.config.api_url}: {sanitized_error(error)}"
            ) from error

        if not isinstance(raw_response, Mapping):
            raise ValueError("Local decision server returned a non-object JSON response.")
        return _decision_result_from_response(
            raw_response,
            record=record,
            fallback_model=self.config.model_id,
            request_id=request_id,
        )

def jev_systemone_config(*, timeout: float) -> LocalSystemOneConfig:
    """Resolve the hosted JEV API configuration."""
    api_key = (
        os.environ.get("JEV_API_KEY", "").strip()
        or os.environ.get("TYPESAFE_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            "JEV evaluation requires JEV_API_KEY or TYPESAFE_API_KEY "
            "in the project .env file."
        )

    model_id = os.environ.get("JEV_MODEL_ID", "jev-latest").strip()
    api_url = os.environ.get(
        "JEV_API_URL", "https://api.typesafe.ai/v1/systemone"
    ).strip()

    if not model_id:
        raise ValueError("JEV_MODEL_ID cannot be empty.")
    if not api_url.startswith("https://"):
        raise ValueError("JEV_API_URL must use HTTPS.")

    return LocalSystemOneConfig(
        decision_model="jev",
        model_id=model_id,
        api_url=api_url,
        api_key=api_key,
        timeout=timeout,
    )


def create_decision_client(
    decision_model: str,
    *,
    timeout: float,
    laya_finetuned_checkpoint: str | None = None,
) -> DecisionClient:
    """Create a local decision adapter or the hosted JEV adapter."""
    normalized = decision_model.lower()
    if normalized not in DECISION_MODELS:
        raise ValueError(f"Unsupported decision model {decision_model!r}.")

    if normalized == "laya":
        return LayaLocalClient.from_environment(
            finetuned_checkpoint=laya_finetuned_checkpoint
        )

    if normalized == "jev":
        return LocalSystemOneClient(jev_systemone_config(timeout=timeout))

    return LocalSystemOneClient(
        local_systemone_config(normalized, timeout=timeout)
    )

