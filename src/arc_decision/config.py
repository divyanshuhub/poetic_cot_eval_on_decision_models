"""Configuration shared by the CLI and pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_CONFIG_ALIASES = {
    "easy": "ARC-Easy",
    "challenge": "ARC-Challenge",
    "normal": "ARC-Challenge",
    "arc-easy": "ARC-Easy",
    "arc-challenge": "ARC-Challenge",
    "all": "all",
    "customer_service": "customer_service",
    "invoice_processing": "invoice_processing",
    "security_incidents": "security_incidents",
    "agent_trace_observability": "agent_trace_observability",
    "default": "default",
}
# DECISION_MODELS = ("laya", "clef", "kev")
DECISION_MODELS = ("laya", "clef", "kev", "jev")
POEM_FORMS = ("haiku", "tanka", "limerick")
SPLITS = ("train", "validation", "test")


def canonical_dataset_config(value: str) -> str:
    """Map a user-friendly dataset configuration alias to its canonical name."""
    norm = value.strip().lower()
    canonical = DATASET_CONFIG_ALIASES.get(norm)
    if canonical is not None:
        return canonical
    return value.strip()


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for one output directory/run."""

    output_dir: Path
    dataset_config: str = "ARC-Easy"
    split: str = "test"
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    decision_model: str = "laya"
    laya_finetuned_checkpoint: str | None = None
    poem_form: str = "tanka"
    temperature: float = 0.0
    seed: int | None = 7
    gemini_requests_per_minute: float = 4.0
    limit: int | None = None
    resume: bool = False
    strict: bool = False
    max_retries: int = 2
    request_timeout: float = 60.0
    include_flagged_reasoning: bool = False
    force_regenerate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "dataset_config", canonical_dataset_config(self.dataset_config))
        object.__setattr__(self, "decision_model", self.decision_model.lower())
        object.__setattr__(self, "poem_form", self.poem_form.lower())
        if self.laya_finetuned_checkpoint is not None:
            checkpoint = self.laya_finetuned_checkpoint.strip()
            if not checkpoint:
                raise ValueError("laya_finetuned_checkpoint cannot be empty.")
            if self.decision_model != "laya":
                raise ValueError("laya_finetuned_checkpoint can only be used with decision_model='laya'.")
            object.__setattr__(self, "laya_finetuned_checkpoint", checkpoint)
        if self.split not in SPLITS:
            raise ValueError(f"Unsupported split {self.split!r}; choose one of: {', '.join(SPLITS)}.")
        if not self.gemini_api_key_env.strip():
            raise ValueError("gemini_api_key_env must name the environment variable containing the Gemini API key.")
        if self.decision_model not in DECISION_MODELS:
            raise ValueError(
                f"Unsupported decision model {self.decision_model!r}; choose one of: "
                f"{', '.join(DECISION_MODELS)}."
            )
        if self.poem_form not in POEM_FORMS:
            raise ValueError(f"Unsupported poem form {self.poem_form!r}; choose one of: {', '.join(POEM_FORMS)}.")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least one when provided.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.gemini_requests_per_minute <= 0:
            raise ValueError("gemini_requests_per_minute must be positive.")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive.")

    def manifest_values(self) -> dict[str, Any]:
        """Return JSON-safe configuration metadata for the run manifest."""
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values
