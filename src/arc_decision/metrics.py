"""Offline metrics for condition-level and paired ARC decision comparisons."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Iterable, Sequence


def as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def as_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def numeric_summary(values: Iterable[object]) -> dict[str, float | int | None]:
    numbers = [number for value in values if (number := as_number(value)) is not None]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "median": median(numbers) if numbers else None,
        "p95": percentile(numbers, 0.95),
    }


def paired_accuracy_difference(
    baseline: Sequence[bool],
    comparison: Sequence[bool],
    *,
    samples: int = 2000,
    seed: int = 7,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Return comparison minus baseline with a deterministic paired bootstrap interval."""
    if len(baseline) != len(comparison):
        raise ValueError("Paired correctness vectors must have identical lengths.")
    if not baseline:
        return {"sample_size": 0, "difference_pp": None, "ci_low_pp": None, "ci_high_pp": None}
    differences = [float(compare) - float(base) for base, compare in zip(baseline, comparison, strict=True)]
    point = sum(differences) / len(differences)
    random_source = random.Random(seed)
    bootstraps = []
    for _ in range(samples):
        bootstraps.append(sum(differences[random_source.randrange(len(differences))] for _ in differences) / len(differences))
    alpha = (1 - confidence) / 2
    return {
        "sample_size": len(differences),
        "difference_pp": point * 100,
        "ci_low_pp": percentile(bootstraps, alpha) * 100 if bootstraps else None,
        "ci_high_pp": percentile(bootstraps, 1 - alpha) * 100 if bootstraps else None,
    }
