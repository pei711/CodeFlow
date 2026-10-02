"""Conservative token estimation used by the context orchestrator.

The runtime deliberately does not depend on a provider-specific tokenizer.  It
uses a small, explainable heuristic and lets provider usage telemetry calibrate
that heuristic for a stable model identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CODE_HINT = re.compile(r"(?:```|[{}();]|\\\\|/|\.json\\b|\.py\\b|\.ts\\b|pytest|traceback|error:)", re.IGNORECASE)
CJK_HINT = re.compile(r"[\u3400-\u9fff]")


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens without pretending to be an exact tokenizer."""

    value = str(text or "")
    if not value:
        return 0
    chars = len(value)
    if len(CJK_HINT.findall(value)) >= max(8, chars // 5):
        return max(1, round(chars * 10 / 18))
    if CODE_HINT.search(value):
        return max(1, round(chars * 10 / 32))
    return max(1, round(chars / 4))


def estimate_sections_tokens(sections: dict[str, str]) -> int:
    return sum(estimate_text_tokens(value) for value in sections.values())


@dataclass(frozen=True)
class TokenEstimate:
    estimated_tokens: int
    usage_source: str = "estimated_proxy"
    calibration_ratio: float = 1.0


def calibrated_estimate(text: str, calibration_ratio: float | None = None) -> TokenEstimate:
    base = estimate_text_tokens(text)
    ratio = float(calibration_ratio or 1.0)
    ratio = min(3.0, max(0.25, ratio))
    return TokenEstimate(
        estimated_tokens=max(0, round(base * ratio)),
        usage_source="calibrated_proxy" if ratio != 1.0 else "estimated_proxy",
        calibration_ratio=ratio,
    )


def context_budget_chars(
    context_window: int | None,
    output_reserve_tokens: int = 16_384,
    utilization_ratio: float = 0.50,
    min_budget_chars: int = 60_000,
    max_budget_chars: int = 800_000,
) -> dict:
    """Return the article's dynamic budget calculation as inspectable metadata."""

    if not context_window:
        return {
            "enabled": False,
            "context_window": None,
            "effective_tokens": None,
            "budget_tokens": None,
            "budget_chars": None,
        }
    effective = max(1, int(context_window) - int(output_reserve_tokens))
    budget_tokens = max(1, int(effective * float(utilization_ratio)))
    # The article's 60k lower bound is useful for large-context models, but it
    # must not exceed the effective window of a small local model.
    effective_chars = effective * 4
    budget_chars = min(max_budget_chars, max(min_budget_chars, budget_tokens * 4), effective_chars)
    return {
        "enabled": True,
        "context_window": int(context_window),
        "output_reserve_tokens": int(output_reserve_tokens),
        "utilization_ratio": float(utilization_ratio),
        "effective_tokens": effective,
        "budget_tokens": budget_tokens,
        "budget_chars": int(budget_chars),
    }
