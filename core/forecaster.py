"""
Phase 3.3 — Core forecasting utilities.

temperature_scale(): Platt scaling to sharpen or smooth probability estimates.
  T < 1 → sharpens (more confident)
  T = 1 → identity (no change)
  T > 1 → smooths (more uncertain)

The optimal T is found by the calibrator's walk-forward optimizer.
"""

from __future__ import annotations

import math
from typing import Optional


def temperature_scale(prob_raw: float, T: float) -> float:
    """
    Apply temperature (Platt) scaling to a raw probability.

    Args:
        prob_raw: Uncalibrated probability in (0, 1).
        T:        Temperature parameter. 1.0 = no change.

    Returns:
        Calibrated probability in (0, 1).
    """
    # Clamp input away from boundaries to avoid log(0)
    prob_raw = max(1e-6, min(1 - 1e-6, prob_raw))
    T = max(1e-4, T)

    logit = math.log(prob_raw / (1.0 - prob_raw))
    scaled_logit = logit / T
    # Clamp scaled_logit to avoid exp overflow → exact 0 or 1
    scaled_logit = max(-30.0, min(30.0, scaled_logit))
    result = 1.0 / (1.0 + math.exp(-scaled_logit))
    return max(1e-6, min(1.0 - 1e-6, result))


def prob_above_threshold(
    consensus_f: float,
    threshold_f: float,
    base_std_f: float,
    temp_T: float = 1.0,
) -> float:
    """
    P(actual_high > threshold_f) under N(consensus_f, base_std_f²),
    then calibrated with temperature scaling.

    Args:
        consensus_f:  Forecast consensus temperature.
        threshold_f:  Contract threshold (e.g. 75°F).
        base_std_f:   Forecast uncertainty std-dev in °F.
        temp_T:       Temperature scaling parameter.

    Returns:
        Calibrated probability in (0, 1).
    """
    if base_std_f <= 0:
        raw = 1.0 if consensus_f > threshold_f else 0.0
    else:
        z = (threshold_f - consensus_f) / (base_std_f * math.sqrt(2))
        raw = 0.5 * math.erfc(z)

    return temperature_scale(raw, temp_T)


def brier_score(predicted: float, outcome: float) -> float:
    """Brier score for a single prediction. outcome ∈ {0.0, 1.0}."""
    return (predicted - outcome) ** 2


def brier_decomposition(
    predictions: list[float],
    outcomes: list[float],
) -> dict[str, float]:
    """
    Brier score decomposition into reliability, resolution, and uncertainty.

    Returns:
      {
        "brier":       overall Brier score,
        "reliability": calibration component (lower = better calibrated),
        "resolution":  resolution component (higher = more informative),
        "uncertainty": base rate uncertainty (irreducible),
        "brier_index": (1 - sqrt(brier)) * 100 — 0–100 skill score,
      }
    """
    if not predictions:
        return {"brier": 0.25, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.25, "brier_index": 0.0}

    n = len(predictions)
    briers = [brier_score(p, o) for p, o in zip(predictions, outcomes)]
    avg_brier = sum(briers) / n

    base_rate = sum(outcomes) / n
    uncertainty = base_rate * (1 - base_rate)

    # Reliability: mean squared deviation of predicted from observed freq
    # Simple approximation: variance of (predicted - outcome)
    reliability = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n - \
                  (sum(p - o for p, o in zip(predictions, outcomes)) / n) ** 2
    reliability = max(0.0, reliability)

    resolution = uncertainty - (avg_brier - reliability)

    brier_index = (1.0 - math.sqrt(max(0.0, avg_brier))) * 100.0

    return {
        "brier": avg_brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_index": brier_index,
    }
