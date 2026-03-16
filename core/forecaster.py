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
from typing import Any, Optional


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


def dynamic_std_f(
    city: str,
    target_date: str,
    model_spread: float,
    lead_hours: float,
    base_std_f: float,
) -> float:
    """
    Compute dynamic forecast uncertainty (std-dev in °F).

    Adjusts base_std_f upward for:
      - High model spread (disagreement between models)
      - Long lead time (more uncertainty farther out)
      - Summer months in hot cities (higher temperature variance)

    Args:
        city:         City name (used for seasonal adjustment lookup).
        target_date:  ISO date string "YYYY-MM-DD".
        model_spread: Std-dev of model ensemble temperatures (°F).
        lead_hours:   Hours between now and target settlement.
        base_std_f:   Baseline std-dev from params.

    Returns:
        Adjusted std-dev in °F (always >= base_std_f).
    """
    std = base_std_f

    # Lead-time scaling: add 0.5°F per 24h of lead beyond 24h, capped at +3°F
    lead_days = max(0.0, (lead_hours - 24.0) / 24.0)
    std += min(3.0, lead_days * 0.5)

    # Model spread contribution: weight model disagreement into uncertainty
    # Add 30% of the model spread on top
    std += 0.3 * max(0.0, model_spread)

    # Seasonal boost for summer (Jun-Aug) — higher variance
    try:
        month = int(target_date[5:7])
        if 6 <= month <= 8:
            std *= 1.15
    except (ValueError, IndexError):
        pass

    return max(base_std_f, std)


def monte_carlo_prob(
    model_forecasts: list[float],
    market_type: str,
    high_f: float | None,
    low_f: float | None,
    sigma: float,
    n_samples: int = 2000,
    seed: int | None = None,
) -> float:
    """
    Estimate P(YES) via Monte Carlo sampling from the model ensemble.

    For each model forecast, draw n_samples/n_models samples from
    N(forecast, sigma²). Count samples satisfying the contract condition.
    Uses dynamic sigma (from dynamic_std_f) which already accounts for
    model spread, lead time, and season — no additional adjustments needed.

    Args:
        model_forecasts: List of raw (bias-corrected) model temperature forecasts.
        market_type:     "above" | "below" | "band".
        high_f:          Upper threshold (°F).
        low_f:           Lower threshold for band contracts (°F).
        sigma:           Per-model forecast std-dev (from dynamic_std_f).
        n_samples:       Total samples across all models.
        seed:            Random seed for reproducibility (None = random).

    Returns:
        Probability in [0.01, 0.99].
    """
    if not model_forecasts:
        return 0.5

    import random
    rng = random.Random(seed)
    n_models = len(model_forecasts)
    per_model = max(1, n_samples // n_models)

    hits = 0
    total = 0
    mtype = (market_type or "above").lower()

    for mu in model_forecasts:
        for _ in range(per_model):
            x = rng.gauss(mu, sigma)
            total += 1
            if mtype == "band":
                # Match analytic: P(low_f - 0.5 <= X <= high_f + 0.5)
                if low_f is not None and high_f is not None:
                    if (low_f - 0.5) <= x <= (high_f + 0.5):
                        hits += 1
            elif mtype == "below":
                # Match analytic: P(X <= high_f + 0.5)
                if high_f is not None and x <= (high_f + 0.5):
                    hits += 1
            else:  # above
                # Match analytic: P(X >= high_f - 0.5)
                if high_f is not None and x >= (high_f - 0.5):
                    hits += 1

    if total == 0:
        return 0.5
    prob = hits / total
    return max(0.01, min(0.99, prob))


def _compute_fair_value_for_market(
    consensus_f: float,
    market_type: str,
    high_f: "float | None",
    low_f: "float | None",
    std_f: float,
) -> "float | None":
    """
    Compute P(YES) for any market type using Gaussian CDF with ±0.5 rounding.

    above: P(X >= high_f - 0.5)
    below: P(X <= high_f + 0.5)
    band:  P(low_f - 0.5 <= X <= high_f + 0.5)
    """
    if std_f <= 0:
        return None

    def cdf(x: float) -> float:
        z = (x - consensus_f) / (std_f * math.sqrt(2))
        return 0.5 * (1.0 + math.erf(z))

    mtype = (market_type or "above").lower()

    if mtype == "band":
        if low_f is None or high_f is None:
            return None
        prob = cdf(high_f + 0.5) - cdf(low_f - 0.5)
    elif mtype == "below":
        if high_f is None:
            return None
        prob = cdf(high_f + 0.5)
    else:  # "above"
        if high_f is None:
            return None
        prob = 1.0 - cdf(high_f - 0.5)

    return max(0.01, min(0.99, prob))


def compute_fair_value(
    forecasts: list,
    city: str,
    target_date: str,
    market_type: str,
    high_f: "float | None",
    low_f: "float | None",
    params: Any,
    conn=None,
    use_mc: bool = True,
) -> "tuple[float | None, float, float, int]":
    """
    Single canonical fair value — used by entry, exit, lifecycle, calibration.

    Applies the full pipeline: bias correction, dynamic std, MC/analytic,
    temperature scaling.

    Returns:
        (fair_value, consensus_f, std_f, n_models)
        fair_value is None if no forecasts are available for city/date.
    """
    import statistics as _st
    from datetime import datetime, timezone as _tz

    relevant = [f for f in forecasts if f.city == city and f.target_date == target_date]
    if not relevant:
        return None, 0.0, getattr(params, "base_std_f", 5.0), 0

    # Lead hours (for dynamic std)
    try:
        eod = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
        lead_hours = max(0.0, (eod - datetime.now(_tz.utc)).total_seconds() / 3600)
    except Exception:
        lead_hours = 48.0

    # Bias correction (requires DB connection)
    bias_table = None
    fine_bias_table = None
    if conn is not None:
        try:
            from research.bias_correction import learn_biases, learn_fine_biases
            bias_table = learn_biases(conn)
            fine_bias_table = learn_fine_biases(conn)
        except Exception:
            pass

    if bias_table:
        from research.bias_correction import apply_bias
        highs = [
            apply_bias(
                f.predicted_high_f, city, f.model_name, bias_table,
                fine_bias_table=fine_bias_table,
                target_date=target_date,
                lead_hours=lead_hours,
            )
            for f in relevant
        ]
    else:
        highs = [f.predicted_high_f for f in relevant]

    consensus_f = _st.mean(highs)
    agreement = _st.stdev(highs) if len(highs) > 1 else 0.0

    ensemble_samples: list[float] = []
    for f, corrected_high in zip(relevant, highs):
        members = getattr(f, "ensemble_members_f", None) or []
        if not members:
            continue
        delta = corrected_high - float(f.predicted_high_f)
        ensemble_samples.extend([float(m) + delta for m in members])

    # When true ensemble members are available, use their empirical spread as the
    # uncertainty input and their rank probabilities as the primary fair value.
    if ensemble_samples:
        agreement = _st.stdev(ensemble_samples) if len(ensemble_samples) > 1 else agreement

    std_f = dynamic_std_f(
        city=city,
        target_date=target_date,
        model_spread=agreement,
        lead_hours=lead_hours,
        base_std_f=getattr(params, "base_std_f", 5.0),
    )

    use_empirical = bool(ensemble_samples) and getattr(params, "use_empirical_ensemble", True)
    if use_empirical:
        fair_value = empirical_prob_for_market(ensemble_samples, market_type, high_f, low_f)
    elif use_mc and highs:
        fair_value = monte_carlo_prob(
            model_forecasts=highs,
            market_type=market_type,
            high_f=high_f,
            low_f=low_f,
            sigma=std_f,
            n_samples=getattr(params, "monte_carlo_samples", 2000),
        )
    else:
        fair_value = _compute_fair_value_for_market(consensus_f, market_type, high_f, low_f, std_f)
        if fair_value is None:
            return None, consensus_f, std_f, len(highs)

    # Temperature scaling
    T = getattr(params, "temp_scaling_T", 1.0)
    if T != 1.0 and 0 < fair_value < 1:
        logit = math.log(fair_value / (1.0 - fair_value))
        fair_value = 1.0 / (1.0 + math.exp(-logit / T))

    fair_value += _segment_probability_adjustment(conn, market_type, lead_hours, agreement, city)
    return max(0.01, min(0.99, fair_value)), consensus_f, std_f, len(highs)


def empirical_prob_for_market(
    samples_f: list[float],
    market_type: str,
    high_f: float | None,
    low_f: float | None,
) -> float | None:
    """Empirical P(YES) from true ensemble-member temperatures when available."""
    if not samples_f:
        return None
    mtype = (market_type or "above").lower()
    hits = 0
    total = len(samples_f)
    for x in samples_f:
        if mtype == "band":
            if low_f is not None and high_f is not None and (low_f - 0.5) <= x <= (high_f + 0.5):
                hits += 1
        elif mtype == "below":
            if high_f is not None and x <= (high_f + 0.5):
                hits += 1
        else:
            if high_f is not None and x >= (high_f - 0.5):
                hits += 1
    return max(0.01, min(0.99, hits / total)) if total else None


def lead_bucket_from_hours(lead_hours: float) -> str:
    if lead_hours < 12:
        return "short"
    if lead_hours < 48:
        return "medium"
    return "long"


def regime_bucket(agreement: float, market_type: str = "above") -> str:
    mtype = (market_type or "above").lower()
    if agreement >= 5.0:
        prefix = "high_disagreement"
    elif agreement >= 3.0:
        prefix = "moderate_disagreement"
    else:
        prefix = "calm"
    return f"{prefix}:{mtype}"


def _segment_probability_adjustment(conn, market_type: str, lead_hours: float, agreement: float, city: str = "") -> float:
    """Return the average stored calibration adjustment for this trade segment."""
    if conn is None:
        return 0.0
    try:
        segments = [
            ("market_type", (market_type or "above").lower()),
            ("lead_bucket", lead_bucket_from_hours(lead_hours)),
            ("regime", regime_bucket(agreement, market_type)),
            ("city", city),
        ]
        adjustments: list[float] = []
        for kind, value in segments:
            row = conn.execute(
                """SELECT prob_adjustment FROM calibration_profiles
                   WHERE segment_kind = ? AND segment_value = ?
                   ORDER BY computed_at DESC, id DESC LIMIT 1""",
                (kind, value),
            ).fetchone()
            if row is not None and row[0] is not None:
                adjustments.append(float(row[0]))
        if not adjustments:
            return 0.0
        adj = sum(adjustments) / len(adjustments)
        return max(-0.15, min(0.15, adj))
    except Exception:
        return 0.0


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
