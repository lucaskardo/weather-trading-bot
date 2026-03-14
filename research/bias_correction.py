"""
Phase B1 — Per-model, per-city bias correction with seasonal + lead-time keys.

Learns systematic forecast errors from historical (predicted, actual) pairs
stored in the DB. Applies corrections during consensus building so that
downstream probability estimates use de-biased temperatures.

Bias = mean(predicted - actual).  Positive bias → model runs hot.
Corrected forecast = predicted - bias.

Key hierarchy (most-specific to least, with fallback):
  (city, model, season, lead_bucket) → highest precision, needs ≥MIN_OBS_FINE
  (city, model)                      → coarse fallback, needs ≥MIN_OBS

Seasons: "DJF" (Dec-Feb), "MAM" (Mar-May), "JJA" (Jun-Aug), "SON" (Sep-Nov)
Lead buckets: "0-24h", "24-72h", "72h+"
"""

from __future__ import annotations

import sqlite3
from typing import Any

MIN_OBS = 10       # minimum observations for coarse (city, model) bias
MIN_OBS_FINE = 5   # minimum observations for fine (city, model, season, lead) bias

_SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _season(target_date: str) -> str:
    """Return season code for an ISO date string."""
    try:
        month = int(target_date[5:7])
        return _SEASONS.get(month, "MAM")
    except (ValueError, IndexError):
        return "MAM"


def _lead_bucket(lead_hours: float) -> str:
    """Bucket a lead time in hours."""
    if lead_hours <= 24:
        return "0-24h"
    elif lead_hours <= 72:
        return "24-72h"
    else:
        return "72h+"


def learn_biases(
    conn: sqlite3.Connection,
    min_obs: int = MIN_OBS,
) -> dict[str, dict[str, float]]:
    """
    Compute per-city, per-model bias from resolved forecasts in the DB.

    Returns coarse bias table: {city: {model_name: bias_f}}
    bias_f > 0 → model runs hot; bias_f < 0 → model runs cold
    """
    rows = conn.execute(
        """
        SELECT f.city, f.model_name,
               f.predicted_high_f - s.actual_high_f AS residual
        FROM forecasts f
        JOIN settlement_cache s
          ON s.city = f.city AND s.target_date = f.target_date
        WHERE f.predicted_high_f IS NOT NULL
          AND s.actual_high_f IS NOT NULL
        """
    ).fetchall()

    accum: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["city"], row["model_name"])
        accum.setdefault(key, []).append(row["residual"])

    biases: dict[str, dict[str, float]] = {}
    for (city, model), residuals in accum.items():
        if len(residuals) < min_obs:
            continue
        bias = sum(residuals) / len(residuals)
        biases.setdefault(city, {})[model] = bias

    return biases


def learn_fine_biases(
    conn: sqlite3.Connection,
    min_obs: int = MIN_OBS_FINE,
) -> dict[tuple[str, str, str, str], float]:
    """
    Compute fine-grained bias keyed on (city, model, season, lead_bucket).

    Falls back gracefully: callers should try fine key first, then coarse.

    Returns:
        Dict mapping (city, model, season, lead_bucket) → bias_f
    """
    rows = conn.execute(
        """
        SELECT f.city, f.model_name, f.target_date, f.publish_time,
               f.predicted_high_f - s.actual_high_f AS residual
        FROM forecasts f
        JOIN settlement_cache s
          ON s.city = f.city AND s.target_date = f.target_date
        WHERE f.predicted_high_f IS NOT NULL
          AND s.actual_high_f IS NOT NULL
        """
    ).fetchall()

    accum: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        seas = _season(row["target_date"] or "")
        # Estimate lead hours from publish_time vs target_date (rough)
        try:
            from datetime import datetime, timezone
            pub = datetime.fromisoformat(row["publish_time"]) if row["publish_time"] else None
            if pub:
                tgt = datetime.fromisoformat(f"{row['target_date']}T12:00:00+00:00")
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                lead_h = max(0.0, (tgt - pub).total_seconds() / 3600)
            else:
                lead_h = 48.0
        except Exception:
            lead_h = 48.0
        lead = _lead_bucket(lead_h)
        key = (row["city"], row["model_name"], seas, lead)
        accum.setdefault(key, []).append(row["residual"])

    fine: dict[tuple[str, str, str, str], float] = {}
    for key, residuals in accum.items():
        if len(residuals) >= min_obs:
            fine[key] = sum(residuals) / len(residuals)

    return fine


def apply_bias(
    predicted_high_f: float,
    city: str,
    model_name: str,
    bias_table: dict[str, dict[str, float]],
    fine_bias_table: dict[tuple[str, str, str, str], float] | None = None,
    target_date: str = "",
    lead_hours: float = 48.0,
) -> float:
    """
    Return bias-corrected forecast temperature.

    Tries fine-grained (city, model, season, lead_bucket) key first when
    fine_bias_table is provided, then falls back to coarse (city, model).
    Returns raw prediction unchanged if no bias estimate exists.
    """
    if fine_bias_table is not None and target_date:
        seas = _season(target_date)
        lead = _lead_bucket(lead_hours)
        fine_key = (city, model_name, seas, lead)
        if fine_key in fine_bias_table:
            return predicted_high_f - fine_bias_table[fine_key]

    # Coarse fallback
    bias = bias_table.get(city, {}).get(model_name, 0.0)
    return predicted_high_f - bias


def get_bias_summary(
    conn: sqlite3.Connection,
    min_obs: int = MIN_OBS,
) -> list[dict[str, Any]]:
    """
    Return a human-readable list of per-(city, model) bias records.

    Useful for the dashboard's Forecast Quality tab.
    """
    rows = conn.execute(
        """
        SELECT f.city, f.model_name,
               COUNT(*) AS n_obs,
               AVG(f.predicted_high_f - s.actual_high_f) AS bias_f,
               AVG(ABS(f.predicted_high_f - s.actual_high_f)) AS mae_f
        FROM forecasts f
        JOIN settlement_cache s
          ON s.city = f.city AND s.target_date = f.target_date
        WHERE f.predicted_high_f IS NOT NULL
          AND s.actual_high_f IS NOT NULL
        GROUP BY f.city, f.model_name
        HAVING COUNT(*) >= ?
        ORDER BY f.city, f.model_name
        """,
        (min_obs,),
    ).fetchall()

    return [
        {
            "city": r["city"],
            "model_name": r["model_name"],
            "n_obs": r["n_obs"],
            "bias_f": round(r["bias_f"], 3),
            "mae_f": round(r["mae_f"], 3),
        }
        for r in rows
    ]
