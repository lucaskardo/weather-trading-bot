"""
Phase B1 — Per-model, per-city bias correction.

Learns systematic forecast errors from historical (predicted, actual) pairs
stored in the DB. Applies corrections during consensus building so that
downstream probability estimates use de-biased temperatures.

Bias = mean(predicted - actual).  Positive bias → model runs hot.
Corrected forecast = predicted - bias.

Requires at least MIN_OBS observations per (city, model) pair before
applying a correction; falls back to zero correction otherwise.
"""

from __future__ import annotations

import sqlite3
from typing import Any

MIN_OBS = 10  # minimum observations before trusting a bias estimate


def learn_biases(
    conn: sqlite3.Connection,
    min_obs: int = MIN_OBS,
) -> dict[str, dict[str, float]]:
    """
    Compute per-city, per-model bias from resolved forecasts in the DB.

    Joins forecasts with settlement_cache on (city, target_date) to get
    actual_high_f, then computes mean(predicted_high_f - actual_high_f)
    per (city, model_name) group.

    Args:
        conn:    SQLite connection.
        min_obs: Minimum number of matched pairs required to trust the bias.

    Returns:
        Nested dict: {city: {model_name: bias_f}}
        bias_f > 0  → model runs hot (overcorrects upward)
        bias_f < 0  → model runs cold
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

    # Accumulate residuals per (city, model)
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


def apply_bias(
    predicted_high_f: float,
    city: str,
    model_name: str,
    bias_table: dict[str, dict[str, float]],
) -> float:
    """
    Return bias-corrected forecast temperature.

    If no bias estimate exists for this (city, model) pair, returns the
    raw prediction unchanged.
    """
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
