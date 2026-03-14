"""
Phase B4 — Autoresearch experiment loop.

Inspired by Karpathy's autoresearch concept: automatically generate parameter
variants, evaluate them with walk-forward backtesting, and promote winners.

Each experiment:
  - Perturbs one or more parameters from the current best
  - Runs walk-forward Brier scoring on historical trade data
  - Compares to the baseline (current production params)
  - Promotes if improvement exceeds PROMOTION_THRESHOLD

Experiments are stored in the `experiments` SQLite table for full audit trail.
The loop can be run on-demand (--autoresearch) or scheduled.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from research.walk_forward import walk_forward_brier
from shared.params import Params, PARAMS

# Minimum Brier improvement (relative) required to promote a candidate
PROMOTION_THRESHOLD = 0.05   # 5% improvement

# Parameter search space: {param_name: (min, max, step)}
_SEARCH_SPACE: dict[str, tuple[float, float, float]] = {
    "base_std_f":           (2.5, 12.0, 0.5),
    "temp_T":               (0.7, 1.8, 0.1),
    "min_executable_edge":  (0.02, 0.15, 0.01),
    "max_kelly_fraction":   (0.05, 0.40, 0.05),
    "stale_forecast_hours": (6.0, 48.0, 6.0),
}

# Parameter combos to test jointly (pairs that interact)
_COMBO_PROPOSALS: list[list[str]] = [
    ["base_std_f", "temp_T"],
    ["min_executable_edge", "max_kelly_fraction"],
]


def load_promoted_params(conn: sqlite3.Connection, params: Params) -> None:
    """
    Apply any previously promoted params from the DB onto a Params instance.

    Called at startup so winning experiment parameters survive restarts without
    ever mutating the module-level PARAMS singleton at experiment time.
    """
    rows = conn.execute("SELECT key, value FROM promoted_params").fetchall()
    for row in rows:
        key, value = row["key"], row["value"]
        if hasattr(params, key):
            setattr(params, key, value)


class ExperimentRegistry:
    """
    Manages parameter experiments with walk-forward validation.

    Usage:
        registry = ExperimentRegistry(conn, params)
        exp_id = registry.propose_experiment()
        result = registry.run_experiment(exp_id, trade_log)
        registry.promote_if_better(exp_id)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        params: Params = PARAMS,
        n_windows: int = 5,
    ):
        self.conn = conn
        self.params = params
        self.n_windows = n_windows

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def propose_experiment(self, description: Optional[str] = None) -> str:
        """
        Auto-generate a parameter variant to test.

        Perturbs one parameter at a time from the current best, cycling
        through the search space systematically. Skips combinations that
        have already been tested.

        Returns:
            experiment_id (UUID string)
        """
        candidate_params = self._propose_candidate()
        exp_id = str(uuid.uuid4())[:8]
        desc = description or self._describe(candidate_params)

        self.conn.execute(
            """INSERT INTO experiments
               (id, description, params_json, status, created_at)
               VALUES (?,?,?,'pending',?)""",
            (
                exp_id,
                desc,
                json.dumps(candidate_params),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()
        return exp_id

    def run_experiment(
        self,
        experiment_id: str,
        trade_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Execute walk-forward backtest with the candidate params.

        Updates the experiment row with results and sets status to 'completed'.

        Args:
            experiment_id: ID from propose_experiment().
            trade_log:     List of trade dicts with fair_value, outcome, threshold_f.

        Returns:
            Result dict with baseline_brier, candidate_brier, improvement_pct.
        """
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Experiment {experiment_id} not found")

        candidate_params_dict = json.loads(row["params_json"])

        # Mark running
        self.conn.execute(
            "UPDATE experiments SET status='running' WHERE id=?", (experiment_id,)
        )
        self.conn.commit()

        # Baseline score (current params)
        baseline_brier = walk_forward_brier(
            asdict(self.params), trade_log, n_windows=self.n_windows
        )

        # Candidate score
        merged = asdict(self.params)
        merged.update(candidate_params_dict)
        candidate_brier = walk_forward_brier(
            merged, trade_log, n_windows=self.n_windows
        )

        improvement_pct = (
            (baseline_brier - candidate_brier) / baseline_brier * 100
            if baseline_brier > 0 else 0.0
        )

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE experiments
               SET status='completed', baseline_brier=?, candidate_brier=?,
                   improvement_pct=?, trade_count=?, completed_at=?
               WHERE id=?""",
            (
                baseline_brier, candidate_brier, improvement_pct,
                len(trade_log), now, experiment_id,
            ),
        )
        self.conn.commit()

        return {
            "experiment_id": experiment_id,
            "baseline_brier": baseline_brier,
            "candidate_brier": candidate_brier,
            "improvement_pct": improvement_pct,
            "trade_count": len(trade_log),
            "params": candidate_params_dict,
        }

    def compare_to_baseline(self, experiment_id: str) -> dict[str, Any]:
        """Return improvement/regression vs current production params."""
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not row:
            return {"error": f"Experiment {experiment_id} not found"}

        return {
            "experiment_id": experiment_id,
            "description": row["description"],
            "status": row["status"],
            "baseline_brier": row["baseline_brier"],
            "candidate_brier": row["candidate_brier"],
            "improvement_pct": row["improvement_pct"],
            "better": (row["improvement_pct"] or 0) > 0,
            "promotable": (row["improvement_pct"] or 0) >= PROMOTION_THRESHOLD * 100,
        }

    def promote_if_better(self, experiment_id: str) -> bool:
        """
        If experiment beats baseline by >PROMOTION_THRESHOLD, persist params to DB.

        Params are stored in the promoted_params table and loaded at startup.
        The global PARAMS singleton is never mutated directly.

        Returns:
            True if promoted, False otherwise.
        """
        comparison = self.compare_to_baseline(experiment_id)
        if not comparison.get("promotable"):
            return False

        row = self.conn.execute(
            "SELECT params_json FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        candidate = json.loads(row["params_json"])

        # Persist to DB — never setattr on the singleton
        now = datetime.now(timezone.utc).isoformat()
        for key, value in candidate.items():
            if hasattr(self.params, key):
                self.conn.execute(
                    """INSERT OR REPLACE INTO promoted_params
                       (key, value, source_exp, promoted_at)
                       VALUES (?, ?, ?, ?)""",
                    (key, float(value), experiment_id, now),
                )

        self.conn.execute(
            "UPDATE experiments SET status='promoted' WHERE id=?", (experiment_id,)
        )
        self.conn.commit()

        import sys
        print(
            f"[autoresearch] PROMOTED experiment {experiment_id}: "
            f"Brier {comparison['baseline_brier']:.4f} → "
            f"{comparison['candidate_brier']:.4f} "
            f"({comparison['improvement_pct']:.1f}% improvement)",
            file=sys.stderr,
        )
        return True

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent experiments for dashboard display."""
        rows = self.conn.execute(
            """SELECT id, description, baseline_brier, candidate_brier,
                      improvement_pct, trade_count, status, created_at
               FROM experiments
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def run_cycle(self, trade_log: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Propose + run one experiment cycle.

        Returns result dict. Promotes automatically if threshold met.
        """
        exp_id = self.propose_experiment()
        result = self.run_experiment(exp_id, trade_log)
        promoted = self.promote_if_better(exp_id)
        result["promoted"] = promoted
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _propose_candidate(self) -> dict[str, float]:
        """
        Generate a candidate parameter dict by perturbing one or two parameters.

        Every 5th experiment proposes a 2-parameter combo from _COMBO_PROPOSALS.
        Otherwise picks the least-tested single parameter and perturbs it.
        """
        # Count total completed experiments to decide single vs combo
        total_done = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status != 'pending'"
        ).fetchone()[0]

        if total_done % 5 == 4 and _COMBO_PROPOSALS:
            # Propose a combo
            combo_idx = (total_done // 5) % len(_COMBO_PROPOSALS)
            combo_keys = _COMBO_PROPOSALS[combo_idx]
            result: dict[str, float] = {}
            for param_name in combo_keys:
                if param_name not in _SEARCH_SPACE:
                    continue
                current_val = getattr(self.params, param_name, None)
                if current_val is None:
                    continue
                low, high, step = _SEARCH_SPACE[param_name]
                candidate = round(current_val + step, 6)
                if candidate > high:
                    candidate = round(current_val - step, 6)
                if low <= candidate <= high:
                    result[param_name] = candidate
            if result:
                return result

        # Single-parameter perturbation
        tested_counts: dict[str, int] = {}
        rows = self.conn.execute(
            "SELECT params_json FROM experiments WHERE status != 'pending'"
        ).fetchall()
        for row in rows:
            try:
                p = json.loads(row["params_json"])
                for k in p:
                    tested_counts[k] = tested_counts.get(k, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        # Pick least-tested parameter
        param_name = min(
            _SEARCH_SPACE.keys(),
            key=lambda k: tested_counts.get(k, 0),
        )

        current_val = getattr(self.params, param_name)
        low, high, step = _SEARCH_SPACE[param_name]

        # Try +step, then -step, then midpoint
        for delta in [step, -step]:
            candidate = round(current_val + delta, 6)
            if low <= candidate <= high:
                return {param_name: candidate}

        return {param_name: round((low + high) / 2, 6)}

    def _describe(self, params: dict) -> str:
        parts = [f"{k}={v:.3f}" for k, v in params.items()]
        return "candidate: " + ", ".join(parts)
