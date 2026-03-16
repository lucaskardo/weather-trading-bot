"""
Shared data types for the weather trading bot.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelForecast:
    """
    A single model's temperature forecast for one city/date.

    Lineage fields (run_id, publish_time, source_url) are required so we can
    prove no future data was used and replay any prediction exactly.
    """
    # Identity
    model_name: str           # e.g. "GFS", "ECMWF", "NOAA_NWS", "AROME", "ICON"
    city: str                 # e.g. "NYC"
    target_date: str          # ISO date "YYYY-MM-DD"

    # Forecast values
    predicted_high_f: float
    predicted_low_f: Optional[float] = None
    confidence: Optional[float] = None   # 0–1, model-reported or derived
    # Optional true ensemble members from upstream providers; when present the
    # canonical fair-value engine uses empirical rank probabilities rather than
    # smoothing around a point estimate.
    ensemble_members_f: Optional[list[float]] = None

    # Lineage — REQUIRED for audit; default empty str so callers must fill them
    run_id: str = ""          # Model run identifier, e.g. "2026031300" (YYYYMMDDCC)
    publish_time: str = ""    # ISO datetime when the model run was published
    source_url: str = ""      # API URL used to retrieve this forecast

    # Internal
    fetched_at: str = ""      # ISO datetime when we fetched it (set by weather client)
    market_id: Optional[str] = None  # foreign key to markets table, if known


@dataclass
class ConsensusForeccast:
    """
    Consensus temperature built from multiple ModelForecasts.
    Kept separate from ModelForecast so provenance is clear.
    """
    city: str
    target_date: str
    consensus_high_f: float
    consensus_low_f: Optional[float]
    agreement: float          # std-dev of model highs (lower = more agreement)
    n_models: int
    model_names: list[str] = field(default_factory=list)
    model_highs_f: list[float] = field(default_factory=list)
    model_confidences: list[float] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)


# Fix typo in class name while keeping backward-compatible alias
ConsensusForeccast.__name__ = "ConsensusForeccast"   # keep original for pickling
ConsensusForecast = ConsensusForeccast
