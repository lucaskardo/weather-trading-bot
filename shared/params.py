"""
Central parameter store for the weather trading bot.

All tuneable constants live here. The calibrator may update numeric fields
at runtime; do not import individual values at module level — always read
through the Params instance so updates are reflected.
"""

from dataclasses import dataclass, field


@dataclass
class Params:
    # ------------------------------------------------------------------ #
    # Router weights (must sum to 1.0)
    # ------------------------------------------------------------------ #
    router_w_sharpe: float = 0.35
    router_w_calibration: float = 0.30
    router_w_exec: float = 0.20
    router_w_dd: float = 0.10
    router_w_instability: float = 0.05
    router_temperature: float = 10.0  # softmax temperature

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    taker_fee_pct: float = 0.01          # 1% Kalshi taker fee
    slippage_buffer_cents: float = 1.0   # additional 1¢ slippage buffer
    min_depth_usd: float = 50.0          # minimum orderbook depth to trade
    min_open_interest: int = 20          # minimum open interest (contracts)

    # ------------------------------------------------------------------ #
    # Risk
    # ------------------------------------------------------------------ #
    max_cluster_exposure_pct: float = 0.15   # max 15% bankroll in one cluster
    max_city_exposure_pct: float = 0.15      # max 15% bankroll of gross notional in one city
    max_portfolio_var95_pct: float = 0.10    # halt/reject if proxy daily VaR95 exceeds 10% bankroll
    same_city_corr: float = 0.85
    same_cluster_corr: float = 0.50
    cross_cluster_corr: float = 0.20
    max_positions_per_city: int = 3
    stale_forecast_hours: float = 6.0        # halt if forecast older than this
    max_round_trips_per_day: int = 3

    # ------------------------------------------------------------------ #
    # Forecasting / calibration
    # ------------------------------------------------------------------ #
    base_std_f: float = 5.0              # baseline forecast uncertainty (°F)
    temp_scaling_T: float = 1.0          # Platt scaling temperature (1 = no change)
    use_differential_evolution: bool = True
    walk_forward_windows: int = 5
    temp_scaling_range: tuple[float, float] = (0.7, 1.8)

    # ------------------------------------------------------------------ #
    # Signal filters
    # ------------------------------------------------------------------ #
    min_executable_edge: float = 0.05    # minimum edge after fees/slippage
    min_confidence: float = 0.0          # minimum model confidence to signal
    min_kelly_fraction: float = 0.01     # skip bets below 1% Kelly
    max_kelly_fraction: float = 0.25     # cap at 25% Kelly
    use_monte_carlo: bool = True         # use MC probability engine vs Gaussian CDF
    use_empirical_ensemble: bool = True  # use true ensemble members when available
    monte_carlo_samples: int = 2000      # MC samples per probability estimate
    # Time-decaying edge threshold: min_edge = base_edge + alpha * exp(beta * hours_to_settlement)
    # At 48h: base_edge + alpha * exp(48 * beta); at 2h: base_edge + alpha * exp(2 * beta)
    edge_decay_alpha: float = 0.002      # additional edge required at max lead (72h+)
    edge_decay_beta: float = 0.05        # decay exponent (controls shape of falloff)

    # ------------------------------------------------------------------ #
    # Cluster definitions
    # ------------------------------------------------------------------ #
    clusters: dict[str, list[str]] = field(default_factory=lambda: {
        "northeast":     ["NYC", "BOS", "DC"],
        "midwest":       ["CHI", "DAL"],
        "south":         ["MIA", "HOU", "ATL"],
        "west":          ["LA", "SF", "SEA"],
        "europe":        ["LON", "PAR", "MUN"],
        "asia":          ["SEO"],
        "south_america": ["BUE", "SAO"],
    })


# Module-level singleton — import this everywhere
PARAMS = Params()


def get_cluster(city: str, params: Params | None = None) -> str | None:
    """Return the cluster name for *city*, or None if not in any cluster."""
    p = params or PARAMS
    for cluster_name, cities in p.clusters.items():
        if city in cities:
            return cluster_name
    return None
