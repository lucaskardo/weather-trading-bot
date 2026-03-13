# weather-trading-bot

A production-grade weather prediction market trading bot for **Kalshi** and **Polymarket**.

This is a full architectural rebuild of the original `forecast-bot`, implementing every recommendation from the system audit: SQLite state, multi-strategy routing, real NWS settlement, risk guards, shadow competition, and walk-forward calibration.

---

## What changed from forecast-bot

| forecast-bot | weather-trading-bot |
|---|---|
| JSON files for state | SQLite WAL-mode (crash-safe, transactional) |
| Single strategy | 4 strategies with scorecard-based router |
| Coordinate-descent calibrator | Walk-forward CV + differential evolution optimizer |
| Band midpoint settlement | Real NWS observations (NOAA CDO + IEM ASOS fallback) |
| Simple Kelly sizing | Half-Kelly + capped softmax capital allocation |
| No risk guards | StaleDataHalt, ClusterCapExceeded, DailyLossHalt |
| No shadow testing | Shadow competition pipeline + promotion gate |
| No position analytics | MFE, MAE, hold time, cluster, edge lineage per trade |
| No reconciliation | Bidirectional local DB ↔ exchange reconciliation |

---

## Architecture

```
weather_trading_bot/
├── main.py                        # Entry point — single cycle or continuous autoloop
│
├── clients/
│   ├── weather.py                 # 5-model forecast fetcher (GFS, ECMWF, ICON, AROME, NOAA/NWS)
│   └── nws_settlement.py          # Official settlement via NOAA CDO + IEM ASOS fallback
│
├── core/
│   ├── forecaster.py              # Temperature scaling, Brier decomposition, prob_above_threshold
│   └── signals.py                 # Edge calculation, executable edge, opportunity finder
│
├── execution/
│   ├── orderbook.py               # VWAP execution price + slippage + fee model
│   └── lifecycle.py               # Position state machine (OPENED → HOLDING → terminal)
│
├── strategies/
│   ├── base.py                    # Signal dataclass + BaseStrategy abstract
│   ├── value_entry.py             # Live: edge-based entry with 4-rule exit logic
│   ├── convergence_exit.py        # Shadow: exit when price converges to fair value
│   ├── model_release.py           # Shadow: enter on large GFS/ECMWF forecast revision
│   ├── disagreement.py            # Shadow: enter on large inter-model spread
│   ├── shadow_logger.py           # Log shadow signals, settle hypothetical PnL
│   ├── promotion.py               # Scan shadow strategies for promotion candidates
│   └── analytics.py               # Per-trade MFE, MAE, hold time, cluster, edge lineage
│
├── strategy_router/
│   ├── scorecard.py               # Weighted composite score (Sharpe, Brier, exec, drawdown, instability)
│   ├── allocator.py               # Capped softmax capital allocator [5%–40%] per strategy
│   ├── selector.py                # 5-gate signal selector + half-Kelly sizing
│   └── brain.py                   # Orchestrator — runs full cycle, coordinates all modules
│
├── risk/
│   ├── guards.py                  # StaleDataHalt, ClusterCapExceeded, DailyLossHalt
│   └── reconciliation.py          # Bidirectional local DB ↔ exchange reconciliation
│
├── research/
│   ├── walk_forward.py            # Time-series CV, out-of-sample Brier per fold
│   ├── optimizer.py               # Differential evolution (scipy) + coordinate-descent fallback
│   └── calibrator.py              # Full pipeline: resolve → Brier → optimize → save
│
├── state/
│   └── db.py                      # SQLite WAL-mode, 8 tables, transaction context manager
│
├── shared/
│   ├── params.py                  # All tuneable parameters + geographic cluster definitions
│   └── types.py                   # ModelForecast, ConsensusForecast with full lineage fields
│
└── tests/                         # 352 tests, 2 skipped (scipy optional)
```

---

## How it works

### Trading cycle

1. **Fetch forecasts** — pull high-temperature predictions from up to 5 weather models per city. Each forecast carries a `run_id`, `publish_time`, and `source_url` for full lineage.
2. **Build consensus** — compute confidence-weighted mean and inter-model agreement.
3. **Generate signals** — for each open market, compute `executable_edge = model_prob - executable_price` (after fees + slippage + VWAP). Only signals above `min_executable_edge` with sufficient orderbook depth pass.
4. **Score strategies** — scorecard weights: Sharpe (35%), Brier calibration (30%), execution quality (20%), max drawdown (10%), parameter instability (5%).
5. **Allocate capital** — capped softmax: minimum 5%, maximum 40% per live strategy, with iterative redistribution when caps bind.
6. **Select trades** — 5 sequential gates: shadow filter → city limit → cluster cap → budget → Kelly too small.
7. **Manage positions** — lifecycle check order: stale data → reversal → convergence → pre-settlement.
8. **Record analytics** — on close: MFE, MAE, hold time, cluster, edge at entry/exit, forecast run IDs.

### Probability math

```
raw_prob  = 1 - CDF((threshold - consensus_f) / base_std_f)
scaled    = sigmoid(logit(raw_prob) / T)      # T learned by calibrator
```

Logit is clamped to `[-30, 30]` and output to `(1e-6, 1-1e-6)` to prevent overflow.

### Executable edge

```
executable_price = VWAP(orderbook, target_size) + slippage + taker_fee
executable_edge  = model_prob - executable_price
```

Only trades with `executable_edge >= min_executable_edge` and `depth >= min_depth_usd` pass.

### Kelly sizing

```
full_kelly = executable_edge / (1 - executable_price)   # YES
full_kelly = executable_edge / executable_price          # NO
trade_size = clamp(bankroll × half_kelly, min_kelly_frac, max_kelly_frac)
```

### Settlement

Official NWS settlement — no band midpoint inference:
- **Primary**: NOAA Climate Data Online (CDO) API with 17-city station map
- **Fallback**: IEM ASOS hourly data (daily max of `tmpf`, skips `M`/`T`/empty)
- Results cached in SQLite to avoid redundant API calls

---

## Shadow competition

Strategies run in shadow mode (`is_shadow=1`) before risking real capital. Shadow predictions are settled at contract expiry using a synthetic $100 notional.

**Promotion criteria** (all must be true):
- `trade_count >= 50` resolved shadow trades
- `scorecard > 75` (out of 100)
- Human review required — the bot flags candidates, never auto-promotes

---

## Risk guards

| Guard | Trigger | Action |
|---|---|---|
| `StaleDataHalt` | All forecasts older than `max_forecast_age_hours` | Halt trading |
| `ClusterCapExceeded` | Cluster exposure + proposed > `cluster_cap_usd` | Block trade |
| `DailyLossHalt` | Realized + unrealized loss ≥ `daily_loss_limit_usd` | Halt trading |
| `CityLimitExceeded` | Open positions in city ≥ `max_positions_per_city` | Block trade |

Reconciliation runs on every startup — bidirectional diff between local DB and exchange. Critical discrepancies are flagged before trading begins.

---

## Calibration

The calibration pipeline runs offline or on a schedule:

1. Load all resolved predictions from DB
2. Fetch outstanding NWS settlement data
3. Compute Brier score decomposition (reliability + resolution + uncertainty)
4. Run differential evolution over `{base_std_f, temp_T}` with walk-forward CV
5. Update `PARAMS` in-place and persist to DB + JSON

Walk-forward validation splits the trade log chronologically, evaluating out-of-sample per fold. Scipy differential evolution is used when available; coordinate-descent is the fallback.

---

## Setup

```bash
git clone git@github.com:lucaskardo/weather-trading-bot.git
cd weather-trading-bot

pip install -e ".[dev]"

cp .env.example .env
# Edit .env — set PAPER_MODE=true to start safely
```

---

## Usage

```bash
# Single dry-run cycle
python3 main.py

# Continuous loop (default: every 5 minutes)
python3 main.py --loop --interval 300

# Live trading, single cycle
python3 main.py --live

# Live trading, continuous loop
python3 main.py --loop --live

# Run tests
python3 -m pytest tests/ -v
```

---

## Configuration (`shared/params.py`)

| Parameter | Default | Description |
|---|---|---|
| `base_std_f` | `5.0` | Forecast std dev in °F (calibrated) |
| `temp_T` | `1.0` | Temperature scaling factor (calibrated) |
| `min_executable_edge` | `0.05` | Minimum edge after fees + slippage |
| `taker_fee_pct` | `0.01` | Exchange taker fee |
| `slippage_buffer_cents` | `1.0` | Slippage buffer in cents |
| `min_depth_usd` | `50.0` | Minimum orderbook depth required |
| `daily_loss_limit_usd` | `50.0` | Daily loss halt threshold |
| `cluster_cap_usd` | `200.0` | Max exposure per geographic cluster |
| `max_positions_per_city` | `3` | Max concurrent positions per city |
| `router_w_sharpe` | `0.35` | Scorecard Sharpe weight |
| `router_w_calibration` | `0.30` | Scorecard Brier calibration weight |
| `router_w_exec` | `0.20` | Scorecard execution quality weight |
| `router_w_dd` | `0.10` | Scorecard drawdown penalty weight |
| `router_w_instability` | `0.05` | Scorecard instability weight |

---

## Database schema

SQLite WAL-mode. 8 tables:

| Table | Purpose |
|---|---|
| `markets` | Open/settled weather markets |
| `forecasts` | Raw model forecasts with run lineage |
| `predictions` | All signals — live and shadow |
| `positions` | Open and closed positions with analytics |
| `portfolio` | Singleton bankroll + daily PnL |
| `daily_pnl` | Per-day realized PnL for loss-limit checks |
| `strategy_metrics` | Scorecard snapshots + promotion audit trail |
| `settlement_cache` | Cached NWS settlement results |

---

## Supported cities

17 cities across 6 geographic clusters:

| Cluster | Cities |
|---|---|
| Northeast | NYC, BOS, DC |
| Midwest | CHI, DAL |
| South | MIA, HOU, ATL |
| West | LA, SF, SEA |
| Europe | LON, PAR, MUN |
| Asia | SEO |
| South America | BUE, SAO |

Cluster definitions drive the `ClusterCapExceeded` risk guard — correlated weather exposure is capped per cluster.

---

## Live trading setup

### Kalshi
1. Create an account at kalshi.com
2. Generate an RSA key pair in account settings
3. Set `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in `.env`
4. Set `PAPER_MODE=false`

### Polymarket
Signal generation supported. Live execution not yet implemented.
