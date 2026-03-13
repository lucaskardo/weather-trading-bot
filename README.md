# weather-trading-bot

A production-grade, self-improving weather prediction market trading bot for **Kalshi** and **Polymarket**.

Fetches live weather markets, runs multi-model temperature forecasts through a calibrated probability engine, routes capital across competing strategies using a scorecard system, and continuously improves itself through walk-forward backtesting and an autoresearch loop. Ships with a real-time React dashboard and full paper-trading mode so you can watch it work before risking capital.

---

## Quick Start

```bash
git clone git@github.com:lucaskardo/weather-trading-bot.git
cd weather-trading-bot

pip install -r requirements.txt

cp .env.example .env
# Add your NOAA_CDO_TOKEN (free) at minimum

# Paper trade with real market data
python main.py --paper

# Continuous paper trading loop (every 5 minutes)
python main.py --paper --loop

# Open the dashboard in a second terminal
python main.py --dashboard
# Then open dashboard/index.html in your browser
```

---

## What It Does

Every cycle the bot:

1. **Fetches live markets** from Kalshi (8 US city series) and Polymarket (weather tag) — **persists them to DB immediately** so market_id is never NULL
2. **Pulls 5-model weather forecasts** — GFS, ECMWF, ICON, AROME, NOAA/NWS — with full run lineage
3. **Applies bias correction** — per-city, per-model offsets learned from historical residuals, integrated directly into signal generation
4. **Computes calibrated probabilities** — Gaussian CDF with **dynamic forecast uncertainty** (scales with model spread, lead time, and season) + Platt temperature scaling, optimized weekly via walk-forward Brier scoring
5. **Calculates executable edge** — VWAP through the real orderbook, Kalshi's price-dependent fee schedule, slippage buffer
6. **Routes capital** — softmax allocation across strategies, capped [5%–40%] per strategy, half-Kelly position sizing
7. **Executes via abstracted executor** — `PaperExecutor` (default) or `KalshiExecutor` (live), swappable without touching strategy code
8. **Enforces risk guards** — stale data halt, cluster cap, daily loss limit, city position limit
9. **Logs everything** to SQLite — predictions, positions with **full entry snapshots** (high_f, consensus_f, edge, market_id), forecasts, settlement data, experiments
10. **Self-improves** — autoresearch loop proposes single or combo parameter variants across 5-param search space, validates out-of-sample, promotes winners

---

## Architecture

```
weather-trading-bot/
│
├── main.py                          # Entry point — all CLI modes
│
├── clients/
│   ├── kalshi_client.py             # Kalshi REST API — markets, orderbooks, ticker parsing
│   ├── polymarket_client.py         # Polymarket Gamma API — weather market discovery
│   ├── weather.py                   # 5-model Open-Meteo fetcher with run_id lineage
│   ├── hrrr.py                      # HRRR hourly nowcast for same-day contracts
│   └── nws_settlement.py            # Official NWS settlement (NOAA CDO + IEM ASOS fallback)
│
├── core/
│   ├── forecaster.py                # Temperature scaling, Brier decomposition, prob_above_threshold, dynamic_std_f
│   └── signals.py                   # Edge calculation, executable edge, opportunity ranking
│
├── execution/
│   ├── exchange_executor.py         # ExchangeExecutor ABC, PaperExecutor, KalshiExecutor
│   ├── orderbook.py                 # VWAP pricing, Kalshi fee schedule, slippage model
│   └── lifecycle.py                 # Position state machine (OPENED → HOLDING → terminal)
│
├── strategies/
│   ├── base.py                      # Signal dataclass + BaseStrategy ABC
│   ├── value_entry.py               # LIVE: calibrated-edge entry, 4-rule exit logic
│   ├── convergence_exit.py          # SHADOW: exit when price converges to fair value
│   ├── model_release.py             # SHADOW: enter on large GFS/ECMWF revision (≥3°F)
│   ├── disagreement.py              # SHADOW: enter on large inter-model spread (≥4°F)
│   ├── shadow_logger.py             # Persist shadow signals, settle hypothetical PnL
│   ├── promotion.py                 # Scan shadows for promotion (score > 75, ≥50 trades)
│   └── analytics.py                 # Per-trade MFE, MAE, hold time, cluster, edge lineage
│
├── strategy_router/
│   ├── scorecard.py                 # Weighted composite: Sharpe(35%) Brier(30%) Exec(20%) DD(10%) Instability(5%)
│   ├── allocator.py                 # Capped softmax allocator [5%–40%], iterative redistribution
│   ├── selector.py                  # 5-gate filter + half-Kelly sizing
│   └── brain.py                     # Cycle orchestrator — wires all modules together
│
├── risk/
│   ├── guards.py                    # StaleDataHalt, ClusterCapExceeded, DailyLossHalt
│   └── reconciliation.py            # Bidirectional local DB ↔ exchange reconciliation
│
├── research/
│   ├── walk_forward.py              # Time-series CV, out-of-sample Brier per fold
│   ├── optimizer.py                 # Differential evolution (scipy) + coordinate-descent fallback
│   ├── calibrator.py                # Full pipeline: resolve → Brier → optimize → save
│   ├── bias_correction.py           # Per-(city, model) bias learning from historical residuals
│   └── autoresearch.py              # ExperimentRegistry: propose → run → compare → promote
│
├── state/
│   └── db.py                        # SQLite WAL-mode, 9 tables, transaction context manager
│
├── shared/
│   ├── params.py                    # All tuneable parameters + geographic cluster definitions
│   └── types.py                     # ModelForecast, ConsensusForecast with lineage fields
│
├── dashboard/
│   ├── api.py                       # Flask micro-server — 17 REST endpoints
│   └── index.html                   # React single-file dashboard — 4 tabs, QuickStats bar, no build step
│
└── tests/                           # 440 tests, 2 skipped (scipy optional)
    ├── test_db.py
    ├── test_nws_settlement.py
    ├── test_executable_edge.py
    ├── test_forecast_lineage.py
    ├── test_strategies.py
    ├── test_router.py
    ├── test_risk_guards.py
    ├── test_research.py
    ├── test_shadow.py
    ├── test_kalshi_client.py
    ├── test_polymarket_client.py
    └── test_phase_b.py
```

---

## Core Concepts

### Executable Edge

The bot never trades on raw model-vs-market spread. Every signal must survive the full cost stack:

```
executable_price = VWAP(orderbook, target_size)
                 + slippage_buffer             # configurable, default 1¢
                 + kalshi_fee(price)           # price-dependent: peaks ~3.5% at 50¢

executable_edge  = model_prob - executable_price
```

Only signals with `executable_edge ≥ min_executable_edge` AND `orderbook_depth ≥ min_depth_usd` pass.

### Kalshi Fee Schedule (Feb 2026)

Unlike a flat fee, Kalshi charges based on `min(price, 1-price)`:

| Contract price | Effective cost bracket | Fee rate |
|---|---|---|
| 1–9¢ / 91–99¢ | 1–9¢ | 3.5% |
| 10–24¢ / 76–90¢ | 10–24¢ | 5.0% |
| 25–50¢ | 25–50¢ | 7.0% (peak) |

Near-50¢ contracts that look profitable on raw edge often aren't after fees. This is modeled exactly.

### Probability Model

```
raw_prob = 1 - CDF((threshold_f - consensus_f) / dynamic_std_f)
scaled   = sigmoid(logit(raw_prob) / T)
```

- `consensus_f` = confidence-weighted average of de-biased model forecasts
- `dynamic_std_f` replaces the static `base_std_f` — adjusts uncertainty upward for high model spread, long lead times, and summer contracts (see below)
- `T` (temperature) calibrated alongside `base_std_f`
- Logit clamped to `[-30, 30]`, output to `(1e-6, 1-1e-6)`

### Dynamic Forecast Uncertainty

Static uncertainty (`base_std_f = 5.0°F` for all cities and lead times) underestimates risk in high-disagreement or long-range scenarios. `dynamic_std_f()` scales it up based on three factors:

```
std = base_std_f
    + min(3.0, (lead_days - 1) × 0.5)   # +0.5°F per day beyond 24h, capped at +3°F
    + 0.3 × model_spread                 # 30% of inter-model std-dev
    × 1.15 if summer (Jun–Aug)           # seasonal variance boost
```

This makes the bot systematically more conservative on week-ahead contracts and high-disagreement markets.

### Bias Correction

Every model has systematic errors that vary by city and season. The bot learns these automatically:

```
bias(city, model) = mean(predicted_high_f - actual_high_f)   # over ≥10 resolved pairs
corrected_high_f  = predicted_high_f - bias
```

After ~50 resolved trades per city/model, the probability engine is materially better calibrated.

### Strategy Router

Strategies compete for capital through a scorecard:

| Component | Weight | Measures |
|---|---|---|
| Sharpe ratio | 35% | Risk-adjusted return consistency |
| Brier score | 30% | Forecast calibration quality |
| Execution quality | 20% | Edge realized vs. edge expected |
| Max drawdown | 10% | Worst peak-to-trough loss |
| Parameter instability | 5% | How much params change between calibrations |

Capital is allocated via capped softmax (min 5%, max 40% per strategy). When all strategies simultaneously hit the upper cap, the bot correctly caps total deployment rather than over-normalizing.

### Shadow Competition

All three non-live strategies run in shadow mode — they generate signals, log predictions, and receive hypothetical PnL at settlement, but never touch real capital. Promotion requires:

- `trade_count ≥ 50` resolved shadow trades
- `scorecard > 75` (out of 100)
- **Human review** — the bot flags candidates and writes to `strategy_metrics`, never auto-promotes

### HRRR Intraday Nowcast

For same-day temperature contracts, HRRR updates hourly at 3km resolution. The bot combines:

```
hrrr_high_f = max(observed_max_so_far, hrrr_forecast_remaining_hours)
```

This gives the freshest possible estimate of the final daily high, often catching market mispricing that emerges when actual temperatures diverge from overnight forecasts.

### Autoresearch Loop

Inspired by Karpathy's autoresearch concept — automated parameter improvement without human intervention:

1. **Propose** — perturb one or two parameters from the current best (see search space below)
2. **Validate** — walk-forward Brier scoring on historical resolved trades
3. **Compare** — measure improvement vs. current production params
4. **Promote** — automatically apply if improvement > 5%; record in `experiments` table for audit

**Search space (5 parameters + 2-param combos every 5th experiment):**

| Parameter | Range | Step |
|---|---|---|
| `base_std_f` | 2.5 – 12.0 | 0.5 |
| `temp_T` | 0.7 – 1.8 | 0.1 |
| `min_executable_edge` | 0.02 – 0.15 | 0.01 |
| `max_kelly_fraction` | 0.05 – 0.40 | 0.05 |
| `stale_forecast_hours` | 6 – 48 | 6 |

Combo proposals (every 5th run): `[base_std_f, temp_T]` and `[min_executable_edge, max_kelly_fraction]`.

---

## CLI Reference

```bash
# Paper trading (real market data, no real orders)
python main.py --paper                    # single cycle
python main.py --paper --loop             # continuous (default: every 5 min)
python main.py --paper --loop --interval 120   # every 2 minutes

# Live trading (requires Kalshi API keys in .env)
python main.py --live                     # single cycle
python main.py --live --loop              # continuous

# Research & calibration
python main.py --calibrate                # run calibration pipeline
python main.py --autoresearch             # propose + run one experiment

# Dashboard
python main.py --dashboard                # start API server (port 5001)
python main.py --dashboard --port 8080    # custom port
# Then open dashboard/index.html in browser
```

---

## Dashboard

Start the bot in one terminal, the dashboard API in another, and open the HTML file in a browser — no build step required. A **QuickStats bar** across the top of every tab shows live bankroll, P&L, open positions, win rate, and last-cycle summary. A **Last Cycle panel** at the top of Tab 1 shows signals generated, executed, and exits from the most recent run.

```bash
# Terminal 1
python main.py --paper --loop

# Terminal 2
python main.py --dashboard

# Browser
open dashboard/index.html
```

**Tab 1 — Live Trading**
- Current bankroll and total P&L
- Open positions with entry price, size, edge
- Per-strategy capital allocation with bar chart
- Cluster exposure heatmap (geographic risk)
- Recent trades table

**Tab 2 — Strategy Performance**
- Per-strategy KPIs: Sharpe, Brier, win rate, trade count, avg hold time
- Live vs. shadow comparison table
- Brier score trend over time
- MFE/MAE scatter per strategy

**Tab 3 — Autoresearch**
- Full experiment registry with baseline vs. candidate Brier comparison
- Improvement % and promotion status per experiment
- Promotion candidates highlighted (score > 75, shadow mode)
- **▶ Run Experiment** button — triggers one autoresearch cycle from the browser

**Tab 4 — Forecast Quality**
- Per-city, per-model bias chart (°F systematic error)
- Model agreement heatmap (inter-model spread)
- Settlement vs. forecast residuals scatter

---

## Database Schema

SQLite WAL-mode — crash-safe, no server required. 9 tables:

| Table | Purpose |
|---|---|
| `markets` | Open and settled weather markets from all exchanges |
| `forecasts` | Raw model forecasts with full run lineage (run_id, publish_time, source_url) |
| `predictions` | All signals — live (`is_shadow=0`) and shadow (`is_shadow=1`) |
| `positions` | Open and closed positions with analytics (MFE, MAE, hold_time, edge lineage) |
| `portfolio` | Singleton bankroll and cash tracking |
| `daily_pnl` | Per-day realized P&L for loss-limit enforcement |
| `strategy_metrics` | Scorecard snapshots and promotion audit trail |
| `settlement_cache` | Cached NWS official settlement data |
| `experiments` | Autoresearch experiment registry with full results |

---

## Configuration

Copy `.env.example` to `.env`:

```env
PAPER_MODE=true
BANKROLL=1000.0
DAILY_LOSS_LIMIT=50.0
POLL_INTERVAL_S=300
NOAA_CDO_TOKEN=          # free: https://www.ncdc.noaa.gov/cdo-web/token
KALSHI_KEY_ID=           # live trading only
KALSHI_PRIVATE_KEY_PATH= # live trading only
DB_PATH=data/bot.db
```

Key parameters in `shared/params.py`:

| Parameter | Default | Calibrated? | Description |
|---|---|---|---|
| `base_std_f` | `5.0` | ✅ Weekly | Baseline forecast std dev in °F (dynamic_std_f scales this up) |
| `temp_T` | `1.0` | ✅ Weekly | Platt temperature scaling factor |
| `min_executable_edge` | `0.05` | ✅ Autoresearch | Min edge after full cost stack |
| `max_kelly_fraction` | `0.25` | ✅ Autoresearch | Kelly fraction cap per position |
| `stale_forecast_hours` | `12.0` | ✅ Autoresearch | Hours before a forecast is stale |
| `slippage_buffer_cents` | `1.0` | — | Slippage padding in cents |
| `min_depth_usd` | `50.0` | — | Min orderbook depth to trade |
| `daily_loss_limit_usd` | `50.0` | — | Daily loss halt threshold |
| `cluster_cap_usd` | `200.0` | — | Max exposure per geographic cluster |
| `max_positions_per_city` | `3` | — | Max concurrent positions per city |

---

## Supported Markets

### Kalshi (8 US cities)
`KXHIGHNY` · `KXHIGHCHI` · `KXHIGHLA` · `KXHIGHDC` · `KXHIGHSF` · `KXHIGHHOU` · `KXHIGHMIA` · `KXHIGHBOS`

### Polymarket
All active weather markets discovered via the `weather` tag on the Gamma API.

### Weather Models

| Model | Provider | Resolution | Update Frequency |
|---|---|---|---|
| GFS | NOAA | ~13km | 4× daily |
| ECMWF IFS | ECMWF | ~9km | 2× daily |
| ICON | DWD | ~13km | 4× daily |
| AROME | Météo-France | ~1.3km | 4× daily (Europe) |
| NOAA/NWS | NOAA | ~3km | Hourly (HRRR) |

### Geographic Clusters

| Cluster | Cities | Purpose |
|---|---|---|
| Northeast | NYC, BOS, DC | Correlated East Coast weather |
| Midwest | CHI, DAL | Central US systems |
| South | MIA, HOU, ATL | Gulf + subtropical |
| West | LA, SF, SEA | Pacific Coast |
| Europe | LON, PAR, MUN | Atlantic weather systems |
| Asia | SEO | East Asia |
| South America | BUE, SAO | Southern Hemisphere |

Cluster definitions drive the `ClusterCapExceeded` risk guard — correlated weather exposure is capped at the cluster level, not just per city.

---

## Potential Integrations & Collaborations

### Data Sources

| Tool / API | Integration Opportunity |
|---|---|
| **Weatherbit API** | Alternative high-resolution US forecasts; useful as a 6th model vote to reduce model risk |
| **Tomorrow.io** | Proprietary ML-based forecasts; their API provides probability distributions directly, which could replace parts of the Gaussian approximation |
| **IBM Environmental Intelligence** | Enterprise weather APIs with ensemble spread data — useful for `DisagreementStrategy` |
| **Copernicus ERA5** | Historical reanalysis dataset for deeper bias correction training (decades of actuals vs. model hindcasts) |
| **NCEP GEFS** | 30-member ensemble from NOAA — provides probabilistic forecasts natively rather than point estimates |
| **Meteomatics** | Sub-hourly gridded forecasts useful for intraday HRRR replacement/supplement |

### Prediction Market Platforms

| Platform | Integration Opportunity |
|---|---|
| **Manifold Markets** | Open platform with API — good for shadow-testing strategies against play-money markets before deploying to Kalshi |
| **Augur v2** | Decentralized prediction market; cross-venue arbitrage if same weather contract exists |
| **Polymarket CLOB** | Live order placement (signals already work; execution client is next milestone) |
| **Metaculus** | Research-grade forecasting community; their aggregated community forecasts on temperature markets could be used as a consensus signal |

### Research & Analytics

| Tool | Integration Opportunity |
|---|---|
| **Weights & Biases (wandb)** | Experiment tracking for the autoresearch loop — log Brier scores, parameter variants, and walk-forward fold results as W&B runs instead of SQLite only |
| **MLflow** | Open-source alternative to wandb; model registry for tracking calibrated parameter versions |
| **Grafana + SQLite plugin** | Production-grade dashboarding on top of the existing `bot.db` — replaces the hand-rolled React dashboard for teams |
| **Prometheus + Alertmanager** | Expose `/metrics` endpoint from the Flask API; alert on daily loss limit approach, stale forecast warnings |
| **Prefect / Airflow** | Replace the manual `--loop` scheduler with a proper DAG — separate fetching, signal generation, execution, and calibration into distinct tasks with retries |

### Notification & Monitoring

| Tool | Integration Opportunity |
|---|---|
| **Slack / Discord webhooks** | Post trade summaries, promotion candidates, and risk guard firings to a channel |
| **Telegram Bot API** | Mobile alerts for large edge signals or daily P&L summaries |
| **PagerDuty** | On-call alerting for `StaleDataHalt` or reconciliation failures in live mode |
| **Sentry** | Error tracking for production deployments — catches API timeouts, DB corruption, unexpected exceptions |

### Execution & Infrastructure

| Tool | Integration Opportunity |
|---|---|
| **Kalshi FIX API** | Lower-latency order placement than REST for time-sensitive intraday signals |
| **Redis** | Replace SQLite settlement cache with Redis for multi-process deployments; pub/sub for real-time signal distribution |
| **Docker + docker-compose** | Containerize bot + dashboard; `docker-compose up` for one-command deployment |
| **Fly.io / Railway** | Zero-config cloud deployment — runs the bot loop as a persistent process without managing a server |
| **GitHub Actions** | Nightly calibration runs as a CI job; PR checks that run the full test suite |

### Academic / Research Collaboration

| Resource | Opportunity |
|---|---|
| **ClimateCorp / Jupiter Intelligence** | Climate-adjusted temperature baselines that account for long-term warming trends in city-level high temperature distributions |
| **WeatherBench2** | Open benchmark for NWP model evaluation — compare our ensemble against state-of-the-art baselines to validate the multi-model approach |
| **ProbabilisticForecastEvaluation (R package)** | Formal verification tools for our Brier decomposition and reliability diagrams |

---

## Live Trading Setup

### Kalshi
1. Create account at [kalshi.com](https://kalshi.com)
2. Go to **Settings → API Keys** → generate RSA key pair
3. Download the private key `.pem` file
4. Set in `.env`:
   ```
   KALSHI_KEY_ID=your-key-id
   KALSHI_PRIVATE_KEY_PATH=/path/to/key.pem
   PAPER_MODE=false
   ```
5. Run: `python main.py --live`

### Polymarket
Signal generation and market discovery are fully implemented. Live order placement (CLOB signing) is the next milestone — contributions welcome.

---

## Safety

| Guard | Trigger | Effect |
|---|---|---|
| `StaleDataHalt` | All forecasts > `max_forecast_age_hours` old | Block all new entries |
| `ClusterCapExceeded` | Cluster exposure + proposed > `cluster_cap_usd` | Block that trade |
| `DailyLossHalt` | Realized + unrealized MTM loss ≥ limit | Block all new entries |
| `CityLimitExceeded` | Open positions in city ≥ `max_positions_per_city` | Block that trade |
| Reconciliation | Any startup — local DB vs. exchange diff | Flag critical discrepancies before first trade |

All trades are **paper by default**. Live trading requires explicit `--live` flag every invocation — there is no persistent "live mode" setting that could accidentally persist.

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt pytest

# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_kalshi_client.py -v
python -m pytest tests/test_phase_b.py -v

# Run calibration
python main.py --calibrate

# Run one autoresearch cycle
python main.py --autoresearch
```

Tests: **440 passing**, 2 skipped (scipy differential evolution — install scipy to enable).

---

## Roadmap

### Recently Completed (V5 Audit)
- [x] Markets persisted to DB before signal generation (market_id always populated)
- [x] Full entry snapshots stored in `positions.entry_reason` JSON
- [x] Bias correction integrated into ValueEntry signal pipeline
- [x] Dynamic forecast uncertainty (`dynamic_std_f`) — lead time + model spread + season
- [x] `ExchangeExecutor` abstraction (`PaperExecutor` / `KalshiExecutor`)
- [x] Autoresearch search space expanded to 5 params + 2-param combo proposals
- [x] Dashboard: QuickStats bar, Last Cycle panel, Run Experiment button, 3 new endpoints

### Next Milestones
- [ ] Polymarket live execution (CLOB order signing)
- [ ] EMOS post-processing on raw forecasts (proper ensemble calibration)
- [ ] Maker-first execution (Kalshi maker rebate exploitation)
- [ ] Cross-venue arbitrage scanner (Kalshi ↔ Polymarket same-contract price differences)
- [ ] Ensemble dispersion signal (trade adjacent bins when model spread high, market narrow)
- [ ] Grafana dashboard integration
- [ ] Docker + Fly.io deployment
- [ ] Slack/Discord trade notifications
- [ ] Walk-forward optimization on strategy router weights (not just `base_std_f`)
