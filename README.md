# Weather Trading Bot

An automated trading bot for weather prediction markets on [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com). It fetches real-time weather forecasts, computes edge against market prices using a probabilistic model, manages positions with a full lifecycle engine, and continuously self-improves via autoresearch.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration (.env)](#configuration-env)
5. [Paper Trading](#paper-trading)
6. [Live Trading (Kalshi)](#live-trading-kalshi)
7. [Dashboard](#dashboard)
8. [Commands Reference](#commands-reference)
9. [Strategies](#strategies)
10. [Risk Management](#risk-management)
11. [Autoresearch & Calibration](#autoresearch--calibration)
12. [Project Structure](#project-structure)
13. [Running Tests](#running-tests)

---

## How It Works

Each trading cycle (default: every 5 minutes):

1. **Fetch markets** — pulls open weather contracts from Kalshi and Polymarket
2. **Fetch forecasts** — pulls multi-model weather forecasts (Open-Meteo: GFS, ECMWF, etc.) for each city/date pair
3. **Compute fair value** — runs a Monte Carlo probability engine with bias correction and Platt temperature scaling to price each contract
4. **Generate signals** — four strategies evaluate edge, confidence, and risk filters
5. **Route capital** — a scorecard allocates budget per strategy based on Sharpe, Brier score, execution quality, and drawdown
6. **Manage positions** — exits positions on convergence, stop-loss, or pre-settlement; settles expired contracts via NWS/IEM data
7. **Log everything** — all signals, orders, fills, and PnL written to SQLite

---

## Architecture

```
main.py                  ← entry point, CLI flags, cycle orchestration
│
├── strategy_router/
│   ├── brain.py         ← orchestrates one full cycle
│   ├── scorecard.py     ← per-strategy scoring (Sharpe, Brier, etc.)
│   ├── allocator.py     ← softmax capital allocation
│   └── selector.py      ← signal filtering, Kelly sizing, budget tracking
│
├── strategies/
│   ├── value_entry.py   ← core fair-value entry + convergence exit
│   ├── convergence_exit.py
│   ├── model_release.py ← trade on new forecast model releases
│   └── disagreement.py  ← trade when models disagree with market
│
├── core/
│   ├── forecaster.py    ← canonical fair value pipeline (bias + MC + scaling)
│   └── signals.py
│
├── clients/
│   ├── kalshi_client.py ← Kalshi REST API
│   ├── polymarket_client.py
│   ├── weather.py       ← Open-Meteo multi-model fetcher + DB store
│   └── nws_settlement.py← NWS/IEM settlement price fetcher
│
├── execution/
│   ├── exchange_executor.py ← PaperExecutor / KalshiExecutor
│   ├── lifecycle.py     ← position exit engine (convergence, stop, pre-settle)
│   └── orderbook.py     ← fee/slippage math
│
├── risk/
│   ├── guards.py        ← stale-data halt, cluster cap, daily loss limit
│   └── reconciliation.py
│
├── research/
│   ├── calibrator.py    ← walk-forward Brier + temperature scaling optimizer
│   ├── autoresearch.py  ← automated experiment proposal + promotion
│   ├── walk_forward.py  ← walk-forward backtester
│   └── bias_correction.py
│
├── state/db.py          ← SQLite schema, WAL mode, migrations
├── shared/params.py     ← all tunable constants (Params dataclass)
└── dashboard/
    ├── api.py           ← Flask REST API (reads bot.db)
    └── index.html       ← single-file React dashboard
```

---

## Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/lucaskardo/weather-trading-bot.git
cd weather-trading-bot

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:
```
requests>=2.31
python-dotenv>=1.0
numpy>=1.26
scipy>=1.12
flask>=3.0
```

---

## Configuration (.env)

Create a `.env` file in the project root. All fields are optional — the bot runs with defaults if the file is missing.

```ini
# .env

# Starting bankroll (only applied on first run when DB is at default $1000)
BANKROLL=5000

# Hard daily loss limit in dollars (overrides the default 5%-of-bankroll rule)
DAILY_LOSS_LIMIT=250

# How often to poll in continuous loop mode (seconds)
POLL_INTERVAL_S=300

# NOAA CDO token — used for historical settlement data
# Free at: https://www.ncdc.noaa.gov/cdo-web/token
# Without this, settlement falls back to IEM ASOS data (no auth required)
NOAA_CDO_TOKEN=your_token_here

# Kalshi API credentials — only required for live trading
KALSHI_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/rsa_private_key.pem
```

Verify your config is loaded correctly:

```bash
python main.py --diagnose
```

---

## Paper Trading

Paper trading uses real market data and real forecasts, but executes orders to your local SQLite database instead of the exchange. No money is at risk.

### Quick start — single cycle

```bash
python main.py --paper
```

Fetches live markets and forecasts, runs all strategies, logs any signals to `data/bot.db`. Prints a summary:

```
──────────────────────────────────────────────────
  CONFIG
──────────────────────────────────────────────────
  .env file        found
  Bankroll         $5000.00
  Daily loss limit $250.00
  Poll interval    300s
  NOAA CDO token   set
  Kalshi API key   not set (only needed for live trading)

[main] kalshi: 14 markets fetched
[main] polymarket: 3 markets fetched
[main] 12 forecast(s) for 6 city/date pair(s)
{'cycle_at': '...', 'signals_generated': 4, 'signals_executable': 2, 'executed': 2, ...}
```

### Continuous paper trading loop

```bash
python main.py --paper --loop
```

Runs a cycle every `POLL_INTERVAL_S` seconds (default 300). Keep this running in a terminal or via a process manager.

### Check current status

```bash
python main.py --status
```

Shows bankroll, cash available, all open positions, and total realized PnL — without running a trading cycle.

### Deploying for continuous paper testing

**Using `nohup` (simplest):**

```bash
nohup python main.py --paper --loop > logs/paper.log 2>&1 &
echo $! > logs/paper.pid

# Stop it:
kill $(cat logs/paper.pid)
```

**Using `screen`:**
```bash
screen -S weatherbot
python main.py --paper --loop
# Ctrl+A D to detach
# screen -r weatherbot to reattach
```

**Using a cron job (single cycle every 5 minutes):**
```bash
crontab -e
# Add:
*/5 * * * * cd /path/to/weather-trading-bot && venv/bin/python main.py --paper >> logs/paper.log 2>&1
```

---

## Live Trading (Kalshi)

> **Warning:** Live trading places real orders with real money. Run paper trading first and confirm the bot behaves as expected before going live.

### Prerequisites

1. A funded [Kalshi](https://kalshi.com) account
2. API credentials (Key ID + RSA private key) from your Kalshi account settings
3. Set credentials in `.env`:
   ```ini
   KALSHI_KEY_ID=your_key_id
   KALSHI_PRIVATE_KEY_PATH=/path/to/rsa_private_key.pem
   ```
4. Verify connectivity:
   ```bash
   python main.py --diagnose
   ```

### Run a single live cycle

```bash
python main.py --live
```

### Continuous live trading

```bash
python main.py --live --loop
```

With a custom poll interval:

```bash
python main.py --live --loop --interval 600
```

### Switching from paper to live

The bot uses the same `data/bot.db` for both modes. Positions logged during paper trading remain in the DB and will affect capital allocation and risk guards. For a clean slate when going live:

```bash
rm data/bot.db      # wipes all paper history
python main.py --live
```

---

## Dashboard

The dashboard is a single-file React app backed by a Flask API server. It reads directly from `data/bot.db` and shows live portfolio stats, open positions, strategy performance, forecast quality, and trade history.

### Start the dashboard

**Terminal 1 — API server:**
```bash
python main.py --dashboard
# or:
python dashboard/api.py
```

**Terminal 2 — open in browser:**
Double-click `dashboard/index.html`, or serve it locally:
```bash
open dashboard/index.html      # macOS
xdg-open dashboard/index.html  # Linux
```

The API listens on `http://localhost:5001` by default. Custom port:
```bash
python main.py --dashboard --port 5002
```

### Running bot + dashboard together

```bash
# Terminal 1 — trading loop
python main.py --paper --loop

# Terminal 2 — dashboard API
python main.py --dashboard

# Browser — open dashboard/index.html
```

The bot posts each cycle's summary to the dashboard automatically. The "Last Cycle" panel at the top of Tab 1 updates on refresh.

### Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| **Live Trading** | Bankroll KPIs, open positions, strategy allocation, cluster exposure, recent signals with freshness badges (● LIVE / Xm ago) |
| **Strategy Performance** | Per-strategy win rate, total PnL, avg hold time, MFE/MAE, Brier score trend |
| **Autoresearch** | Experiment registry, promotion candidates, run-experiment button |
| **Forecast Quality** | Per-city/model forecast bias, model agreement spread, settlement vs forecast residuals |
| **History** | Closed positions table with full entry/exit detail, daily PnL history with cumulative totals |

---

## Commands Reference

```bash
# Test all API connections and DB health
python main.py --diagnose

# Show portfolio + open positions (no trading)
python main.py --status

# Single paper cycle
python main.py --paper

# Continuous paper loop (5-minute default interval)
python main.py --paper --loop

# Continuous paper loop with custom interval
python main.py --paper --loop --interval 120

# Single live cycle
python main.py --live

# Continuous live loop
python main.py --live --loop

# Run calibration (optimises temperature scaling T)
python main.py --calibrate

# Run one autoresearch experiment
python main.py --autoresearch

# Start dashboard API server
python main.py --dashboard

# Dashboard on a custom port
python main.py --dashboard --port 5002
```

---

## Strategies

Four strategies run every cycle. Each generates signals independently; the router allocates capital based on recent performance.

### ValueEntry
Enters when the model's fair value diverges from the market price by more than `min_executable_edge` (default 5¢ after fees). Uses a Monte Carlo probability engine over multi-model consensus forecasts with dynamic uncertainty scaling and Platt temperature calibration.

Also handles **ConvergenceExit**: exits a position mid-trade if fair value has converged toward the market price (edge gone) or the position has moved against the model.

### ModelRelease
Trades the information event when a new forecast model run is released (e.g. the 00Z GFS cycle). Prices can move when a major model shifts; this strategy enters quickly and exits once the market has repriced.

### Disagreement
Trades when two or more models disagree significantly (high spread in °F). Takes a position in the direction of the consensus, betting the outlier model is wrong.

### Capital routing

Each strategy is scored on a 0–100 scale:

| Component | Weight |
|-----------|--------|
| Sharpe ratio | 35% |
| Brier score (calibration) | 30% |
| Execution quality | 20% |
| Max drawdown | 10% |
| Score instability | 5% |

Capital is allocated via softmax over scores, then Kelly criterion sizes each individual bet (capped at 25% of allocated budget per trade).

---

## Risk Management

### Hard limits

| Guard | Default | Override |
|-------|---------|----------|
| Daily loss limit | 5% of bankroll | `DAILY_LOSS_LIMIT=250` in `.env` |
| Cluster exposure cap | 15% of bankroll per geographic cluster | `params.max_cluster_exposure_pct` |
| Max positions per city | 3 | `params.max_positions_per_city` |
| Stale data halt | Halts if best forecast > 6h old | `params.stale_forecast_hours` |
| Max Kelly fraction | 25% per trade | `params.max_kelly_fraction` |
| Min executable edge | 5¢ after fees/slippage | `params.min_executable_edge` |

**Geographic clusters** (each capped independently): Northeast (NYC/BOS/DC), Midwest (CHI/DAL), South (MIA/HOU/ATL), West (LA/SF/SEA), Europe (LON/PAR/MUN), Asia (SEO), South America (BUE/SAO).

### Position exits

| Exit type | Trigger |
|-----------|---------|
| **Settlement** | NWS/IEM actual temperature fetched after market close → WON or LOST |
| **Convergence** | Fair value converges to market price (edge below threshold) |
| **Stop-loss** | Position moves against model by configurable amount |
| **Pre-settlement** | Liquidity dries up in final hours; exit early |

---

## Autoresearch & Calibration

### Calibration

Calibration optimises the Platt temperature scaling parameter `T` using walk-forward cross-validation on resolved trades:

```bash
python main.py --calibrate
```

The calibrator reads all resolved predictions from `data/bot.db`, runs 5-fold walk-forward Brier score evaluation, and updates `PARAMS.temp_scaling_T` in-process.

### Autoresearch

Autoresearch proposes and tests parameter mutations (edge thresholds, std scaling, Kelly fractions, etc.) using the same walk-forward backtester. Experiments that improve Brier score are automatically promoted to live parameters.

```bash
# Run one experiment cycle from CLI
python main.py --autoresearch

# Or trigger from the dashboard (Autoresearch tab → ▶ Run Experiment)
```

Requires at least 6 resolved trades in the database. All experiments are logged to the `experiments` table and visible in the dashboard.

---

## Project Structure

```
weather-trading-bot/
├── main.py                     # Entry point + all CLI flags
├── requirements.txt
├── .env                        # Your config (not committed)
├── data/
│   └── bot.db                  # SQLite database (auto-created on first run)
├── logs/                       # Log files
├── clients/
│   ├── kalshi_client.py        # Kalshi REST API client
│   ├── polymarket_client.py    # Polymarket API client
│   ├── weather.py              # Open-Meteo multi-model weather fetcher
│   └── nws_settlement.py       # NWS/IEM settlement fetcher (no auth required)
├── core/
│   └── forecaster.py           # Canonical fair value pipeline
├── strategies/
│   ├── base.py                 # Signal dataclass, BaseStrategy
│   ├── value_entry.py          # ValueEntry + convergence exit
│   ├── convergence_exit.py
│   ├── model_release.py
│   └── disagreement.py
├── strategy_router/
│   ├── brain.py                # Cycle orchestrator
│   ├── scorecard.py            # Strategy scoring
│   ├── allocator.py            # Capital allocation
│   └── selector.py             # Signal filtering + Kelly sizing
├── execution/
│   ├── exchange_executor.py    # PaperExecutor / KalshiExecutor
│   ├── lifecycle.py            # Position exit engine
│   └── orderbook.py            # Fee/slippage math
├── risk/
│   ├── guards.py               # Hard risk guards
│   └── reconciliation.py       # Position reconciliation
├── research/
│   ├── calibrator.py           # Temperature scaling calibration
│   ├── autoresearch.py         # Automated experiment engine
│   ├── walk_forward.py         # Walk-forward backtester
│   └── bias_correction.py      # Forecast bias database
├── state/
│   └── db.py                   # SQLite schema + init
├── shared/
│   ├── params.py               # All tunable parameters (Params dataclass)
│   └── types.py                # Shared type definitions
├── dashboard/
│   ├── api.py                  # Flask REST API server
│   └── index.html              # Single-file React dashboard
└── tests/                      # 459 unit + integration tests
```

---

## Running Tests

```bash
python3 -m pytest tests/ -q
```

Expected output:
```
459 passed, 2 skipped in 0.50s
```

Run a specific test file:
```bash
python3 -m pytest tests/test_v8_bugs.py -v
python3 -m pytest tests/test_strategies.py -v
python3 -m pytest tests/test_risk_guards.py -v
```

---

## Troubleshooting

**`0 markets fetched` from Kalshi**
Kalshi's weather series ticker names change seasonally. Run `--diagnose` to see what's live. The client searches by keyword and should auto-discover active series.

**`Not enough resolved trades to run experiments`**
Autoresearch requires at least 6 settled positions. Run paper trading for a few days until contracts expire and settle.

**`StaleDataHalt`**
The Open-Meteo API may be temporarily unavailable. The bot retries on the next cycle. Check with `--diagnose`.

**Dashboard shows no data**
Make sure `python main.py --dashboard` is running before opening `index.html`. The API must be on `localhost:5001`.

**`Cannot start live mode: missing KALSHI_KEY_ID`**
Add your Kalshi API credentials to `.env`. See [Live Trading](#live-trading-kalshi).
