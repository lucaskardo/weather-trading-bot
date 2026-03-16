# Operator Runbook

## Daily paper-trading workflow

1. Run API smoke checks:
   - `python scripts/smoke_kalshi.py`
   - `python scripts/smoke_polymarket.py`
2. Run reconciliation preview:
   - `python scripts/smoke_reconciliation.py --exchange all`
3. Start paper trading with dashboard:
   - `python main.py --paper --dashboard`
4. Review dashboard panels:
   - Decision cockpit
   - Truth metrics
   - Portfolio risk
   - Revision audit
5. Review any reconciliation discrepancies before trusting paper PnL.

## Before enabling live capital

1. All smoke scripts pass.
2. Reconciliation preview reports zero critical discrepancies.
3. Dashboard health and source freshness look normal.
4. Recent paper Sharpe and Brier are within configured gates.
5. No stale-feed, calibration-drift, or drawdown kill-switch conditions are active.

## Incident handling

### Feed stale or source drift
- Halt new entries.
- Check `/api/source_health`.
- Re-run smoke scripts.

### Reconciliation discrepancy
- Run `python scripts/smoke_reconciliation.py --exchange all`.
- If critical, investigate DB/exchange mismatch before continuing.
- Only use `--auto-correct` after reviewing the preview output.

### Paper/live divergence
- Check `decision_audit`, `predictions`, `fills`, and dashboard truth metrics.
- Confirm current params and calibration profiles.
