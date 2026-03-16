# Go-Live Checklist

## Required before live trading

- [ ] Full test suite green
- [ ] Kalshi smoke test passes
- [ ] Polymarket smoke test passes
- [ ] Reconciliation smoke preview shows zero critical discrepancies
- [ ] Dashboard endpoints load correctly
- [ ] Paper trading persists positions, orders, fills, and predictions
- [ ] Portfolio risk endpoint within limits
- [ ] Revision audit timestamps populated for revision-sensitive trades
- [ ] Recent paper metrics meet internal thresholds

## Recommended launch sequence

1. Run 24-48h paper with dashboard open.
2. Run smoke scripts again.
3. Reconcile local positions against live market snapshots.
4. Start with smallest live capital tranche.
5. Monitor:
   - `/api/portfolio_risk`
   - `/api/truth_metrics`
   - `/api/revision_audit`
   - `/api/source_health`

## Abort conditions

- Any critical reconciliation discrepancy
- Missing or stale forecast sources
- Daily drawdown kill-switch triggered
- Calibration drift exceeds tolerance
- API smoke scripts fail
