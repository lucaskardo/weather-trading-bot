# Timestamp & Latency Integrity Rules

This bot records a minimum viable timestamp trail for revision-sensitive decisions.

Required fields for any revision-driven or model-release trade:

- `provider_publish_time`: when the upstream forecast provider published the model output
- `model_run_time`: model cycle identifier, e.g. `YYYYMMDDCC`
- `bot_fetch_time`: when the bot retrieved the forecast
- `parse_to_signal_time`: when the forecast was converted into a signal candidate
- `market_snapshot_time`: when the market prices used for the decision were observed
- `order_sent_time`: when the order was sent to the executor
- `fill_received_time`: when the simulated/live fill response was received

Current storage:
- forecast lineage is stored in `forecasts`
- decision timing trail is stored in `decision_audit`

Operational rule:
- revision-style signals should be treated as valid only when the revision is confirmed by either
  - consistent run-to-run momentum, or
  - multiple watched models moving in the same direction

This keeps paper and live revision alpha from being overstated by stale or noisy model releases.
