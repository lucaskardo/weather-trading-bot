# Live API smoke tests

Run these before enabling real capital.

```bash
python scripts/smoke_kalshi.py --series KXHIGHNY
python scripts/smoke_polymarket.py
```

What they verify:
- environment variables present for Kalshi live mode
- weather markets can be fetched
- a sample orderbook can be read from Kalshi
- Polymarket weather markets can be fetched and normalized

They do **not** place orders.
