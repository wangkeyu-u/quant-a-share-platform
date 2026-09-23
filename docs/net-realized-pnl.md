# Realized P&L excluded trading costs

Recorded 2026-09-24. This repairs trade-level accounting, not market-performance evidence.

## Failure

`PaperTrader` debited commission and proportional slippage from cash on entry and exit, but stored `avg_cost` at the gross buy price. A sell calculated `pnl` as the gross sell-minus-buy price difference. `trade_stats` used that value for win rate and average gain/loss, so these statistics could disagree with the equity curve.

With 1,000 initial cash, 1% commission, a buy at 100 and a sell at 101, the pre-fix account ended at 990.00 while the sell record showed `pnl=+9.90` and win rate `1.0`. These are synthetic bars used to isolate the accounting error, not market results.

## Decision

Store average acquisition cost including entry costs already debited from cash. On each sale, realized `pnl` equals net proceeds after exit costs minus that cost basis allocated to the shares sold. Partial sales retain the same per-share basis for the remaining shares. The account's cash and equity equations do not change; the trade record and derived statistics now use the same costs.

Fee-free trades retain their previous P&L. Old saved reports are not recalculated and should not be compared to new trade win rates without regenerating them. `float_pnl` now includes entry costs in its basis but, as an unrealized figure, does not pre-charge a future exit fee.

## Reproduce and validate

```bash
python3 -m unittest discover -s tests -p test_execution_timing.py -v
python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. AI_GATEWAY_BASE_URL= AI_GATEWAY_API_KEY= python3 tests/headless_test.py
```

The regression uses actual `Backtest` and `PaperTrader` execution with synthetic prices. It checks that a gross price gain smaller than costs is a net loss, that win rate follows that loss, and that partial exits' realized P&L reconciles to final cash after the position closes. All 30 offline unit tests and the existing headless full-chain checks passed locally. The latter uses explicit synthetic data; no AkShare or remote model call is needed. Its simulated metrics are not market results.

This does not resolve adjusted-data revisions, survivorship, exchange fill constraints or strategy-selection bias.
