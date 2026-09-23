# Close signals executed before they were available

Recorded 2026-09-22. This is an execution-timing repair, not a market-performance evaluation.

## Failure and decision

The previous engine used `position[i]` and `close[i]` in the same rebalance. Rules and ML features consume that close, so the simulated order received a price unavailable after the signal was calculated. Paper trading duplicated this logic. Pipeline benchmark metrics also used a separate, fee-free close-to-close calculation.

The engine now delegates account execution to `PaperTrader.step_bar`. Each bar executes the preceding close's target at its open, marks holdings at its close, then queues the new signal. Trades retain both `signal_date` and execution `date`. Dates must increase and open/close prices must be finite and positive. The first bar stays in cash; the final signal remains pending. “Next” means the next observed row for this symbol, not the next calendar day.

A next-close fill would also delay execution, but would skip the following session's intraday exposure. A same-close fill requires an intraday decision/cutoff model that this daily-data application does not have. Next observed open is the explicit convention used here, identified as `close-signal-next-observed-open-v1` in backtest, account and pipeline ML results.

The buy-and-hold benchmark buys once at the second observed open, with the same initial cash, `stake`, commission and proportional slippage. It holds without subsequent rebalancing or terminal liquidation. The strategy can rebalance daily. Pipeline metrics now consume this exact benchmark curve.

## Reproduce the failure

From the repository root after installing the README's test dependencies:

```bash
python - <<'PY'
import subprocess
import pandas as pd
from quant.backtest.engine import Backtest
old = {}
exec(subprocess.check_output(
    ['git', 'show', '77366f2:quant/backtest/engine.py'], text=True), old)
df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=3),
                   'open': [100, 200, 200], 'close': [100, 200, 200]})
for name, engine in [('previous', old['Backtest']), ('fixed', Backtest)]:
    result = engine(df, initial_cash=1000, commission=0, slippage=0).run(
        pd.Series([1, 0, 0]))
    print(name, result['equity'].tolist(), result['metrics']['总收益率'])
PY
```

Observed locally: previous `[1000, 2000, 2000]`, return `1.0`; fixed `[1000, 1000, 1000]`, return `0.0`. The signal at the first close cannot earn the jump from 100 to the next open at 200. This loads the actual pre-fix engine from Git; a shallow checkout needs that commit fetched first.

## Validation

`python -m unittest discover -s tests -p 'test_*.py' -v` passed **19 tests**: the existing nine label-boundary tests and ten execution tests. New cases cover gaps, signal/execution dates, terminal signals, open fills independent of that close, hand-calculated costs, paper/backtest parity, repeated runs, non-default indexes, buy-once benchmark accounting, invalid bars leaving state intact, and pipeline JSON agreement with engine metrics. Pipeline training/data/strategy-discovery calls are mocked; its reporting and backtest use real code. No market or remote model API is called.

Both GUI simulation actions use the same bar method. Their Python syntax was checked; interactive Tk behavior and the separate full `headless_test.py` smoke script were not run in this validation.

## Limits and compatibility

- `PaperTrader.step(date, price, target)` is replaced by `step_bar(date, open_price, close_price, signal)`; do not pre-shift signals when migrating. `Backtest.run` keeps its API and now preserves the input dataframe's index when aligning signals. Missing aligned signals mean a zero target.
- Old reports use different timing and benchmark assumptions and must be regenerated. They are not corrected results under this protocol.
- Observed opens are assumed fillable. There is no exchange calendar, suspension/price-limit/volume simulation, integer-lot sizing or forced final liquidation. Slippage remains a proportional cash cost; trade `price` records the bar open.
- Equity includes fees. Existing per-trade `pnl` and win-rate summaries still use gross price differences, not fee-adjusted realized P&L.
- Adjusted-data revisions, survivorship, in-sample strategy selection remain unresolved. Automatic synthetic fallback was subsequently removed in the [source-provenance repair](data-provenance.md). This repair does not establish tradable performance.
