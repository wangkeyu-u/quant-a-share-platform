# Forward labels crossed the training cutoff

Recorded 2026-09-16 against the implementation before this fix. This is a code-level timing defect; no corrected market-performance result is claimed.

## Failure

The old path computed `fwd5 = close.shift(-5) / close - 1` over the complete history, then passed rows `[lo:t]` to training. A row at `t-1` therefore contained the price at `t+4`. Excluding the feature date `t` did not exclude future outcomes.

Pooled training filtered `date < t`, with the same problem. Its holdout also split a sorted row array, so multiple symbols observed on the same date could land on opposite sides. Ordinary `TimeSeriesSplit` had no label-window purge.

Two additional defects shared this path: the terminal classification label converted an unknown next close into zero, and `horizon` changed loop bounds/model metadata without changing the fixed five-session regression target. Ending the prediction loop at `n-horizon` also forced recent observations to a zero position despite inference needing no future label.

## Fix

`build_features(df, horizon)` records observation date, target-end date, next-session direction and horizon-session return. Terminal outcomes stay missing. Dates must be unique and increasing per symbol.

All training paths require both `feature_date < cutoff` and `label_end < cutoff`. Holdout and CV split distinct dates, then purge training rows whose target end reaches the first validation date. The same rule handles symbols with different observed calendars; a five-row global gap would not.

The classifier and regressor share this conservative set. Classification could use more recent data than the horizon regressor, but separate training populations are not implemented. Here, a session means the next observed row for that symbol, not a calendar day.

`target_return` is the active regression target; `fwd5` and `fwd20` remain diagnostic columns. Saved models and training reports identify `available-before-cutoff-v1` and the requested horizon. Existing model files must be retrained. The pooled walk-forward `model_path` argument remains for call compatibility and does not load a pre-trained model.

## Validation

Run from the repository root:

```bash
python -m unittest discover -s tests -p test_temporal_validation.py -v
```

Nine tests exercise:

- horizons 1/5/20, missing terminal labels and invalid dates;
- observed features remaining unchanged when later prices are altered;
- training inputs matching a fresh feature build using only the visible prefix;
- earlier model scores remaining unchanged when future prices change;
- pooled date grouping and purging across sparse symbol calendars;
- the rolling pooled window and each symbol's own label horizon;
- an unavailable CV score being `None`, not negative infinity;
- actual sklearn classification/regression fitting, prediction and model save/load, in both single and pooled paths.

The causal tests use a recording gateway whose fitted score depends on the training targets. The integration test uses real GBM estimators with five trees and a small grid. Both use seeded synthetic OHLCV; neither calls market or model APIs. Environment and case results are retained in [temporal-validation.json](temporal-validation.json).

As a negative control, replacing `_known_before` with the old feature-date-only filter makes both the prefix-rebuild test and the future-perturbation test fail. This isolates the missing availability condition; it is not a benchmark against a reconstructed historical binary.

Reproduce that expected failure without editing source files:

```bash
python - <<'PY'
import sys, unittest
from unittest.mock import patch
sys.path.insert(0, 'tests')
from test_temporal_validation import TemporalTests
from quant.ml import trainer
from quant.ml.features import FEATURE_DATE
names = ['test_train_inputs_equal_rebuilding_only_available_history',
         'test_future_perturbation_preserves_past_model_scores']
suite = unittest.TestSuite(TemporalTests(name) for name in names)
with patch.object(trainer, '_known_before',
                  side_effect=lambda frame, cutoff: frame[frame[FEATURE_DATE] < cutoff]):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
assert len(result.failures) == 2 and not result.errors
PY
```

## Remaining limits

This fixes the identified label/cutoff leak, not the entire backtest methodology. Same-close execution of close-derived signals, revised adjusted prices, survivorship, in-sample strategy selection and source fallback still need separate work. CV scores can be unavailable when a fold has one class; the current parameter fallback remains the first grid entry and is not evidence of successful tuning.
