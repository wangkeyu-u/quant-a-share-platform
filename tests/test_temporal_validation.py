"""Temporal data boundaries using synthetic prices, with no market/API calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from quant.ai.backends import LocalBackend
from quant.ml.features import CLS_TARGET, FEATURE_COLS, FEATURE_DATE, LABEL_END, REG_TARGET, build_features
from quant.ml import trainer


def prices(n=360, seed=7):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, .02, n)))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                         "open": close, "high": close * 1.01, "low": close * .99,
                         "close": close, "volume": rng.integers(100, 10000, n)})


class RecordingGateway:
    def __init__(self):
        self.fits = []
        self.scores = []

    def fit_pair(self, X, yc, yr, *args):
        self.fits.append((X.copy(), yc.copy(), yr.copy()))
        return float(yc.mean()), float(yr.mean())

    def score_pair(self, pair, X):
        self.scores.append((X.index[0], pair))
        return pair


class TemporalTests(unittest.TestCase):
    def test_horizon_changes_target_and_terminal_labels_stay_missing(self):
        df = prices()
        for horizon in (1, 5, 20):
            with self.subTest(horizon=horizon):
                f = build_features(df, horizon)
                self.assertAlmostEqual(f.loc[150, REG_TARGET], df.close.iloc[150+horizon] / df.close.iloc[150] - 1)
                self.assertEqual(f.loc[150, LABEL_END], df.date.iloc[150+horizon])
                self.assertTrue(f[REG_TARGET].tail(horizon).isna().all())
                self.assertTrue(pd.isna(f[CLS_TARGET].iloc[-1]))

    def test_invalid_horizons_and_dates_fail_explicitly(self):
        for horizon in (0, -1, 2.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                build_features(prices(), horizon)
        for df in (prices().iloc[::-1], pd.concat([prices().iloc[:5], prices().iloc[:5]])):
            with self.assertRaises(ValueError):
                build_features(df)

    def test_future_prices_cannot_change_observed_features(self):
        df = prices()
        changed = df.copy()
        changed.loc[250:, ["open", "high", "low", "close"]] *= 17
        assert_frame_equal(build_features(df)[FEATURE_COLS].iloc[:250],
                           build_features(changed)[FEATURE_COLS].iloc[:250])

    def test_train_inputs_equal_rebuilding_only_available_history(self):
        df = prices(280)
        gw = RecordingGateway()
        with patch.object(trainer, "get_gateway", return_value=gw):
            trainer.walk_forward_signals(df, horizon=20, lookback=220, retrain_every=100)
        self.assertEqual(len(gw.fits), 1)
        expected = trainer._clean(build_features(df.iloc[:220], 20))
        X, yc, yr = gw.fits[0]
        assert_frame_equal(X, expected[FEATURE_COLS])
        assert_series_equal(yc, expected[CLS_TARGET])
        assert_series_equal(yr, expected[REG_TARGET])
        self.assertEqual(gw.scores[-1][0], 279)  # Predict recent rows without future labels.

    def test_future_perturbation_preserves_past_model_scores(self):
        df = prices(280)
        changed = df.copy()
        changed.loc[250:, ["open", "high", "low", "close"]] *= 17
        scores = []
        for data in (df, changed):
            gw = RecordingGateway()
            with patch.object(trainer, "get_gateway", return_value=gw):
                trainer.walk_forward_signals(data, horizon=20, lookback=220, retrain_every=1)
            scores.append([(i, pair) for i, pair in gw.scores if i < 250])
        self.assertTrue(scores[0])
        self.assertEqual(scores[0], scores[1])

    def test_holdout_and_cv_purge_by_symbol_label_end_and_group_dates(self):
        a, b = prices(500), prices(500, 11).iloc[::2].reset_index(drop=True)
        pool = trainer._clean(trainer._pool_frame([("a", a), ("b", b)], 20)).sort_values(FEATURE_DATE)
        tr, te = trainer._holdout(pool, .2)
        self.assertLess(tr[LABEL_END].max(), te[FEATURE_DATE].min())
        self.assertFalse(set(tr[FEATURE_DATE]) & set(te[FEATURE_DATE]))
        for train_idx, val_idx in trainer._purged_splits(tr):
            train, val = tr.iloc[train_idx], tr.iloc[val_idx]
            self.assertLess(train[LABEL_END].max(), val[FEATURE_DATE].min())
            self.assertFalse(set(train[FEATURE_DATE]) & set(val[FEATURE_DATE]))
            expected = tr[tr[FEATURE_DATE].isin(val[FEATURE_DATE])]
            assert_frame_equal(val, expected)

    def test_pooled_training_uses_each_symbols_observed_sessions(self):
        target = prices(320)
        sparse = prices(320, 9).iloc[::2].reset_index(drop=True)
        gw = RecordingGateway()
        with patch.object(trainer, "get_gateway", return_value=gw):
            trainer.walk_forward_pooled(target, [("a", target), ("b", sparse)], "unused", horizon=20,
                                        lookback=300, retrain_every=100, window=160)
        parts = []
        for data in (target, sparse):
            known = data[data.date < target.date.iloc[300]]
            f = trainer._clean(build_features(known, 20))
            parts.append(f[f[FEATURE_DATE] >= target.date.iloc[140]])
        expected = pd.concat(parts)[FEATURE_COLS].sort_values(FEATURE_COLS).reset_index(drop=True)
        actual = gw.fits[0][0].sort_values(FEATURE_COLS).reset_index(drop=True)
        assert_frame_equal(actual, expected)

    def test_all_failed_cv_scores_are_null_not_infinity(self):
        with patch.object(trainer, "cross_val_score", side_effect=ValueError("one class")):
            _, score = trainer._best_params(None, None, lambda p: object(), "roc_auc", [])
        self.assertIsNone(score)

    def test_real_estimators_single_and_pooled_training(self):
        # A small grid is an integration smoke check, not performance tuning.
        grid = [{"n_estimators": 5, "max_depth": 2, "learning_rate": .1}]
        df = prices(380)
        with TemporaryDirectory() as directory, patch.object(trainer, "PARAM_GRID", grid), \
                patch.object(trainer, "get_gateway", return_value=LocalBackend()), \
                patch.object(trainer, "MODELS_DIR", directory):
            for pooled in (False, True):
                path = str(Path(directory) / f"{pooled}.joblib")
                if pooled:
                    report = trainer.train_pooled([("a", df), ("b", prices(380, 11))], path, horizon=20)
                else:
                    report = trainer.train_models(df, "synthetic", horizon=20, model_path=path)
                self.assertGreater(report["purged_n"], 0)
                self.assertEqual(joblib.load(path)["horizon"], 20)
                self.assertEqual(report["label_protocol"], "available-before-cutoff-v1")
                self.assertGreater(report["train_n"], 50)


if __name__ == "__main__":
    unittest.main()
