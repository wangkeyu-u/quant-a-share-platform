"""Synthetic timing/accounting regressions; no market or model API calls."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant.backtest.engine import Backtest
from quant.backtest.metrics import compute_metrics
from quant.trading.paper import EXECUTION_PROTOCOL, PaperTrader


def bars(opens=(100, 200, 150), closes=(100, 220, 160)):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-08"]),
        "open": opens, "close": closes,
    }, index=[10, 20, 30])


class ExecutionTests(unittest.TestCase):
    def run_backtest(self, frame, signals, **kwargs):
        return Backtest(frame, initial_cash=1000, commission=0, slippage=0, **kwargs).run(
            pd.Series(signals, index=frame.index))

    def test_gap_fills_at_next_observed_open_and_marks_at_close(self):
        result = self.run_backtest(bars(), [1, 0, 1])
        self.assertEqual(result["equity"].tolist(), [1000, 1100, 750])
        self.assertEqual([t["price"] for t in result["trades"]], [200, 150])
        self.assertEqual([t["side"] for t in result["trades"]], ["BUY", "SELL"])
        self.assertEqual([t["signal_date"] for t in result["trades"]], bars().date.tolist()[:2])
        self.assertEqual([t["date"] for t in result["trades"]], bars().date.tolist()[1:])
        self.assertEqual(result["execution_protocol"], EXECUTION_PROTOCOL)

    def test_same_close_windfall_is_not_earned(self):
        result = self.run_backtest(bars((100, 200, 200), (100, 200, 200)), [1, 0, 0])
        self.assertEqual(result["equity"].tolist(), [1000, 1000, 1000])
        self.assertEqual(result["metrics"]["总收益率"], 0)

    def test_terminal_signal_waits_and_single_bar_stays_cash(self):
        for frame in (bars(), bars().iloc[:1]):
            result = self.run_backtest(frame, [0] * (len(frame) - 1) + [1])
            self.assertEqual(result["trades"], [])
            self.assertEqual(result["equity"].tolist(), [1000] * len(frame))

    def test_close_change_does_not_change_that_open_fill(self):
        base = bars()
        changed = bars()
        changed.loc[20, "close"] = 400
        first = self.run_backtest(base, [1, 0, 0])
        second = self.run_backtest(changed, [1, 0, 0])
        self.assertEqual(first["trades"], second["trades"])
        self.assertNotEqual(first["equity"].iloc[1], second["equity"].iloc[1])

    def test_fees_and_paper_backtest_share_accounting(self):
        frame = bars()
        kwargs = dict(initial_cash=1000, commission=.01, slippage=.01)
        result = Backtest(frame, **kwargs).run(pd.Series([1, 0, 0], index=frame.index))
        paper = PaperTrader(**kwargs)
        for row, signal in zip(frame.itertuples(), [1, 0, 0]):
            paper.step_bar(row.date, row.open, row.close, signal)
        expected_final = 1000 / (200 * 1.02) * 150 * .98
        self.assertAlmostEqual(result["equity"].iloc[-1], expected_final)
        self.assertEqual(result["equity"].tolist(), paper.equity_curve)
        self.assertEqual(result["trades"], paper.trades)
        paper.reset()
        paper.step_bar(frame.date.iloc[0], 100, 100, 0)
        self.assertEqual(paper.trades, [])
        self.assertEqual(paper.equity_curve, [1000])

    def test_repeated_run_resets_trades_and_preserves_original_index(self):
        frame = bars()
        engine = Backtest(frame)
        signals = pd.Series({30: 0, 10: 1, 20: 0})
        first = engine.run(signals)
        second = engine.run(signals)
        self.assertEqual(first["trades"], second["trades"])
        self.assertEqual(len(second["trades"]), 2)
        self.assertEqual(second["equity"].index.tolist(), [10, 20, 30])
        self.assertEqual(engine.run(signals * 0)["trades"], [])

    def test_benchmark_buys_once_with_same_costs_and_stake(self):
        result = Backtest(bars(), initial_cash=1000, commission=.01, slippage=.01,
                          stake=.5).run(pd.Series(0., index=bars().index))
        # 2.5 shares at 200 plus 10 in costs; cash=490, held to final close.
        self.assertEqual(result["benchmark"].tolist(), [1000, 1040, 890])
        full = Backtest(bars(), initial_cash=1000, commission=.01, slippage=.01).run(
            pd.Series(0., index=bars().index))
        self.assertAlmostEqual(full["benchmark"].iloc[-1], 1000 / 204 * 160)

    def test_invalid_bar_leaves_account_and_pending_signal_unchanged(self):
        for date, op, cl, signal in [
            ("2026-01-02", 100, 100, 1), (None, 100, 100, 1),
            ("2026-01-05", 0, 100, 1), ("2026-01-05", 100, float("nan"), 1),
            ("2026-01-05", 100, 100, 2), ("2026-01-05", 100, 100, float("inf")),
        ]:
            with self.subTest(bar=(date, op, cl, signal)):
                paper = PaperTrader(initial_cash=1000, commission=0, slippage=0)
                paper.step_bar("2026-01-02", 100, 100, 1)
                with self.assertRaises(ValueError):
                    paper.step_bar(date, op, cl, signal)
                self.assertEqual(paper.trades, [])
                self.assertEqual(paper.pending_signal, 1)
                paper.step_bar("2026-01-05", 200, 200, 0)
                self.assertEqual(paper.shares, 5)

    def test_invalid_backtest_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            Backtest(bars().iloc[:0])
        for kwargs in (dict(stake=2), dict(commission=-.1), dict(slippage=1), dict(initial_cash=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PaperTrader(**kwargs)
        duplicate = bars()
        duplicate.index = [0, 0, 1]
        with self.assertRaises(ValueError):
            Backtest(duplicate)

    def test_pipeline_reports_engine_benchmark_and_protocol(self):
        from quant.pipeline import runner
        frame = bars()
        frame.attrs["data_source"] = "akshare"
        gateway = SimpleNamespace(log=[], remote=None, log_summary=lambda: {})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "REPORTS_DIR", tmp), \
             patch.object(runner, "get_gateway", return_value=gateway), \
             patch.object(runner, "load_data", return_value=frame), \
             patch.object(runner, "train_models", return_value={}), \
             patch.object(runner, "walk_forward_signals", return_value=pd.Series([1, 0, 0], index=frame.index)), \
             patch.object(runner, "list_strategies", return_value=[]):
            report, path = runner.run_pipeline(["SYNTHETIC"], end="20260108")
            ml = report["symbols"]["SYNTHETIC"]["ml"]
            expected = Backtest(frame).run(pd.Series(0., index=frame.index))
            self.assertEqual(ml["benchmark"], runner._round(compute_metrics(expected["benchmark"])))
            self.assertEqual(ml["execution_protocol"], EXECUTION_PROTOCOL)
            self.assertEqual(report["data_source"], "akshare")
            with open(path) as source:
                self.assertEqual(json.load(source), report)


if __name__ == "__main__":
    unittest.main()
