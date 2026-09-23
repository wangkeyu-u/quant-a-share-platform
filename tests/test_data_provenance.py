"""Source isolation tests with a fake AkShare provider; no network calls."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from quant.data import loader, store


def provider_bars():
    return pd.DataFrame({
        "日期": ["2023-01-02", "2023-01-03"],
        "开盘": [10, 11], "最高": [11, 12], "最低": [9, 10],
        "收盘": [10.5, 11.5], "成交量": [1000, 1100],
    })


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_patch = patch.object(store, "CACHE_DIR", self.tmp.name)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def market(self, response=None, error=None):
        api = Mock()
        if error:
            api.stock_zh_a_hist.side_effect = error
        else:
            api.stock_zh_a_hist.return_value = provider_bars() if response is None else response
        return patch.multiple(loader, _AKSHARE_OK=True, ak=api, create=True), api

    def test_missing_provider_fails_without_creating_cache(self):
        with patch.object(loader, "_AKSHARE_OK", False):
            with self.assertRaisesRegex(loader.DataSourceError, "unavailable"):
                store.load("600519", "20230101", "20230103")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_provider_error_empty_and_bad_schema_fail_closed(self):
        for response, error in ((None, RuntimeError("offline")),
                                (pd.DataFrame(), None),
                                (pd.DataFrame({"日期": ["2023-01-02"]}), None)):
            with self.subTest(response=response, error=error):
                patcher, _ = self.market(response, error)
                with patcher, self.assertRaises(loader.DataSourceError):
                    store.load("600519", "20230101", "20230103")
                self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_synthetic_is_explicit_never_market_cached_and_honors_range(self):
        with patch.object(loader, "_AKSHARE_OK", False):
            demo = store.load("600519", "20230102", "20230103", source="synthetic")
            self.assertEqual(demo.attrs["data_source"], "synthetic")
            self.assertEqual(demo.date.tolist(), pd.bdate_range("2023-01-02", "2023-01-03").tolist())
            self.assertEqual(store.cached_symbols(), [])
            self.assertEqual(list(Path(self.tmp.name).iterdir()), [])
            with self.assertRaises(loader.DataSourceError):
                store.load("600519", "20230102", "20230103")
        with self.assertRaises(ValueError):
            store.load("600519", "20230107", "20230108", source="synthetic")

    def test_legacy_csv_is_ignored_then_replaced_by_verified_market_data(self):
        legacy = Path(store.cache_file("600519"))
        legacy.write_text("date,open,high,low,close,volume\n2023-01-02,999,999,999,999,1\n")
        self.assertIsNone(store.load_cached("600519"))
        patcher, api = self.market()
        with patcher:
            result = store.load("600519", "20230101", "20230103")
        self.assertEqual(result.attrs["data_source"], "akshare")
        self.assertEqual(result.close.tolist(), [10.5, 11.5])
        self.assertEqual(api.stock_zh_a_hist.call_count, 1)
        meta = json.loads(Path(self.tmp.name, "600519.json").read_text())
        self.assertEqual((meta["source"], meta["symbol"], meta["adjust"]),
                         ("akshare", "600519", "qfq"))
        self.assertEqual(store.cached_symbols(), ["600519"])

    def test_verified_cache_reused_and_expansion_refetched(self):
        patcher, api = self.market()
        with patcher:
            first = store.load("600519", "20230101", "20230103")
            again = store.load("600519", "20230102", "20230103")
            store.load("600519", "20221201", "20230103")
        self.assertEqual(first.close.tolist(), again.close.tolist())
        self.assertEqual(api.stock_zh_a_hist.call_count, 2)
        self.assertEqual(api.stock_zh_a_hist.call_args.kwargs["start_date"], "20221201")

    def test_adjustment_or_tampering_invalidates_cache(self):
        patcher, api = self.market()
        with patcher:
            store.update("600519", "20230101", "20230103")
            store.update("600519", "20230101", "20230103", adjust="hfq")
        self.assertEqual(api.stock_zh_a_hist.call_count, 2)
        self.assertIsNone(store.load_cached("600519", "qfq"))
        Path(store.cache_file("600519")).write_text("corrupt")
        self.assertIsNone(store.load_cached("600519", "hfq"))
        self.assertEqual(store.cached_symbols(), [])

    def test_pipeline_rejects_wrong_source_before_training_or_report(self):
        from quant.pipeline import runner
        frame = loader.get_daily("600519", "20230102", "20230104", source="synthetic")
        gateway = SimpleNamespace(log=[], remote=None, log_summary=lambda: {})
        with patch.object(runner, "REPORTS_DIR", self.tmp.name), \
             patch.object(runner, "get_gateway", return_value=gateway), \
             patch.object(runner, "load_data", return_value=frame), \
             patch.object(runner, "train_models") as train:
            with self.assertRaisesRegex(loader.DataSourceError, "expected akshare"):
                runner.run_pipeline(["600519"], end="20230104", source="market")
            train.assert_not_called()
            self.assertFalse(Path(self.tmp.name, "report.json").exists())

    def test_demo_seed_is_stable_across_processes(self):
        script = ('from quant.data.mock import generate_mock; '
                  'print(generate_mock("DEMO", "20230102", "20230103").close.tolist())')
        outputs = [subprocess.check_output([sys.executable, "-c", script], text=True)
                   for _ in range(2)]
        self.assertEqual(outputs[0], outputs[1])

    def test_pipeline_source_and_cli_are_explicit(self):
        from quant.pipeline import runner
        from run_pipeline import main
        frame = loader.get_daily("600519", "20230102", "20230104", source="synthetic")
        frame = frame.reset_index(drop=True)
        gw = SimpleNamespace(log=[], remote=object(), log_summary=lambda: {})
        with patch.object(runner, "REPORTS_DIR", self.tmp.name), \
             patch.object(runner, "get_gateway", return_value=gw), \
             patch.object(runner, "load_data", return_value=frame) as data, \
             patch.object(runner, "train_models", return_value={}), \
             patch.object(runner, "walk_forward_signals", return_value=pd.Series([0] * len(frame))), \
             patch.object(runner, "list_strategies", return_value=[]), \
             patch.object(runner.analysis, "summarize_report") as commentary:
            report, path = runner.run_pipeline(["600519"], end="20230104", source="synthetic")
            self.assertEqual(report["data_source"], "synthetic")
            self.assertEqual(report["symbols"]["600519"]["data_source"], "synthetic")
            self.assertEqual(json.loads(Path(path).read_text())["data_source"], "synthetic")
            self.assertEqual(data.call_args.kwargs["source"], "synthetic")
            commentary.assert_not_called()
        with patch("sys.argv", ["run_pipeline.py", "--symbols", "600519", "--synthetic"]), \
             patch("run_pipeline.run_pipeline", return_value=({"summary": []}, "demo.json")) as run:
            main()
            self.assertEqual(run.call_args.kwargs["source"], "synthetic")


if __name__ == "__main__":
    unittest.main()
