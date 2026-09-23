# Generated prices entered the market-data cache

Recorded 2026-09-23. This is a data-provenance repair, not a new market-performance result.

## Failure

The old `get_daily` generated deterministic prices when AkShare was unavailable, raised an error, or returned no rows. `store.update` wrote those prices to the same `data_cache/<code>.csv` used for actual AkShare prices. The pipeline read that cache and reported Sharpe/returns without recording the source. A later online run could therefore reuse synthetic bars as if they were market observations.

The pre-fix loader and store at commit `82acc23` were run in a temporary directory with AkShare unavailable: they returned 22 generated rows for `600519` in January 2023, wrote `600519.csv`, and wrote no source manifest. The frame had no `data_source` attribute. No existing user cache was altered for this reproduction.

## Decision and implementation

`market` is the default mode. Missing AkShare, provider exceptions, empty results and malformed OHLCV now raise `DataSourceError`; the pipeline does not write a new report on such a load failure. Existing report files are not retroactively removed. `synthetic` is an explicit GUI choice or CLI `--synthetic` option. Synthetic frames carry `data_source=synthetic`, are excluded from the market cache and produce a report with top-level and per-symbol `data_source=synthetic`.

A market cache is accepted only when its adjacent JSON manifest identifies AkShare, the requested symbol and adjustment, covers the requested date range, and matches the CSV's SHA-256 digest. A legacy CSV without a manifest or a mismatched/corrupted pair is ignored. When the requested range exceeds the verified cache coverage, the full requested interval is fetched again; this replaces the previous incremental behavior to avoid mixing source or adjustment segments. Successful market fetches create a new manifest. The cache hash checks accidental/stale content changes; it does not independently authenticate the third-party provider.

Synthetic runs skip remote AI report commentary; the GUI also skips remote market commentary for synthetic data. The bundled stock-name list can still fall back to static examples if its separate refresh fails. That list is only a symbol picker; it is not price data. The demo generator now uses a stable code hash and stays inside the requested date range.

## Reproduce and validate

```bash
python -m pip install numpy pandas scikit-learn joblib
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. AI_GATEWAY_BASE_URL= AI_GATEWAY_API_KEY= python tests/headless_test.py
```

The first command installs offline test dependencies. The 28 tests include nine new provenance cases with a fake AkShare provider: unavailable/failed/empty/malformed response; synthetic isolation; untrusted legacy cache; verified reuse and range refetch; adjustment and hash invalidation; deterministic demo seed across processes; explicit CLI/report source, and rejection of a mismatched source before training. The pipeline integration test mocks training, source loading and strategy discovery but uses real report writing. The final command ran the existing full headless smoke with explicit synthetic data and remote AI disabled: all checks passed, including single/pooled training, walk-forward, backtest/paper parity, and report provenance. It took about a minute locally. Its simulated metrics are not market results.

Existing cached CSV files without manifests are intentionally re-fetched in market mode. Existing reports cannot be retroactively classified by this change and should not be presented as real-market results. Live AkShare behavior and GUI interaction were not exercised in this validation.

## Remaining limits

A source record establishes which code path supplied the bars, not their point-in-time accuracy. AkShare's adjusted history can be revised; exchange fill constraints, survivorship, symbol selection and outer strategy-selection bias remain separate research risks. Synthetic prices are useful for deterministic integration checks only.
