from __future__ import annotations

import pandas as pd

from quant.backtest.metrics import compute_metrics, trade_stats
from quant.trading.paper import EXECUTION_PROTOCOL, PaperTrader


class Backtest:
    """多头日频回测：收盘信号在下一根观察行情开盘再平衡。

    position 按 df 的原始索引对齐，缺失信号视为零仓位。
    基准在第二根开盘按相同 stake、手续费和滑点买入并持有。
    """

    def __init__(self, df: pd.DataFrame, initial_cash: float = 1_000_000.0,
                 commission: float = 0.0003, slippage: float = 0.0005,
                 stake: float = 1.0):
        if df.empty or not df.index.is_unique:
            raise ValueError("bars must be non-empty with a unique index")
        self.df = df.copy()
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stake = float(stake)
        self.trades: list = []
        self.equity: pd.Series | None = None
        self.benchmark: pd.Series | None = None

    def run(self, position: pd.Series) -> dict:
        if not position.index.is_unique:
            raise ValueError("signal index must be unique")
        signals = position.reindex(self.df.index).fillna(0).astype(float)
        account = PaperTrader(self.initial_cash, self.commission, self.slippage, self.stake)
        benchmark = PaperTrader(self.initial_cash, self.commission, self.slippage, self.stake)
        bench_equity = []
        for i, (row, signal) in enumerate(zip(self.df.itertuples(index=False), signals)):
            account.step_bar(row.date, row.open, row.close, signal)
            # Buy once at the second observed open. Do not rebalance the benchmark.
            if i < 2:
                benchmark.step_bar(row.date, row.open, row.close, 1.0)
            bench_equity.append(benchmark.cash + benchmark.shares * float(row.close))

        self.trades = account.trades
        self.equity = pd.Series(account.equity_curve, index=self.df.index, name="equity")
        self.benchmark = pd.Series(bench_equity, index=self.df.index, name="benchmark")
        return self.result()

    def result(self) -> dict:
        if self.equity is None:
            raise RuntimeError("run must be called before result")
        metrics = compute_metrics(self.equity)
        metrics.update(trade_stats(self.trades))
        return {
            "execution_protocol": EXECUTION_PROTOCOL,
            "equity": self.equity,
            "benchmark": self.benchmark,
            "trades": self.trades,
            "metrics": metrics,
        }
