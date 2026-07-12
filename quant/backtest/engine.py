from __future__ import annotations

import pandas as pd

from quant.backtest.metrics import compute_metrics, trade_stats


class Backtest:
    """单标的、多头、事件驱动日频回测引擎。

    position 为 0/1 仓位序列(1=满仓)。逐根 K 线按目标仓位再平衡,
    买入/卖出均计手续费与滑点。支持仓位比例 stake(默认 1.0=满仓)。
    """

    def __init__(self, df: pd.DataFrame, initial_cash: float = 1_000_000.0,
                 commission: float = 0.0003, slippage: float = 0.0005,
                 stake: float = 1.0):
        self.df = df.reset_index(drop=True)
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stake = float(stake)
        self.trades: list = []
        self.equity: pd.Series | None = None

    def run(self, position: pd.Series) -> dict:
        df = self.df
        close = df["close"].astype(float).values
        n = len(df)
        pos = position.reindex(df.index).fillna(0).astype(float).values
        dates = df["date"].values

        cash = self.initial_cash
        shares = 0.0
        buy_fee = self.commission + self.slippage
        sell_fee = self.commission + self.slippage

        equity = []
        pending_buy_price = 0.0
        pending_buy_shares = 0.0
        avg_cost = 0.0

        for i in range(n):
            price = close[i]
            target = pos[i]
            equity_now = cash + shares * price
            target_value = target * self.stake * equity_now
            target_shares = target_value / price if price > 0 else 0.0
            delta = target_shares - shares

            if delta > 1e-9:
                # 按可用现金封顶,避免含费成本超过现金而无法建仓
                max_affordable = cash / (price * (1 + buy_fee)) if price > 0 else 0.0
                buy_shares = min(delta, max_affordable)
                cost = buy_shares * price * (1 + buy_fee)
                if buy_shares > 1e-9 and cost <= cash + 1e-6:
                    if shares + buy_shares > 0:
                        avg_cost = (avg_cost * shares + price * buy_shares) / (shares + buy_shares)
                    cash -= cost
                    shares += buy_shares
                    self.trades.append({
                        "date": pd.Timestamp(dates[i]),
                        "side": "BUY",
                        "price": round(price, 2),
                        "shares": round(buy_shares, 2),
                        "amount": round(cost, 2),
                        "pnl": None,
                    })
            elif delta < -1e-9:
                sell_shares = -delta
                proceeds = sell_shares * price * (1 - sell_fee)
                pnl = (price - avg_cost) * sell_shares if avg_cost > 0 else 0.0
                cash += proceeds
                shares -= sell_shares
                if shares < 1e-9:
                    shares = 0.0
                    avg_cost = 0.0
                self.trades.append({
                    "date": pd.Timestamp(dates[i]),
                    "side": "SELL",
                    "price": round(price, 2),
                    "shares": round(sell_shares, 2),
                    "amount": round(proceeds, 2),
                    "pnl": round(pnl, 2),
                })
            equity.append(cash + shares * price)

        self.equity = pd.Series(equity, index=df.index, name="equity")
        return self.result()

    def result(self) -> dict:
        df = self.df
        equity = self.equity
        benchmark = pd.Series(
            self.initial_cash * df["close"].values / df["close"].values[0],
            index=df.index, name="benchmark",
        )
        metrics = compute_metrics(equity)
        trades = trade_stats(self.trades)
        metrics.update(trades)
        return {
            "equity": equity,
            "benchmark": benchmark,
            "trades": self.trades,
            "metrics": metrics,
        }
