"""模拟交易(虚拟账户)。与回测共用同一套再平衡逻辑,但以账户视角记录持仓。"""
from __future__ import annotations

import pandas as pd


class PaperTrader:
    """虚拟资金账户:逐日按信号调仓,跟踪现金/持仓/浮动盈亏。"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 commission: float = 0.0003, slippage: float = 0.0005,
                 stake: float = 1.0):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stake = float(stake)
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0.0
        self.avg_cost = 0.0
        self.trades: list = []
        self.equity_curve: list = []
        self.dates: list = []
        self.position = 0.0

    def step(self, date, price: float, target: float) -> dict:
        """推进一个交易日。target: 目标仓位(0/1)。"""
        fee = self.commission + self.slippage
        equity_now = self.cash + self.shares * price
        target_value = target * self.stake * equity_now
        target_shares = target_value / price if price > 0 else 0.0
        delta = target_shares - self.shares

        if delta > 1e-9:
            max_affordable = self.cash / (price * (1 + fee)) if price > 0 else 0.0
            buy_shares = min(delta, max_affordable)
            cost = buy_shares * price * (1 + fee)
            if buy_shares > 1e-9 and cost <= self.cash + 1e-6:
                if self.shares + buy_shares > 0:
                    self.avg_cost = (self.avg_cost * self.shares + price * buy_shares) / (self.shares + buy_shares)
                self.cash -= cost
                self.shares += buy_shares
                self.trades.append({"date": pd.Timestamp(date), "side": "BUY",
                                    "price": round(price, 2), "shares": round(buy_shares, 2),
                                    "amount": round(cost, 2), "pnl": None})
        elif delta < -1e-9:
            sell_shares = -delta
            proceeds = sell_shares * price * (1 - fee)
            pnl = (price - self.avg_cost) * sell_shares if self.avg_cost > 0 else 0.0
            self.cash += proceeds
            self.shares -= sell_shares
            if self.shares < 1e-9:
                self.shares, self.avg_cost = 0.0, 0.0
            self.trades.append({"date": pd.Timestamp(date), "side": "SELL",
                                "price": round(price, 2), "shares": round(sell_shares, 2),
                                "amount": round(proceeds, 2), "pnl": round(pnl, 2)})

        self.position = target
        equity = self.cash + self.shares * price
        self.dates.append(pd.Timestamp(date))
        self.equity_curve.append(equity)
        return {
            "date": pd.Timestamp(date),
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "equity": equity,
            "avg_cost": self.avg_cost,
            "float_pnl": (price - self.avg_cost) * self.shares,
        }

    def summary(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "shares": self.shares,
            "trades": self.trades,
            "equity": pd.Series(self.equity_curve, index=pd.Index(self.dates)),
        }
