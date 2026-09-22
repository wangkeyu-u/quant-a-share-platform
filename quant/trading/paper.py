"""模拟交易(虚拟账户)。与回测共用同一套再平衡逻辑,但以账户视角记录持仓。"""
from __future__ import annotations

import math

import pandas as pd

EXECUTION_PROTOCOL = "close-signal-next-observed-open-v1"


class PaperTrader:
    """虚拟资金账户:逐日按信号调仓,跟踪现金/持仓/浮动盈亏。"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 commission: float = 0.0003, slippage: float = 0.0005,
                 stake: float = 1.0):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stake = float(stake)
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        fees = (self.commission, self.slippage)
        if any(not math.isfinite(x) or x < 0 for x in fees) or sum(fees) >= 1:
            raise ValueError("fees must be non-negative and total less than 1")
        if not math.isfinite(self.stake) or not 0 <= self.stake <= 1:
            raise ValueError("stake must be between 0 and 1")
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0.0
        self.avg_cost = 0.0
        self.trades: list = []
        self.equity_curve: list = []
        self.dates: list = []
        self.position = 0.0
        self.pending_signal = 0.0
        self.signal_date = None

    def step_bar(self, date, open_price: float, close_price: float, signal: float) -> dict:
        """执行上一根收盘信号，按本根收盘估值，再保存本根信号。

        首根保持现金；末根信号留待下一根行情，不虚构成交。
        date 必须严格递增。signal 是 [0, 1] 的目标仓位。
        """
        date = pd.Timestamp(date)
        open_price, close_price, signal = map(float, (open_price, close_price, signal))
        if pd.isna(date) or (self.dates and date <= self.dates[-1]):
            raise ValueError("bar dates must be non-null and strictly increasing")
        if any(not math.isfinite(p) or p <= 0 for p in (open_price, close_price)):
            raise ValueError("open and close must be finite positive prices")
        if not math.isfinite(signal) or not 0 <= signal <= 1:
            raise ValueError("signal must be finite and between 0 and 1")

        self._rebalance(date, open_price, self.pending_signal)
        self.position = self.pending_signal
        self.pending_signal, self.signal_date = signal, date
        equity = self.cash + self.shares * close_price
        self.dates.append(date)
        self.equity_curve.append(equity)
        return {
            "date": date, "cash": self.cash, "shares": self.shares,
            "position": self.position, "equity": equity,
            "avg_cost": self.avg_cost,
            "float_pnl": (close_price - self.avg_cost) * self.shares,
        }

    def _rebalance(self, date, price: float, target: float):
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
                self.trades.append({"date": pd.Timestamp(date), "signal_date": self.signal_date, "side": "BUY",
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
            self.trades.append({"date": pd.Timestamp(date), "signal_date": self.signal_date, "side": "SELL",
                                "price": round(price, 2), "shares": round(sell_shares, 2),
                                "amount": round(proceeds, 2), "pnl": round(pnl, 2)})

    def summary(self) -> dict:
        return {
            "execution_protocol": EXECUTION_PROTOCOL,
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "shares": self.shares,
            "trades": self.trades,
            "equity": pd.Series(self.equity_curve, index=pd.Index(self.dates)),
        }
