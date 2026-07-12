import pandas as pd

from quant.indicators.tech import boll
from quant.strategies.base import Strategy


class BollBreakoutStrategy(Strategy):
    name = "boll_breakout"
    description = "布林带突破:收盘价突破上轨买入,跌破中轨卖出"
    params = {
        "window": (20, 5, 120, 1),
        "ndays": (2.0, 1.0, 4.0, 0.5),
    }

    def generate_signals(self, df: pd.DataFrame, window: int = 20,
                         ndays: float = 2.0) -> pd.Series:
        upper, mid, _ = boll(df["close"], window, ndays)
        close = df["close"]
        enter = close > upper
        exit_ = close < mid
        position = pd.Series(0, index=df.index)
        pos = 0
        for i in range(len(df)):
            if enter.iloc[i]:
                pos = 1
            elif exit_.iloc[i]:
                pos = 0
            position.iloc[i] = pos
        return position
