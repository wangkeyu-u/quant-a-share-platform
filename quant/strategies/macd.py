import pandas as pd

from quant.indicators.tech import macd
from quant.strategies.base import Strategy


class MACDStrategy(Strategy):
    name = "macd"
    description = "MACD 金叉买入,死叉卖出"
    params = {
        "fast": (12, 3, 60, 1),
        "slow": (26, 5, 250, 1),
        "signal": (9, 3, 60, 1),
    }

    def generate_signals(self, df: pd.DataFrame, fast: int = 12,
                         slow: int = 26, signal: int = 9) -> pd.Series:
        dif, dea, _ = macd(df["close"], fast, slow, signal)
        cross_up = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        cross_down = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        position = pd.Series(0, index=df.index)
        pos = 0
        for i in range(len(df)):
            if cross_up.iloc[i]:
                pos = 1
            elif cross_down.iloc[i]:
                pos = 0
            position.iloc[i] = pos
        return position
