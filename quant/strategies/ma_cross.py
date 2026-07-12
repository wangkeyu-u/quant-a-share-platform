import pandas as pd

from quant.indicators.tech import ma
from quant.strategies.base import Strategy


class MACrossStrategy(Strategy):
    name = "ma_cross"
    description = "双均线交叉:短均线上穿长均线买入,下穿卖出"
    params = {
        "fast": (5, 3, 60, 1),
        "slow": (20, 5, 250, 1),
    }

    def generate_signals(self, df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
        close = df["close"]
        ma_fast = ma(close, fast)
        ma_slow = ma(close, slow)
        cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
        position = pd.Series(0, index=df.index)
        pos = 0
        for i in range(len(df)):
            if cross_up.iloc[i]:
                pos = 1
            elif cross_down.iloc[i]:
                pos = 0
            position.iloc[i] = pos
        return position
