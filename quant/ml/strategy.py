"""ML 策略:用已到期标签滚动训练分类器和回归器,按预测阈值生成信号。"""
from __future__ import annotations

import pandas as pd

from quant.ml.trainer import walk_forward_signals
from quant.strategies.base import Strategy


class MLStrategy(Strategy):
    name = "ml_direction"
    description = "机器学习:GBM 双模型(次日方向+前向收益),按标签可用日期滚动训练"
    params = {
        "horizon": (5, 5, 20, 5),
        "lookback": (60, 30, 250, 10),
        "window": (750, 250, 1500, 250),
        "retrain_every": (20, 10, 60, 10),
        "prob_threshold": (0.55, 0.5, 0.7, 0.05),
        "return_threshold": (0.0, -0.01, 0.03, 0.01),
    }

    def generate_signals(self, df: pd.DataFrame, horizon: int = 5, lookback: int = 60,
                         window: int = 750, retrain_every: int = 20,
                         prob_threshold: float = 0.55,
                         return_threshold: float = 0.0) -> pd.Series:
        return walk_forward_signals(
            df, horizon=horizon, lookback=lookback, window=window,
            retrain_every=retrain_every, prob_threshold=prob_threshold,
            return_threshold=return_threshold)
