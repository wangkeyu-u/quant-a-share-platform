from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """策略基类。所有策略实现 generate_signals(df, **params) -> 0/1 仓位序列。"""

    name: str = "base"
    description: str = ""
    # 参数定义: 参数名 -> (默认值, 最小值, 最大值, 步长), 供 GUI 自动生成滑块
    params: dict = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """返回与 df 等长的仓位序列: 1=满仓多头, 0=空仓。"""
        raise NotImplementedError
