"""显式合成演示行情：按代码和日期区间生成可重复的 OHLCV。"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def generate_mock(code: str, start: str, end: str) -> pd.DataFrame:
    """根据 code 生成确定性模拟日线(同代码结果稳定)。

    采用分段漂移(牛/熊/震荡交替)+ 高斯噪声,使均线类策略能产生交叉信号。
    """
    start_d = pd.to_datetime(start)
    end_d = pd.to_datetime(end)
    days = pd.bdate_range(start_d, end_d)  # 仅工作日
    n = len(days)
    if n == 0:
        raise ValueError("synthetic date range has no business days")

    seed = int.from_bytes(hashlib.sha256(str(code).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    vol = rng.uniform(0.012, 0.022)
    # 分段趋势,产生均线交叉
    n_seg = min(n, max(1, n // 40))
    seg_len = n // n_seg
    rets = np.empty(n)
    idx = 0
    for s in range(n_seg):
        seg_drift = rng.uniform(-0.0018, 0.0018)  # 含正负漂移
        length = seg_len if s < n_seg - 1 else n - idx
        rets[idx:idx + length] = rng.normal(seg_drift, vol, length)
        idx += length

    close = 20.0 * np.exp(np.cumsum(rets))
    close = np.clip(close, 1.0, None)

    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.integers(1_000_000, 60_000_000, n).astype(float)

    df = pd.DataFrame({
        "date": days,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })
    return df
