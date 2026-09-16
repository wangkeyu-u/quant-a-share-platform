"""OHLCV 特征与带可用日期的监督标签。

分类目标是次日方向,回归目标是 horizon 个观察交易日后的收益。
fwd5/fwd20 保留为诊断列。MACD 等特征仍含价格量纲,跨股票训练并未证明
尺度不变性。训练必须按 label_end 过滤已知结果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators.tech import boll, kdj, macd, ma, rsi

# 分类保持次日方向;回归周期由 horizon 决定。
CLS_TARGET = "target"
REG_TARGET = "target_return"
FEATURE_DATE = "feature_date"
LABEL_END = "label_end"

FEATURE_COLS = [
    # —— 收益率(多周期)——
    "ret", "ret5", "ret10", "ret20", "ret60",
    # —— 均线相对位置(多周期)——
    "ma5_20", "ma10_60", "ma20_60", "ma60_120", "ma5_60",
    # —— MACD ——
    "macd_dif", "macd_dea", "macd_hist",
    # —— RSI(双周期)——
    "rsi14", "rsi6",
    # —— 布林带 ——
    "boll_width", "pct_b", "close_to_mid",
    # —— 波动率(年化口径的滚动标准差)——
    "vol20", "vol60",
    # —— 成交量(量比 / 量变化)——
    "vol_ratio5", "vol_ratio20", "vol_chg",
    # —— KDJ ——
    "kdj_k", "kdj_d", "kdj_j",
    # —— 真实波幅(归一化)——
    "atr_ratio",
    # —— 距阶段高点的距离(动量/超买)——
    "dist_20h", "dist_60h",
]


def _tr(high, low, close):
    prev = close.shift(1)
    return pd.concat([
        (high - low),
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)


def build_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """返回特征表(含分类/回归标签)。索引与 df 对齐。"""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of observed sessions")
    dates = pd.to_datetime(df["date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("date must be non-null, unique and increasing within each symbol")
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)
    # 收益率
    out["ret"] = ret
    out["ret5"] = close.pct_change(5)
    out["ret10"] = close.pct_change(10)
    out["ret20"] = close.pct_change(20)
    out["ret60"] = close.pct_change(60)

    # 均线相对位置
    ma5, ma10, ma20, ma60, ma120 = (ma(close, w) for w in (5, 10, 20, 60, 120))
    out["ma5_20"] = ma5 / ma20 - 1
    out["ma10_60"] = ma10 / ma60 - 1
    out["ma20_60"] = ma20 / ma60 - 1
    out["ma60_120"] = ma60 / ma120 - 1
    out["ma5_60"] = ma5 / ma60 - 1

    # MACD
    dif, dea, hist = macd(close)
    out["macd_dif"] = dif
    out["macd_dea"] = dea
    out["macd_hist"] = hist

    # RSI
    out["rsi14"] = rsi(close, 14)
    out["rsi6"] = rsi(close, 6)

    # 布林带
    up, mid, lo = boll(close, 20, 2)
    out["boll_width"] = (up - lo) / mid
    out["pct_b"] = (close - lo) / (up - lo)
    out["close_to_mid"] = close / mid - 1

    # 波动率(滚动标准差,近似年化)
    out["vol20"] = ret.rolling(20).std()
    out["vol60"] = ret.rolling(60).std()

    # 成交量
    out["vol_ratio5"] = vol / vol.rolling(5).mean()
    out["vol_ratio20"] = vol / vol.rolling(20).mean()
    out["vol_chg"] = vol.pct_change()

    # KDJ
    k, d, j = kdj(close, high, low)
    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj_j"] = j

    # 真实波幅(归一化到收盘价)
    atr = _tr(high, low, close).rolling(14).mean()
    out["atr_ratio"] = atr / close

    # 距阶段高点距离
    out["dist_20h"] = close / close.rolling(20).max() - 1
    out["dist_60h"] = close / close.rolling(60).max() - 1

    # —— 标签 ——
    next_close = close.shift(-1)
    out[CLS_TARGET] = (next_close > close).astype(float).where(next_close.notna())
    out["fwd5"] = close.shift(-5) / close - 1                      # 5 日前向收益(回归)
    out["fwd20"] = close.shift(-20) / close - 1                    # 20 日前向收益(回归)
    out[REG_TARGET] = close.shift(-horizon) / close - 1
    out[FEATURE_DATE] = dates
    out[LABEL_END] = dates.shift(-horizon)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
