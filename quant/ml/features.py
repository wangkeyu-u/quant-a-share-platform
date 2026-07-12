"""特征工程:由 OHLCV 构造大量量价/波动率/成交量/技术指标特征,并生成多周期标签。

设计要点(为"用大量数据把模型训强"服务):
- 特征全部是「比率 / 归一化」形态(与价格量纲无关),因此多个股票的特征可以直接
  拼接做联合(pooled)训练,不会因为股价绝对值不同而产生偏差。
- 标签同时给出:次日方向(分类)、5日/20日前向收益(回归)。前者用于概率门控,
  后者用于判断"该不该出手"以及预期盈亏空间。
- target 用 shift(-h) 生成,是监督标签;walk-forward 预测时只使用当日特征,不会
  引入未来信息。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators.tech import boll, kdj, macd, ma, rsi

# 分类标签列(次日方向)与默认回归标签列(5 日前向收益)
CLS_TARGET = "target"
REG_TARGET = "fwd5"

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """返回特征表(含分类/回归标签)。索引与 df 对齐。"""
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
    out[CLS_TARGET] = (close.shift(-1) > close).astype(int)        # 次日涨跌(分类)
    out["fwd5"] = close.shift(-5) / close - 1                      # 5 日前向收益(回归)
    out["fwd20"] = close.shift(-20) / close - 1                    # 20 日前向收益(回归)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
