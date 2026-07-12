"""本地行情缓存:增量更新 + 离线回退,支撑全程自动化数据管道。"""
from __future__ import annotations

import os

import pandas as pd

from quant.data.loader import get_daily

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_ROOT, "data_cache")


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_file(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}.csv")


def load_cached(symbol: str) -> pd.DataFrame | None:
    p = cache_file(symbol)
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["date"])
    return None


def _filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def update(symbol: str, start: str, end: str, adjust: str = "qfq",
           force: bool = False) -> pd.DataFrame:
    """增量更新某标的行情到本地缓存,返回 [start, end] 区间数据。

    - 已有缓存且覆盖 end:直接返回区间切片。
    - 否则仅补抓 last_date 之后的交易日,合并去重后落盘。
    - akshare 失败时 get_daily 自动回退模拟数据,保证管道不中断。
    """
    _ensure_dir()
    cached = None if force else load_cached(symbol)
    if cached is not None and not cached.empty:
        last = cached["date"].max()
        if pd.Timestamp(end) <= last:
            return _filter(cached, start, end)
        nxt = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        new = get_daily(symbol, nxt, end, adjust)
        if new is not None and not new.empty:
            merged = (pd.concat([cached, new])
                      .drop_duplicates(subset=["date"])
                      .sort_values("date")
                      .reset_index(drop=True))
            merged.to_csv(cache_file(symbol), index=False)
            return _filter(merged, start, end)
        return _filter(cached, start, end)

    df = get_daily(symbol, start, end, adjust)
    if df is not None and not df.empty:
        df.to_csv(cache_file(symbol), index=False)
    return df


def load(symbol: str, start: str = "20200101", end: str | None = None) -> pd.DataFrame:
    """读取(按需增量更新)某标的行情。"""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")
    return update(symbol, start, end)


def cached_symbols() -> list:
    if not os.path.isdir(CACHE_DIR):
        return []
    return [f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
