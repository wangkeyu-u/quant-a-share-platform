"""日线行情：真实数据失败时显式报错，合成演示需显式选择。"""
from __future__ import annotations

import pandas as pd

try:
    import akshare as ak
    _AKSHARE_OK = True
except ImportError:  # pragma: no cover - AkShare 未安装时由 market 模式报错
    _AKSHARE_OK = False

from quant.data.mock import generate_mock

_SAMPLE_STOCKS = [
    ("600519", "贵州茅台"),
    ("000001", "平安银行"),
    ("600036", "招商银行"),
    ("000858", "五粮液"),
    ("601318", "中国平安"),
    ("600276", "恒瑞医药"),
    ("000333", "美的集团"),
    ("601888", "中国中免"),
    ("300750", "宁德时代"),
    ("002594", "比亚迪"),
]


def akshare_available() -> bool:
    return _AKSHARE_OK


def sample_stocks() -> pd.DataFrame:
    """内置样例股票列表(离线可用)。"""
    return pd.DataFrame(_SAMPLE_STOCKS, columns=["code", "name"])


def get_stock_list() -> pd.DataFrame:
    """返回 A 股代码-名称列表 (columns: code, name)。失败则返回内置样例。"""
    if _AKSHARE_OK:
        try:
            df = ak.stock_info_a_code_name()  # 列: 代码, 名称
            df = df.rename(columns={"代码": "code", "名称": "name"})
            df["code"] = df["code"].astype(str).str.zfill(6)
            return df[["code", "name"]]
        except Exception:
            pass
    return pd.DataFrame(_SAMPLE_STOCKS, columns=["code", "name"])


class DataSourceError(RuntimeError):
    """要求的行情来源不可用或没有返回数据。"""


def get_daily(code: str, start: str, end: str, adjust: str = "qfq",
              source: str = "market") -> pd.DataFrame:
    """获取日线行情。market=AkShare；synthetic=显式合成演示。"""
    if source not in ("market", "synthetic"):
        raise ValueError("source must be 'market' or 'synthetic'")
    code = str(code).zfill(6)
    if source == "synthetic":
        df = generate_mock(code, start, end)
        df.attrs["data_source"] = "synthetic"
        return df
    if not _AKSHARE_OK:
        raise DataSourceError("AkShare is unavailable; install it or explicitly select synthetic demo mode")
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust=adjust,
        )
    except Exception as exc:
        raise DataSourceError(f"AkShare request failed for {code}") from exc
    if df is None or df.empty:
        raise DataSourceError(f"AkShare returned no bars for {code} in {start}..{end}")
    required = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    }
    if not required.keys() <= set(df.columns):
        raise DataSourceError(f"AkShare response for {code} is missing OHLCV columns")
    try:
        df = df.rename(columns=required)
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="raise")
        if (df["date"].isna().any() or df["date"].duplicated().any()
                or df[["open", "high", "low", "close", "volume"]].isna().any().any()
                or (df[["open", "high", "low", "close"]] <= 0).any().any()
                or (df["volume"] < 0).any()):
            raise ValueError("invalid date or OHLCV value")
        df = df.sort_values("date").reset_index(drop=True)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)
        if df.empty:
            raise ValueError("no bars inside requested range")
    except (ValueError, TypeError, KeyError) as exc:
        raise DataSourceError(f"AkShare response for {code} has invalid OHLCV values") from exc
    df.attrs["data_source"] = "akshare"
    df.attrs["adjust"] = adjust
    return df
