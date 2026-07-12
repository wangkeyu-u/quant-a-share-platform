"""行情数据加载:优先 akshare 真实 A 股数据,失败回退确定性模拟数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import akshare as ak
    _AKSHARE_OK = True
except Exception:  # pragma: no cover - akshare 未安装时降级
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


def get_daily(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """获取日线行情。

    返回列: date, open, high, low, close, volume (按日期升序)。
    code 为 6 位 A 股代码。优先 akshare,失败回退模拟数据。
    """
    code = str(code).zfill(6)
    if _AKSHARE_OK:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as exc:  # pragma: no cover
            print(f"[loader] akshare 拉取失败({code}),使用模拟数据: {exc}")
    return generate_mock(code, start, end)
