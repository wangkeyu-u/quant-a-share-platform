"""只缓存可校验来源的真实行情；合成演示不写入真实缓存。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from quant.data.loader import DataSourceError, get_daily

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = str(_ROOT / "data_cache")
CACHE_SCHEMA = 1


def cache_file(symbol: str) -> str:
    if not re.fullmatch(r"\d{6}", str(symbol)):
        raise ValueError("market symbol must be six digits")
    return str(Path(CACHE_DIR) / f"{symbol}.csv")


def _manifest_file(symbol: str) -> Path:
    return Path(cache_file(symbol)).with_suffix(".json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_verified(symbol: str, adjust: str = "qfq") -> tuple[pd.DataFrame, dict] | None:
    csv_path, manifest_path = Path(cache_file(symbol)), _manifest_file(symbol)
    if not csv_path.is_file() or not manifest_path.is_file():
        return None
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (meta.get("schema") != CACHE_SCHEMA or meta.get("source") != "akshare"
                or meta.get("symbol") != symbol or meta.get("adjust") != adjust
                or meta.get("sha256") != _digest(csv_path)):
            return None
        frame = pd.read_csv(csv_path, parse_dates=["date"])
        if frame.empty or frame["date"].isna().any() or frame["date"].duplicated().any():
            return None
        frame.attrs.update(data_source="akshare", adjust=adjust)
        return frame, meta
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return None


def load_cached(symbol: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """只有带匹配来源、调整方式和内容哈希的缓存可读取。"""
    verified = _read_verified(symbol, adjust)
    return verified[0] if verified else None


def _filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    result = df.loc[mask].reset_index(drop=True)
    result.attrs = df.attrs.copy()
    return result


def _write_verified(symbol: str, adjust: str, start: str, end: str, frame: pd.DataFrame):
    directory = Path(CACHE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = Path(cache_file(symbol)), _manifest_file(symbol)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix="bars-", suffix=".csv",
                                     delete=False, encoding="utf-8") as tmp:
        tmp_csv = Path(tmp.name)
        frame.to_csv(tmp, index=False)
    try:
        meta = {
            "schema": CACHE_SCHEMA, "source": "akshare", "symbol": symbol,
            "adjust": adjust, "requested_start": start, "requested_end": end,
            "sha256": _digest(tmp_csv),
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix="source-", suffix=".json",
                                         delete=False, encoding="utf-8") as tmp:
            tmp_json = Path(tmp.name)
            json.dump(meta, tmp, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp_csv, csv_path)
            os.replace(tmp_json, manifest_path)
        finally:
            tmp_json.unlink(missing_ok=True)
    finally:
        tmp_csv.unlink(missing_ok=True)


def update(symbol: str, start: str, end: str, adjust: str = "qfq",
           force: bool = False, source: str = "market") -> pd.DataFrame:
    """读取覆盖指定区间的真实缓存；范围扩大时重新抓取完整区间。"""
    if source not in ("market", "synthetic"):
        raise ValueError("source must be 'market' or 'synthetic'")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must be on or before end")
    if source == "synthetic":
        return get_daily(symbol, start, end, adjust, source="synthetic")
    symbol = str(symbol).zfill(6)
    cache_file(symbol)  # validate before any provider or filesystem call
    if not force:
        verified = _read_verified(symbol, adjust)
        if verified:
            frame, meta = verified
            if (pd.Timestamp(meta["requested_start"]) <= pd.Timestamp(start)
                    and pd.Timestamp(meta["requested_end"]) >= pd.Timestamp(end)):
                result = _filter(frame, start, end)
                if not result.empty:
                    return result
                raise DataSourceError(f"Verified cache has no bars for {symbol} in {start}..{end}")
    frame = get_daily(symbol, start, end, adjust, source="market")
    _write_verified(symbol, adjust, start, end, frame)
    return _filter(frame, start, end)


def load(symbol: str, start: str = "20200101", end: str | None = None,
         source: str = "market") -> pd.DataFrame:
    """按来源读取日线。market 默认失败即报错；synthetic 永不写入真实缓存。"""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")
    return update(symbol, start, end, source=source)


def cached_symbols() -> list[str]:
    directory = Path(CACHE_DIR)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json")
                  if re.fullmatch(r"\d{6}", p.stem) and _read_verified(p.stem))
