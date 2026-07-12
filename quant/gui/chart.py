"""matplotlib 绘图辅助(K线 + 权益曲线),供 tkinter 嵌入使用。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 中文字体配置:优先 macOS 自带 PingFang SC,逐级回退到其它常见中文字体,
# 最后回退 DejaVu Sans(无 CJK 字形时会降级显示)。
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Arial Unicode MS", "Heiti SC", "STHeiti",
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题


def plot_kline(df: pd.DataFrame, signals: pd.Series = None,
               ma_lines: dict = None, max_bars: int = 250) -> plt.Figure:
    """绘制 K 线 + 成交量 + 买卖点 + 可选 MA 叠加。"""
    if len(df) > max_bars:
        df_d = df.iloc[-max_bars:].reset_index(drop=True)
        sig_d = signals.iloc[-max_bars:].reset_index(drop=True) if signals is not None else None
    else:
        df_d = df.reset_index(drop=True)
        sig_d = signals.reset_index(drop=True) if signals is not None else None

    fig = plt.Figure(figsize=(9, 5), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    x = np.arange(len(df_d))
    low = df_d["low"].values
    high = df_d["high"].values
    close = df_d["close"].values
    open_ = df_d["open"].values

    for i in range(len(df_d)):
        up = close[i] >= open_[i]
        color = "red" if up else "green"  # A股习惯:红涨绿跌
        ax1.plot([x[i], x[i]], [low[i], high[i]], color=color, lw=0.6)
        body = abs(close[i] - open_[i])
        ax1.add_patch(Rectangle((x[i] - 0.4, min(open_[i], close[i])), 0.8,
                                max(body, 1e-6), color=color))

    if ma_lines:
        for name, series in ma_lines.items():
            s = series.iloc[-len(df_d):].reset_index(drop=True)
            ax1.plot(x, s.values, label=name, lw=1.0)
        ax1.legend(loc="upper left", fontsize=8)

    ax1.set_title("K线 / 买卖点", fontsize=10)
    ax1.grid(True, alpha=0.25)

    if sig_d is not None:
        pos = sig_d.values.astype(int)
        prev = np.r_[0, pos[:-1]]
        entries = np.where((pos == 1) & (prev == 0))[0]
        exits = np.where((pos == 0) & (prev == 1))[0]
        if len(entries):
            ax1.scatter(entries, low[entries] * 0.992, marker="^", color="red",
                        s=60, zorder=5, label="买入")
        if len(exits):
            ax1.scatter(exits, high[exits] * 1.008, marker="v", color="green",
                        s=60, zorder=5, label="卖出")

    ax2.bar(x, df_d["volume"].values, color="gray", width=0.8)
    ax2.set_title("成交量", fontsize=10)
    ax2.grid(True, alpha=0.25)

    step = max(1, len(df_d) // 8)
    tick_idx = x[::step]
    tick_labels = [d.strftime("%Y-%m-%d") for d in df_d["date"].iloc[::step]]
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(tick_labels, rotation=30, fontsize=7)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=30, fontsize=7)
    fig.tight_layout()
    return fig


def plot_equity(equity: pd.Series, benchmark: pd.Series = None,
                title: str = "权益曲线") -> plt.Figure:
    """绘制策略权益曲线与基准对比。"""
    fig = plt.Figure(figsize=(9, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(range(len(equity)), equity.values, label="策略", color="blue", lw=1.3)
    if benchmark is not None:
        ax.plot(range(len(benchmark)), benchmark.values, label="基准(买入持有)",
                color="gray", lw=1.0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
