import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def compute_metrics(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """由权益曲线计算绩效指标。"""
    equity = equity.astype(float)
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    start, end = equity.iloc[0], equity.iloc[-1]
    total_return = end / start - 1
    years = len(equity) / periods_per_year
    annual_return = (end / start) ** (1 / years) - 1 if years > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_drawdown = float(drawdown.min())

    rf_daily = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = (rets.mean() - rf_daily) / rets.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "起始资金": float(start),
        "结束资金": float(end),
        "总收益率": float(total_return),
        "年化收益": float(annual_return),
        "最大回撤": float(max_drawdown),
        "夏普比率": float(sharpe),
        "卡玛比率": float(calmar),
        "交易天数": int(len(equity)),
    }


def trade_stats(trades: list) -> dict:
    """由成交记录统计:交易次数、胜率、盈亏比等。"""
    if not trades:
        return {"交易次数": 0, "胜率": 0.0, "平均盈利": 0.0, "平均亏损": 0.0}
    closed = [t for t in trades if t.get("pnl") is not None]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    return {
        "交易次数": len(trades),
        "平仓次数": len(closed),
        "胜率": float(win_rate),
        "平均盈利": avg_win,
        "平均亏损": avg_loss,
    }
