"""规则策略参数寻优:网格搜索 + 回测,按夏普排序。"""
from __future__ import annotations

import itertools

from quant.backtest.engine import Backtest


def small_grid(strategy, points=(0, 1, 2)):
    """由策略默认参数生成小网格:[min, default, max](去重)。"""
    grid = {}
    for name, (default, lo, hi, step) in strategy.params.items():
        vals = sorted({lo, default, hi})
        grid[name] = vals
    return grid


def optimize(strategy, df: object, param_grid: dict | None = None,
            initial_cash: float = 1_000_000.0, commission: float = 0.0003,
            slippage: float = 0.0005, stake: float = 1.0, top_n: int = 5) -> list:
    """对所有参数组合跑回测,返回按夏普降序的结果列表。"""
    if param_grid is None:
        param_grid = small_grid(strategy)
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        sig = strategy.generate_signals(df, **params)
        res = Backtest(df, initial_cash, commission, slippage, stake).run(sig)
        m = res["metrics"]
        results.append({
            "params": params,
            "总收益率": m["总收益率"],
            "年化收益": m["年化收益"],
            "最大回撤": m["最大回撤"],
            "夏普比率": m["夏普比率"],
            "交易次数": m["交易次数"],
        })
    results.sort(key=lambda r: r["夏普比率"], reverse=True)
    # 过滤掉 0 交易的退化组合:平直净值夏普=0,会排在亏损策略前面,
    # 导致"什么都不做"被误判为最优。仅当所有组合都无交易时才回退保留。
    traded = [r for r in results if r["交易次数"] > 0]
    return (traded if traded else results)[:top_n]
