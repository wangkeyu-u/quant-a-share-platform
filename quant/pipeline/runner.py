"""全流程编排:数据更新 → 特征 → ML训练(双模型/超参寻优) → ML回测 →
规则寻优 → (可选)跨股票联合训练 → 报告(含与买入持有基准的对比)。

新增(数据饥渴版):
- ML 训练改为双模型 + 时序交叉验证超参寻优;
- 支持 --pooled 跨股票联合训练(用全市场数据);
- 报告里给出 ML 策略的样本外夏普,并与同标的「买入持有」基准对比,
  如实反映策略有没有跑赢大盘,不粉饰。
"""
from __future__ import annotations

import datetime
import json
import os

import pandas as pd

from quant.ai.gateway import get_gateway
from quant.ai import analysis
from quant.backtest.engine import Backtest
from quant.backtest.metrics import compute_metrics
from quant.backtest.optimize import optimize
from quant.data.store import load as load_data
from quant.ml.trainer import train_models, train_pooled, walk_forward_signals, walk_forward_pooled
from quant.strategies.registry import get_strategy, list_strategies

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(_ROOT, "reports")


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _bench_metrics(df: pd.DataFrame, initial_cash: float) -> dict:
    """买入持有基准绩效(用收盘价买入并持有)。"""
    close = df["close"].values
    equity = initial_cash * (close / close[0])
    return compute_metrics(pd.Series(equity, index=df.index))


def run_pipeline(symbols: list, start: str = "20100101",
                 end: str | None = None, top_n: int = 3,
                 pooled: bool = False) -> tuple:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")

    # 统一 AI 网关:清零本次运行的调用日志,并记下远程是否可用
    gw = get_gateway()
    gw.log.clear()
    remote_configured = gw.remote is not None

    # 先一次性把所有标的行情拉到内存,供 pooled 模式复用
    raw = {sym: load_data(sym, start, end) for sym in symbols}
    raw = {s: d for s, d in raw.items() if d is not None and not d.empty}
    if not raw:
        raise RuntimeError("没有任何标的能加载到行情(检查代码/网络)")

    pooled_model_path = None
    if pooled:
        print(f"[pipeline] 跨股票联合训练(使用 {len(raw)} 只标的的全部数据) ...")
        pm = train_pooled([(s, d) for s, d in raw.items()])
        pooled_model_path = pm["model_path"]
        print(f"           pooled 样本外准确率={pm['test_accuracy']}  AUC={pm['test_auc']}  R2={pm['test_r2']}")

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "pooled" if pooled else "per_symbol",
        "start": start, "end": end,
        "symbols": {}, "summary": [],
    }

    for sym in raw:
        df = raw[sym]
        print(f"[pipeline] 处理 {sym} ...")
        sym_rep: dict = {"bars": int(len(df))}

        # 1) ML 训练 + 保存模型(双模型 / 超参寻优)
        ml_metrics = train_models(df, sym)
        # 2) ML 信号回测(walk-forward,置信度门控)
        if pooled:
            sig_ml = walk_forward_pooled(df, [(s, d) for s, d in raw.items()],
                                         pooled_model_path)
        else:
            sig_ml = walk_forward_signals(df)
        res_ml = Backtest(df).run(sig_ml)
        bench = _bench_metrics(df, 1_000_000)

        sym_rep["ml"] = {
            "train": _round(ml_metrics),
            "backtest": _round({k: v for k, v in res_ml["metrics"].items()}),
            "benchmark": _round(bench),
        }

        # 3) 规则策略参数寻优
        rule_opt = {}
        for sname, _ in list_strategies():
            if sname == "ml_direction":
                continue
            strat = get_strategy(sname)
            rule_opt[sname] = _round(optimize(strat, df, top_n=top_n))
        sym_rep["rule_optimize"] = rule_opt

        report["symbols"][sym] = sym_rep
        report["summary"].append({
            "symbol": sym,
            "ml_sharpe": res_ml["metrics"]["夏普比率"],
            "ml_return": res_ml["metrics"]["总收益率"],
            "ml_trades": res_ml["metrics"]["交易次数"],
            "bench_sharpe": bench["夏普比率"],
            "bench_return": bench["总收益率"],
            "beats_benchmark": res_ml["metrics"]["夏普比率"] > bench["夏普比率"],
        })

    # 远程 AI 分析(可选):远程网关未配置或请求失败都优雅跳过,不影响主流程
    report["ai_commentary"] = None
    if remote_configured:
        try:
            print("[pipeline] 调用 AI 网关生成报告点评 ...")
            report["ai_commentary"] = analysis.summarize_report(report)
        except Exception as e:  # 网络/鉴权/超时等,均降级跳过
            report["ai_commentary"] = f"(远程 AI 网关调用失败,已跳过: {type(e).__name__})"

    report["ai_gateway"] = {
        **gw.log_summary(),
        "remote_configured": remote_configured,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return report, path
