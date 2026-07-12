"""无界面核心链路测试:数据->指标->策略->回测->模拟交易。"""
import sys
import numpy as np
import pandas as pd

from quant.data.mock import generate_mock
from quant.data.store import load as store_load, cached_symbols
from quant.indicators.tech import ma, macd, rsi, boll, kdj
from quant.strategies.registry import get_strategy, list_strategies
from quant.backtest.engine import Backtest
from quant.backtest.optimize import optimize
from quant.ml.features import FEATURE_COLS, build_features
from quant.ml.trainer import train_models, walk_forward_signals, train_pooled, walk_forward_pooled
from quant.pipeline.runner import run_pipeline
from quant.trading.paper import PaperTrader

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(f"[{'OK' if cond else 'FAIL'}] {name}")


def main():
    df = generate_mock("600519", "2023-01-01", "2024-12-31")
    check("行情列完整", set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns))
    check("收盘价无 NaN", not df["close"].isna().any())
    check("行情行数>200", len(df) > 200)

    # 指标
    m = ma(df["close"], 20)
    dif, dea, hist = macd(df["close"])
    r = rsi(df["close"])
    up, mid, lo = boll(df["close"])
    k, d, j = kdj(df["close"], df["high"], df["low"])
    check("指标长度一致", len(m) == len(df) and len(dif) == len(df) and len(r) == len(df))
    check("指标非全 NaN", m.notna().sum() > 100 and r.notna().sum() > 100)
    check("布林带顺序 upper>lower", (up.dropna() > lo.dropna()).all())

    # 策略
    for sname, _ in list_strategies():
        strat = get_strategy(sname)
        params = {k: v[0] for k, v in strat.params.items()}
        sig = strat.generate_signals(df, **params)
        check(f"策略 {sname} 信号长度", len(sig) == len(df))
        check(f"策略 {sname} 取值∈{{0,1}}", set(np.unique(sig.dropna())).issubset({0, 1}))

    # 回测
    strat = get_strategy("ma_cross")
    sig = strat.generate_signals(df, fast=5, slow=20)
    bt = Backtest(df, initial_cash=1_000_000, commission=0.0003, slippage=0.0005, stake=1.0)
    res = bt.run(sig)
    eq = res["equity"]
    check("回测权益长度一致", len(eq) == len(df))
    check("回测权益有限值", np.isfinite(eq).all())
    check("回测指标含夏普", "夏普比率" in res["metrics"])
    check("回测指标含最大回撤", "最大回撤" in res["metrics"])
    print("  回测绩效:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in res["metrics"].items()})

    # 基准
    bench = res["benchmark"]
    check("基准长度一致", len(bench) == len(df))

    # 模拟交易(同样信号)
    pt = PaperTrader(initial_cash=1_000_000, commission=0.0003, slippage=0.0005, stake=1.0)
    for i in range(len(df)):
        pt.step(df["date"].iloc[i], df["close"].iloc[i], float(sig.iloc[i]))
    summ = pt.summary()
    check("模拟交易权益长度一致", len(summ["equity"]) == len(df))
    check("模拟交易末值≈回测末值", abs(summ["equity"].iloc[-1] - eq.iloc[-1]) < 1e-6)
    print("  模拟交易末值:", round(summ["equity"].iloc[-1], 2),
          "成交笔数:", len(summ["trades"]))

    # 手续费影响:无费应优于有费
    bt0 = Backtest(df, commission=0.0, slippage=0.0, stake=1.0)
    res0 = bt0.run(sig)
    print("  调试 有费末值:", round(eq.iloc[-1], 4), " 零费末值:", round(res0["equity"].iloc[-1], 4),
          " 有费成交:", len(res["trades"]), " 零费成交:", len(res0["trades"]))
    check("零费率权益>=有费率", res0["equity"].iloc[-1] >= eq.iloc[-1])

    # 特征工程
    feats = build_features(df)
    check("特征含全部 FEATURE_COLS", set(FEATURE_COLS).issubset(feats.columns))
    check("特征含 target", "target" in feats.columns)
    check("target 取值∈{0,1}", set(np.unique(feats["target"].dropna())).issubset({0, 1}))

    # ML 训练(双模型 + 时序 CV 超参寻优,保存模型)
    mlm = train_models(df, "TEST600519")
    check("ML 训练返回样本外准确率", "test_accuracy" in mlm and 0 <= mlm["test_accuracy"] <= 1)
    check("ML 训练返回 AUC", "test_auc" in mlm)
    check("ML 训练返回 R2", "test_r2" in mlm)
    check("ML 模型文件已保存", __import__("os").path.exists(mlm["model_path"]))

    # walk-forward 信号(无前视 + 置信度门控)
    wf = walk_forward_signals(df)
    check("wf 信号长度一致", len(wf) == len(df))
    check("wf 信号取值∈{0,1}", set(np.unique(wf.dropna())).issubset({0, 1}))

    # 跨股票联合(pooled)训练 + walk-forward(用全市场数据,无前视)
    df2 = generate_mock("000001", "2023-01-01", "2024-12-31")
    pm = train_pooled([("TEST600519", df), ("TEST000001", df2)])
    check("pooled 训练返回样本外准确率", "test_accuracy" in pm)
    check("pooled 模型文件已保存", __import__("os").path.exists(pm["model_path"]))
    wfp = walk_forward_pooled(df, [("TEST600519", df), ("TEST000001", df2)], pm["model_path"])
    check("pooled wf 信号长度一致", len(wfp) == len(df))
    check("pooled wf 信号取值∈{0,1}", set(np.unique(wfp.dropna())).issubset({0, 1}))

    # 参数寻优:top_n 不应出现 0 交易退化组合
    ma_opt = optimize(get_strategy("ma_cross"), df, top_n=3)
    check("寻优返回 top_n 结果", len(ma_opt) == 3)
    check("寻优结果均为有交易组合", all(r["交易次数"] > 0 for r in ma_opt))

    # 数据缓存(增量更新 / 离线回退)
    sym = "TEST_CACHE_600519"
    d1 = store_load(sym, "20230101", "20231231")
    check("缓存加载返回行情", not d1.empty and len(d1) > 100)
    check("缓存已落地", sym in cached_symbols())
    d2 = store_load(sym, "20230101", "20240601")  # 区间扩展,应触发增量补数
    check("缓存增量后行数增多", len(d2) >= len(d1))

    # 全流程编排(pipeline,含与买入持有基准对比)
    report, path = run_pipeline(["TEST_PIPE_600519"], start="20230101", top_n=2)
    check("pipeline 生成报告文件", __import__("os").path.exists(path))
    check("pipeline 报告含 summary", "summary" in report and len(report["summary"]) == 1)
    check("pipeline 报告含 ML 训练指标", "ml" in report["symbols"]["TEST_PIPE_600519"])
    check("pipeline 报告含基准对比", "benchmark" in report["symbols"]["TEST_PIPE_600519"]["ml"])

    failed = [n for n, ok in PASS if not ok]
    print("\n=== 结果:", "全部通过" if not failed else f"{len(failed)} 项失败: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
