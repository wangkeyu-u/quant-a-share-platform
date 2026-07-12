"""量化平台 - 全流程自动化一键脚本。

数据更新 → 特征 → ML 训练 → ML 回测 → 规则策略参数寻优 → 生成报告(json)。

示例:
    python run_pipeline.py --sample                 # 用全部内置样例股票 + 长历史跑全流程
    python run_pipeline.py --sample --pooled        # 跨股票联合训练(用全市场数据)
    python run_pipeline.py --symbols 600519 000001
    python run_pipeline.py --symbols 600519 --start 20100101
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant.data.loader import sample_stocks
from quant.pipeline.runner import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="量化平台全流程自动化(数据饥渴版)")
    ap.add_argument("--symbols", nargs="*", default=None, help="股票代码列表,如 600519 000001")
    ap.add_argument("--sample", action="store_true", help="使用全部内置样例股票(默认)")
    ap.add_argument("--pooled", action="store_true", help="跨股票联合训练(用全市场数据训练一个大盘模型)")
    ap.add_argument("--start", default="20100101", help="起始日期 YYYYMMDD(默认 2010 年起,吃更多历史)")
    ap.add_argument("--end", default=None, help="结束日期 YYYYMMDD(默认今天)")
    ap.add_argument("--top-n", type=int, default=3, help="每个规则策略保留前 N 组参数")
    args = ap.parse_args()

    if args.symbols:
        symbols = [str(s).zfill(6) for s in args.symbols]
    else:
        symbols = sample_stocks()["code"].tolist()  # 全部 10 只样例股票

    print(f"开始全流程{'[联合训练]' if args.pooled else ''}: {symbols}  区间 {args.start}~{args.end or '今天'}")
    report, path = run_pipeline(symbols, args.start, args.end, top_n=args.top_n, pooled=args.pooled)

    print(f"\n完成,报告已保存: {path}")
    print("-" * 56)
    for s in report["summary"]:
        print(f"  {s['symbol']}  ML夏普={s['ml_sharpe']:.3f}  "
              f"ML收益={s['ml_return']:.2%}  交易={s['ml_trades']} 笔")
    print("-" * 56)


if __name__ == "__main__":
    main()
