"""远程 LLM 分析函数 —— 所有「用大模型分析数据」的入口。

这些函数只通过 quant.ai.gateway 的远程后端发请求,绝不各自直连。
未配置远程网关时调用会抛 RuntimeError(由调用方决定如何降级)。
"""
from __future__ import annotations

import json


def _gw():
    from quant.ai.gateway import get_gateway
    return get_gateway()


def news_sentiment(text: str) -> str:
    """判断一段财经新闻/公告对股价的情绪倾向。"""
    prompt = (
        "请判断以下财经新闻/公告对股价的情绪倾向(正面/负面/中性),"
        "并给一句不超过 40 字的中文理由:\n\n" + str(text)
    )
    return _gw().analyze(
        prompt, system="你是一名严谨的量化研究员,只做客观情绪标注,不臆测。")


def summarize_report(report: dict) -> str:
    """把回测报告浓缩成几条关键中文结论。"""
    blob = json.dumps(report, ensure_ascii=False, default=str)
    if len(blob) > 4000:
        blob = blob[:4000] + "\n...(截断)"
    prompt = (
        "以下是某量化平台的回测报告(JSON)。请用中文给出 3 条关键结论:\n"
        "1) 哪些标的/策略样本外表现最好;\n"
        "2) 策略是否普遍跑赢买入持有基准;\n"
        "3) 主要风险点。\n\n" + blob
    )
    return _gw().analyze(prompt, system="你是客观的量化策略分析师,只总结已有数据。")


def market_commentary(df, metrics: dict) -> str:
    """基于近期行情与策略绩效,生成一段客观短线点评。"""
    recent = df.tail(20)[["date", "open", "high", "low", "close", "volume"]].to_string()
    prompt = (
        f"近期行情(最后 20 根):\n{recent}\n\n"
        f"策略样本外绩效:{json.dumps(metrics, ensure_ascii=False, default=str)}\n\n"
        "请给一段中文短线点评(不超过 150 字),风格客观、不喊单、不做投资建议。"
    )
    return _gw().analyze(prompt, system="你是资深交易员,客观点评行情与策略表现。")
