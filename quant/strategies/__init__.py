"""策略包。"""
from quant.strategies.base import Strategy
from quant.strategies.registry import get_strategy, list_strategies

__all__ = ["Strategy", "get_strategy", "list_strategies"]
