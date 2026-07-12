from quant.strategies.ma_cross import MACrossStrategy
from quant.strategies.macd import MACDStrategy
from quant.strategies.boll_breakout import BollBreakoutStrategy

# REGISTRY 在模块加载时只放入不依赖 ML 的策略,ML 策略延迟到首次使用时导入,
# 避免 quant.ml.strategy -> quant.strategies.base -> quant.strategies.__init__
# -> quant.strategies.registry -> quant.ml.strategy 的循环导入。
REGISTRY = {
    MACrossStrategy.name: MACrossStrategy(),
    MACDStrategy.name: MACDStrategy(),
    BollBreakoutStrategy.name: BollBreakoutStrategy(),
}


def _ensure_ml():
    if "ml_direction" not in REGISTRY:
        from quant.ml.strategy import MLStrategy
        REGISTRY[MLStrategy.name] = MLStrategy()


def list_strategies():
    _ensure_ml()
    return [(s.name, s.description) for s in REGISTRY.values()]


def get_strategy(name: str):
    _ensure_ml()
    return REGISTRY[name]
