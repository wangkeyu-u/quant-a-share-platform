"""统一 AI 网关包:本地 ML 推理 + 远程 LLM 都从同一个入口进出。"""
from quant.ai.config import GatewayConfig, load_config
from quant.ai.backends import LocalBackend, RemoteLLMBackend
from quant.ai.gateway import AIGateway, CallRecord, get_gateway
from quant.ai import analysis

__all__ = [
    "GatewayConfig", "load_config",
    "LocalBackend", "RemoteLLMBackend",
    "AIGateway", "CallRecord", "get_gateway",
    "analysis",
]
