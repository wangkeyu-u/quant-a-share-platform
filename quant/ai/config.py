"""AI 网关配置:从环境变量读取(OpenAI 兼容协议)。

本地 ML 推理始终启用;远程 LLM 仅当 AI_GATEWAY_BASE_URL 与
AI_GATEWAY_API_KEY 都设置时才启用。详见仓库根目录 .env.example。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GatewayConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    remote_enabled: bool = False
    local_enabled: bool = True
    timeout: int = 60


def load_config() -> GatewayConfig:
    base_url = (os.environ.get("AI_GATEWAY_BASE_URL") or "").strip()
    api_key = (os.environ.get("AI_GATEWAY_API_KEY") or "").strip()
    model = (os.environ.get("AI_GATEWAY_MODEL") or "gpt-4o-mini").strip()
    try:
        timeout = int(os.environ.get("AI_GATEWAY_TIMEOUT", "60"))
    except ValueError:
        timeout = 60
    remote_enabled = bool(base_url and api_key)
    return GatewayConfig(
        base_url=base_url, api_key=api_key, model=model,
        remote_enabled=remote_enabled, timeout=timeout,
    )
