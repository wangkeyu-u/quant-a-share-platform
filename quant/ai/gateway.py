"""统一 AI 网关。

把所有「调用 AI」的地方收拢到这一个入口:
- 本地任务(train / inference / signal / predict) -> LocalBackend(GBM)
- 远程任务(sentiment / summary / commentary / chat) -> RemoteLLMBackend(OpenAI 兼容)

按 task 类型自动路由。所有调用都记入 self.log,可随时审计
「平台的 AI 是不是都走了同一个网关」。取用单例 get_gateway()。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from quant.ai.backends import LocalBackend, RemoteLLMBackend
from quant.ai.config import GatewayConfig, load_config

# 走远程 LLM 的任务类型
REMOTE_TASKS = {"sentiment", "summary", "commentary", "chat", "analyze"}


@dataclass
class CallRecord:
    ts: str
    backend: str        # "local" | "remote"
    task: str
    status: str         # "ok" | "error"
    latency_ms: float
    detail: str = ""


class AIGateway:
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_config()
        self.local = LocalBackend()
        self.remote = RemoteLLMBackend(self.config) if self.config.remote_enabled else None
        self.log: list[CallRecord] = []
        self._lock = threading.Lock()

    # ---- 记账 ----
    def _record(self, backend, task, status, latency_ms, detail=""):
        with self._lock:
            self.log.append(CallRecord(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                backend=backend, task=task, status=status,
                latency_ms=round(latency_ms, 1), detail=str(detail),
            ))

    def log_summary(self) -> dict:
        with self._lock:
            local_n = sum(1 for r in self.log if r.backend == "local")
            remote_n = sum(1 for r in self.log if r.backend == "remote")
            err_n = sum(1 for r in self.log if r.status == "error")
        return {"total": len(self.log), "local": local_n,
                "remote": remote_n, "errors": err_n,
                "remote_configured": self.remote is not None}

    # ---- 本地 ML:训练 ----
    def make_estimator(self, params, kind: str = "clf"):
        return self.local.make_estimator(params, kind)

    def fit_pair(self, X, y_cls, y_reg, clf_params=None, reg_params=None):
        t0 = time.time()
        try:
            out = self.local.fit_pair(X, y_cls, y_reg, clf_params, reg_params)
            self._record("local", "train", "ok", (time.time() - t0) * 1000)
            return out
        except Exception as e:  # pragma: no cover - 训练失败要上抛,这里只记账
            self._record("local", "train", "error", (time.time() - t0) * 1000, e)
            raise

    # ---- 本地 ML:推理 ----
    def predict_pair(self, model_pair, X):
        t0 = time.time()
        out = self.local.predict_pair(model_pair, X)
        self._record("local", "inference", "ok", (time.time() - t0) * 1000)
        return out

    def score_pair(self, model_pair, X):
        return self.local.score_pair(model_pair, X)

    def predict_signal(self, model_pair, X, prob_threshold, return_threshold):
        return self.local.predict_signal(model_pair, X, prob_threshold, return_threshold)

    # ---- 远程 LLM ----
    def analyze(self, prompt, system=None, **kwargs):
        if not self.remote:
            raise RuntimeError(
                "AI 网关远程后端未配置:请设置环境变量 AI_GATEWAY_BASE_URL "
                "与 AI_GATEWAY_API_KEY(OpenAI 兼容协议)。")
        t0 = time.time()
        try:
            out = self.remote.complete(prompt, system, **kwargs)
            self._record("remote", "analyze", "ok", (time.time() - t0) * 1000,
                         f"len={len(out)}")
            return out
        except Exception as e:
            self._record("remote", "analyze", "error", (time.time() - t0) * 1000, e)
            raise

    # ---- 统一路由入口 ----
    def call(self, task: str, **kwargs):
        """所有 AI 调用最终都收口到这里,按 task 自动选后端。

        task in {train, inference, signal, predict} -> 本地后端
        task in {sentiment, summary, commentary, chat, analyze} -> 远程后端
        """
        if task == "train":
            return self.fit_pair(kwargs["X"], kwargs["y_cls"], kwargs["y_reg"],
                                 kwargs.get("clf_params"), kwargs.get("reg_params"))
        if task == "predict":
            return self.predict_pair(kwargs["model_pair"], kwargs["X"])
        if task == "signal":
            return self.predict_signal(
                kwargs["model_pair"], kwargs["X"],
                kwargs["prob_threshold"], kwargs["return_threshold"])
        if task in REMOTE_TASKS:
            return self.analyze(kwargs["prompt"], system=kwargs.get("system"), **kwargs)
        raise ValueError(f"未知 AI 任务类型: {task}")


_GATEWAY: Optional[AIGateway] = None


def get_gateway(config: Optional[GatewayConfig] = None) -> AIGateway:
    global _GATEWAY
    if _GATEWAY is None or config is not None:
        _GATEWAY = AIGateway(config)
    return _GATEWAY
