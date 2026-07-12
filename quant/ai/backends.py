"""AI 网关的两个后端。

- LocalBackend:封装 scikit-learn 双模型(GBM 分类 + 回归)的训练与推理。
  所有本地 ML 的「构造/训练/预测」都只能从这里走,保证单一入口。
- RemoteLLMBackend:OpenAI 兼容的远程大模型调用(/v1/chat/completions)。
  所有外部 LLM 分析(新闻情绪、报告摘要、行情点评)都只能从这里走。

注意:本文件不 import quant.ml.*,避免与 trainer -> gateway 形成循环导入。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from quant.ai.config import GatewayConfig


class LocalBackend:
    """本地 ML 推理后端 —— 单一入口,以后想换远程模型只改这一处。"""

    DEFAULT_PARAM = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.8, "random_state": 42,
    }

    def make_estimator(self, params, kind: str = "clf"):
        p = dict(params or self.DEFAULT_PARAM)
        p.setdefault("subsample", 0.8)
        p.setdefault("random_state", 42)
        if kind == "clf":
            return GradientBoostingClassifier(**p)
        return GradientBoostingRegressor(**p)

    def fit_pair(self, X, y_cls, y_reg, clf_params=None, reg_params=None):
        """训练一对模型(分类器 + 回归器),返回 (clf, reg)。"""
        clf = self.make_estimator(clf_params, "clf").fit(X, y_cls)
        reg = self.make_estimator(reg_params, "reg").fit(X, y_reg)
        return clf, reg

    def predict_pair(self, model_pair, X):
        """批量预测:返回 (上涨概率数组, 预测前向收益数组)。"""
        clf, reg = model_pair
        proba = clf.predict_proba(X)[:, 1]
        pred = reg.predict(X)
        return proba, pred

    def score_pair(self, model_pair, X):
        """单条预测:返回 (prob_up, pred_return)。"""
        proba, pred = self.predict_pair(model_pair, X)
        return float(proba[0]), float(pred[0])

    def predict_signal(self, model_pair, X, prob_threshold, return_threshold):
        """置信度门控:概率>门槛 且 预期收益>门槛 才持仓(1),否则空仓(0)。"""
        prob_up, pred_ret = self.score_pair(model_pair, X)
        return 1 if (prob_up > prob_threshold and pred_ret > return_threshold) else 0


class RemoteLLMBackend:
    """远程大模型后端 —— OpenAI 兼容协议,单一入口。"""

    def __init__(self, config: GatewayConfig):
        self.config = config

    def complete(self, prompt, system=None, temperature=0.3, max_tokens=800):
        import requests  # 本地后端不需要,延迟到真正远程调用时才 import

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(prompt)})
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=body,
                             timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
