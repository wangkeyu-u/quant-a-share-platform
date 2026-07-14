# 量化炒股平台（沪深A股）/ Quantitative Trading Platform (A-Share)

> 面向沪深 A 股的本地量化交易平台 —— 行情获取、技术指标、策略回测、模拟交易、双模型 ML 训练、walk-forward 防前视、参数寻优、桌面 GUI、一键全流程 + 每日定时任务。纯 Python，无需任何 token。
>
> A local quantitative trading platform for China A-shares — market data, technical indicators, strategy backtesting, simulated trading, dual-model ML training, walk-forward anti-lookahead, parameter optimization, desktop GUI, one-click pipeline + daily scheduled tasks. Pure Python, no token required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-tkinter%20%2B%20matplotlib-3776AB)]()
[![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-akshare-success)](https://akshare.akfamily.xyz/)
[![No Token](https://img.shields.io/badge/No%20Token-Required-brightgreen)]()

---

## Overview (English)

A local-first quantitative trading platform for China A-shares (Shanghai/Shenzhen). It integrates market data fetching, technical indicators, event-driven backtesting, simulated trading, and a dual-model ML pipeline (GBM classification + regression) with walk-forward anti-lookahead training — all in pure Python with a tkinter desktop GUI.

**Key capabilities:**
- **Market data**: Real-time A-share data via [akshare](https://akshare.akfamily.xyz/), with automatic fallback to deterministic mock data when offline
- **Backtesting**: Event-driven, daily-frequency, long-only; supports commission, slippage, position sizing; outputs equity curve, performance metrics, and trade records
- **ML pipeline**: One-click full pipeline — data update → feature engineering → ML training → ML backtesting → rule optimization → report. Walk-forward training strictly uses `[0, t-1]` history to predict day `t`, **never using future data**
- **Unified AI gateway**: All AI calls (local GBM inference + remote LLM) route through a single gateway with automatic task-based routing and full audit logging
- **Built-in strategies**: MA crossover, MACD, Bollinger breakout, ML direction (gradient boosting)
- **Red = up, Green = down** (Chinese stock market convention)

---

# 量化炒股平台（沪深A股）

一个面向沪深 A 股的**本地量化交易平台**：行情获取、技术指标、策略回测、模拟交易一体化，纯 Python 桌面应用（tkinter + matplotlib），**无需任何 token**。

> 真实行情优先通过 [akshare](https://akshare.akfamily.xyz/) 拉取东方财富/新浪等公开数据；无网络或接口异常时自动回退到**确定性模拟数据**，软件永远可运行。

---

## 功能特性

- **行情 / 信号**：选股 + 自定义区间，绘制 K 线、成交量与买卖点（红涨绿跌，A 股习惯）。
- **回测引擎**：事件驱动、日频、多头；支持佣金、滑点、仓位比例；输出权益曲线、绩效指标与成交记录。
- **绩效指标**：总收益、年化收益、最大回撤、夏普比率、卡玛比率、胜率等。
- **模拟交易**：虚拟资金账户，可一键跑全程历史模拟，也可「下一交易日」逐日步进，模拟实盘决策。
- **内置策略（可插拔）**：
  - `ma_cross` 双均线交叉
  - `macd` MACD 金叉死叉
  - `boll_breakout` 布林带突破
  - `ml_direction` 机器学习（梯度提升预测次日涨跌，walk-forward 无前视）

---

## 数据自动化与训练流水线（全自动）

最核心的能力：**一键 / 定时跑通「数据更新 → 特征工程 → ML 训练 → ML 回测 → 规则寻优 → 报告」全流程**，无需人工干预。

### 一键脚本

```bash
# 对指定标的跑全流程
python run_pipeline.py --symbols 600519 000001 --start 20200101 --top-n 3

# 或用内置样例股票（离线可用）
python run_pipeline.py --sample --start 20200101 --top-n 3
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--symbols` | 股票代码列表（6 位），如 `600519 000001` |
| `--sample` | 使用内置 10 只样例股票（离线兜底） |
| `--start` | 起始日期 `YYYYMMDD`，默认 `20200101` |
| `--end` | 结束日期，默认今天 |
| `--top-n` | 每个规则策略保留的寻优组合数，默认 3 |

流程产物：
- `data_cache/<code>.csv`：增量更新的本地行情缓存（再次运行只补抓新交易日，速度更快）
- `models/<code>.joblib`：训练好的 ML 模型（可复用）
- `reports/report.json`：结构化报告（各标的 ML 绩效 + 规则策略寻优结果 + 汇总）

### 每日定时任务

**方式一（推荐）：WorkBuddy 自动化**
已内置名为「量化平台每日训练流水线」的每日定时任务（默认 18:00，A 股收盘后运行），自动执行上面的 `run_pipeline.py --sample` 并汇报结果。可在 WorkBuddy 自动化面板中修改执行时间或标的。

**方式二：系统 crontab**
如需脱离 WorkBuddy、由操作系统直接调度，可加一条 crontab：

```bash
# 每天 18:00 运行，日志写入 pipeline.log
0 18 * * * cd /绝对路径/quant_platform && \
  /绝对路径/python_env/bin/python run_pipeline.py --sample --start 20200101 >> pipeline.log 2>&1
```

> 训练数据防前视：`walk_forward_signals` 严格只用 `[0, t-1]` 的历史训练、再用第 t 日特征预测，**不会用到未来信息**；标签 `target = 次日是否上涨（close.shift(-1) > close）` 仅作监督目标。

---

## 统一 AI 网关

平台里**所有「调用 AI」的地方都收口到同一个网关**（`quant/ai/`），本地推理与远程大模型共用一个入口，按任务类型自动路由，且每次调用都记入日志可审计。

- **本地后端（`LocalBackend`）**：封装 scikit-learn 双模型（GBM 分类 + 回归）的构造 / 训练 / 预测。ML 训练（`train_models` / `train_pooled`）与 walk-forward 预测（`walk_forward_signals` / `walk_forward_pooled`）全部经此，不直连 sklearn。
- **远程后端（`RemoteLLMBackend`）**：OpenAI 兼容协议的 `/v1/chat/completions` 调用，用于「用 AI 分析数据」——新闻情绪、报告摘要、行情点评（见 `quant/ai/analysis.py`）。
- **路由规则**：`train / inference / signal / predict` → 本地；`sentiment / summary / commentary / chat` → 远程。`gateway.call(task, **kwargs)` 是统一入口，`get_gateway()` 返回单例。

### 配置（远程 LLM 可选）

复制 `.env.example` 为 `.env` 并填写（也可直接 export 环境变量）。**只有 `AI_GATEWAY_BASE_URL` 与 `AI_GATEWAY_API_KEY` 同时设置，远程后端才启用**；未设置时远程调用会优雅跳过，主流程不受影响。

```bash
cp .env.example .env
# 编辑 .env:填入你的 AI_GATEWAY_BASE_URL / AI_GATEWAY_API_KEY / AI_GATEWAY_MODEL
```

兼容任意 OpenAI 协议服务（OpenAI、DeepSeek、通义、本地 Ollama 等）。配置后，跑 `run_pipeline.py` 会在报告里附加 `ai_commentary`（用远程大模型对本次回测做中文总结），并在报告 `ai_gateway` 字段给出本次调用的后端分布与计数。

---

## 无界面测试

```bash
PYTHONPATH=. python tests/headless_test.py
```

覆盖：行情 / 指标 / 策略 / 回测 / 模拟交易 / 特征工程 / ML 训练 / walk-forward / 参数寻优 / 数据缓存增量 / 全流程编排，全部通过即代表链路健康。

---

## 环境要求

- Python 3.10+（已在 Python 3.13 验证）
- 操作系统：macOS / Windows / Linux（桌面环境）
- 联网（用于拉取真实 A 股行情，可选）

---

## 安装

```bash
# 1. 进入项目目录
cd quant_platform

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

> 依赖：akshare、pandas、numpy、matplotlib。tkinter 通常随 Python 自带；
> 若 Linux 报 `_tkinter` 缺失，请安装系统包（如 `sudo apt install python3-tk`）。

---

## 运行

```bash
python run.py
```

启动后会出现桌面窗口，包含四个标签页：**行情 / 信号**、**回测**、**模拟交易**、**说明**。

**说明** 标签页内嵌「AI 行情点评」面板：选中某只股票并加载行情后，点击 **生成 AI 点评**，远程大模型（经统一 AI 网关）会基于该股票近期行情与已有回测绩效生成一段客观短线点评。未配置远程网关时按钮会给出 `.env` 配置指引，不影响主流程。

---

## 使用说明

### 1. 行情 / 信号
- 选择或输入股票代码，设置起始 / 结束日期（`YYYYMMDD`）。
- 选择策略与参数（参数滑块随策略自动生成），点击 **加载行情并计算信号**。
- 画布展示 K 线 + 成交量 + 买卖点；顶部状态栏显示触发的买入次数。

### 2. 回测
- 设置初始资金、佣金率（默认 0.0003）、滑点率（默认 0.0005）、仓位比例（1.0 = 满仓）。
- 点击 **运行回测**，查看权益曲线、绩效指标表与成交记录。

### 3. 模拟交易
- **运行全程模拟**：按历史信号跑完整虚拟盘。
- **下一交易日 ▶**：逐日步进，模拟实盘逐根 K 线决策。
- **重置**：清空账户重新来过。

---

## 目录结构

```
quant_platform/
├── run.py                  # 桌面应用入口
├── run_pipeline.py         # 一键全流程脚本（数据→训练→回测→寻优→报告）
├── requirements.txt
├── README.md
├── tests/
│   └── headless_test.py    # 无界面核心链路测试
├── data_cache/             # 行情增量缓存（自动生成）
├── models/                 # 训练好的 ML 模型（自动生成）
├── reports/                # 报告 report.json（自动生成）
└── quant/
    ├── ai/                 # 统一 AI 网关（本地 GBM 推理 + 远程 LLM）
    ├── data/               # 行情加载（akshare + 离线兜底）+ 本地缓存 store
    ├── indicators/         # MA / MACD / RSI / KDJ / BOLL
    ├── strategies/         # 策略（base + 具体策略 + 注册表）
    ├── backtest/           # 回测引擎 + 绩效指标 + 参数寻优
    ├── ml/                 # 特征工程 / 训练 / walk-forward / ML 策略
    ├── pipeline/           # 全流程编排 runner
    ├── trading/            # 模拟交易账户
    └── gui/                # tkinter 界面 + matplotlib 绘图
```

---

## 扩展新策略

继承 `quant.strategies.base.Strategy`，实现 `generate_signals(df, **params)` 返回 0/1 仓位序列，并在 `registry.py` 注册即可在 GUI 中出现（参数定义会自动生成滑块）：

```python
from quant.strategies.base import Strategy

class MyStrategy(Strategy):
    name = "my_strategy"
    description = "示例策略"
    params = {"window": (10, 3, 60, 1)}

    def generate_signals(self, df, window=10):
        # 返回与 df 等长的 0/1 序列
        ...
```

---

## 数据源说明

- 默认通过 `akshare.stock_zh_a_hist` 获取沪深 A 股前复权日线。
- 首次运行「刷新股票列表」会从 akshare 拉取全市场代码表（需联网，可能稍慢）。
- 任何网络 / 接口异常都会自动切换为内置模拟数据，不影响软件运行。

---

## ⚠️ 风险提示

- 本软件仅用于**量化学习与策略研究**，所有交易均为虚拟模拟，**不构成任何投资建议**。
- 历史回测收益不代表未来表现；实盘前请充分验证并自担风险。
- akshare 数据来自第三方公开渠道，可能存在延迟或误差。
