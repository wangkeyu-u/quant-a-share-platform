"""量化炒股平台 - 桌面 GUI(tkinter + matplotlib)。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime, timedelta
import threading

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from quant.data.loader import get_daily, get_stock_list, sample_stocks
from quant.indicators.tech import ma
from quant.strategies.registry import get_strategy, list_strategies
from quant.backtest.engine import Backtest
from quant.trading.paper import PaperTrader
from quant.gui.chart import plot_equity, plot_kline
from quant.ai.gateway import get_gateway
from quant.ai import analysis

PERIODS_PER_YEAR = 252


ABOUT_CONTENT = (
    "量化炒股平台 (沪深A股) — 使用说明\n"
    "================================\n\n"
    "1. 行情 / 信号\n"
    "   - 选择或输入股票代码,设置起止日期,选择策略与参数。\n"
    "   - 点击『加载行情并计算信号』绘制 K 线、成交量与买卖点。\n"
    "   - 数据源:优先 akshare 真实 A 股日线;无网络/接口异常时自动回退到确定性模拟数据。\n\n"
    "2. 回测\n"
    "   - 设置初始资金、佣金率、滑点率、仓位比例(1.0=满仓)。\n"
    "   - 运行回测后查看权益曲线、绩效指标(年化/最大回撤/夏普/卡玛/胜率)与成交记录。\n\n"
    "3. 模拟交易\n"
    "   - 『运行全程模拟』按历史信号跑完整虚拟盘;『下一交易日』逐日步进,模拟实盘决策。\n\n"
    "4. AI 行情点评\n"
    "   - 在『说明』页点击『生成 AI 点评』,远程大模型(经统一 AI 网关)会基于该股票\n"
    "     近期行情与已有回测绩效生成一段客观短线点评。未配置远程网关时给出 .env 指引。\n\n"
    "内置策略:双均线交叉、MACD 金叉死叉、布林带突破、ML 双模型(可插拔扩展)。\n\n"
    "⚠️ 风险提示\n"
    "   - 本软件仅用于量化学习与策略研究,所有交易均为虚拟模拟,不构成任何投资建议。\n"
    "   - 历史回测收益不代表未来表现;实盘前请充分验证并自担风险。\n"
    "   - akshare 数据为第三方公开数据,可能存在延迟或误差。\n"
)


def fmt_money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)


def fmt_pct(v) -> str:
    try:
        return f"{float(v) * 100:,.2f}%"
    except Exception:
        return str(v)


class QuantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("量化炒股平台 · 沪深A股回测/模拟交易")
        self.root.geometry("1080x760")

        self.df = None
        self.signals = None
        self.data_key = None
        self.bt_result = None
        self.paper = None
        self.sim_index = 0
        self.param_controls = {}

        # 股票列表
        self.stocks = sample_stocks()
        self.stock_names = {f"{r.code} {r.name}": r.code for r in self.stocks.itertuples()}

        # 默认日期
        end = datetime.today()
        start = end - timedelta(days=730)
        self.start_var = tk.StringVar(value=start.strftime("%Y%m%d"))
        self.end_var = tk.StringVar(value=end.strftime("%Y%m%d"))
        self.stock_var = tk.StringVar(value=list(self.stock_names)[0])
        self.strategy_var = tk.StringVar(value=list_strategies()[0][0])

        self._build()

    # ---------- 构建界面 ----------
    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=1, padx=6, pady=6)

        self.tab_market = ttk.Frame(nb)
        self.tab_backtest = ttk.Frame(nb)
        self.tab_paper = ttk.Frame(nb)
        self.tab_about = ttk.Frame(nb)
        nb.add(self.tab_market, text="行情 / 信号")
        nb.add(self.tab_backtest, text="回测")
        nb.add(self.tab_paper, text="模拟交易")
        nb.add(self.tab_about, text="说明")

        self._build_market()
        self._build_backtest()
        self._build_paper()
        self._build_about()

    def _build_market(self):
        top = ttk.Frame(self.tab_market)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Label(top, text="股票:").grid(row=0, column=0, sticky=tk.W)
        self.stock_combo = ttk.Combobox(top, textvariable=self.stock_var,
                                       values=list(self.stock_names), width=22, state="readonly")
        self.stock_combo.grid(row=0, column=1, padx=4)
        ttk.Button(top, text="刷新股票列表", command=self._refresh_stocks).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="开始(YYYYMMDD):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.start_var, width=14).grid(row=1, column=1, padx=4, sticky=tk.W)
        ttk.Label(top, text="结束(YYYYMMDD):").grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(top, textvariable=self.end_var, width=14).grid(row=1, column=3, padx=4, sticky=tk.W)

        ttk.Label(top, text="策略:").grid(row=2, column=0, sticky=tk.W)
        self.strat_combo = ttk.Combobox(top, textvariable=self.strategy_var,
                                        values=[s[0] for s in list_strategies()],
                                        width=18, state="readonly")
        self.strat_combo.grid(row=2, column=1, padx=4, sticky=tk.W)
        self.strat_combo.bind("<<ComboboxSelected>>", lambda e: self._build_param_controls())
        ttk.Button(top, text="加载行情并计算信号", command=self._on_compute).grid(row=2, column=3, padx=4)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(top, textvariable=self.status_var, foreground="blue").grid(row=3, column=0, columnspan=4, sticky=tk.W)

        # 参数区
        self.param_frame = ttk.LabelFrame(self.tab_market, text="策略参数")
        self.param_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        self._build_param_controls()

        # 图区
        self.kline_frame = ttk.Frame(self.tab_market)
        self.kline_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=6, pady=4)

    def _build_param_controls(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_controls = {}
        strat = get_strategy(self.strategy_var.get())
        if not strat.params:
            ttk.Label(self.param_frame, text="(该策略无参数)").pack(anchor=tk.W)
            return
        for i, (pname, (default, lo, hi, step)) in enumerate(strat.params.items()):
            ttk.Label(self.param_frame, text=pname).grid(row=i, column=0, sticky=tk.W, padx=4)
            var = tk.DoubleVar(value=default)
            scale = ttk.Scale(self.param_frame, from_=lo, to=hi, variable=var,
                              orient=tk.HORIZONTAL, length=200)
            scale.grid(row=i, column=1, padx=4)
            lab = ttk.Label(self.param_frame, text=str(default), width=8)
            lab.grid(row=i, column=2, padx=4)

            def _upd(v, lab=lab, step=step):
                try:
                    val = float(v)
                    if step >= 1:
                        val = int(round(val))
                    lab.config(text=str(val))
                except Exception:
                    pass

            scale.config(command=_upd)
            self.param_controls[pname] = (var, step)

    def _refresh_stocks(self):
        self.status_var.set("正在刷新股票列表(可能需联网)...")
        self.root.update_idletasks()
        try:
            df = get_stock_list()
            self.stocks = df
            self.stock_names = {f"{r.code} {r.name}": r.code for r in df.itertuples()}
            self.stock_combo["values"] = list(self.stock_names)
            if list(self.stock_names):
                self.stock_var.set(list(self.stock_names)[0])
            self.status_var.set(f"已加载 {len(self.stock_names)} 只股票")
        except Exception as e:
            self.status_var.set(f"刷新失败: {e}")

    def _current_params(self) -> dict:
        out = {}
        strat = get_strategy(self.strategy_var.get())
        for pname, (var, step) in self.param_controls.items():
            val = var.get()
            if step >= 1:
                val = int(round(val))
            out[pname] = val
        return out

    def _ensure_data(self):
        code = self.stock_names.get(self.stock_var.get())
        if not code:
            code = str(self.stock_var.get()).split()[0]
        key = (code, self.start_var.get(), self.end_var.get())
        if key != self.data_key or self.df is None:
            df = get_daily(code, self.start_var.get(), self.end_var.get)
            self.df = df
            self.data_key = key
            self.signals = None
            self.bt_result = None
            self.paper = None
        return code

    def _on_compute(self):
        try:
            code = self._ensure_data()
            strat = get_strategy(self.strategy_var.get())
            params = self._current_params()
            self.signals = strat.generate_signals(self.df, **params)
            # MA 叠加(展示用)
            ma_lines = {}
            for w in (5, 20, 60):
                if len(self.df) > w:
                    ma_lines[f"MA{w}"] = ma(self.df["close"], w)
            fig = plot_kline(self.df, self.signals, ma_lines)
            self._show_fig(self.kline_frame, fig)
            n_buy = int(((self.signals == 1) & (self.signals.shift(1) == 0)).sum())
            self.status_var.set(
                f"{code} 已加载 {len(self.df)} 根K线,信号买入触发 {n_buy} 次 | 参数: {params}")
        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {e}")

    # ---------- 回测 ----------
    def _build_backtest(self):
        ctrl = ttk.Frame(self.tab_backtest)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.cash_var = tk.StringVar(value="1000000")
        self.comm_var = tk.StringVar(value="0.0003")
        self.slip_var = tk.StringVar(value="0.0005")
        self.stake_var = tk.StringVar(value="1.0")

        rows = [
            ("初始资金", self.cash_var),
            ("佣金率", self.comm_var),
            ("滑点率", self.slip_var),
            ("仓位比例", self.stake_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(ctrl, text=label).grid(row=0, column=i * 2, padx=4, sticky=tk.W)
            ttk.Entry(ctrl, textvariable=var, width=12).grid(row=0, column=i * 2 + 1, padx=4)

        ttk.Button(ctrl, text="运行回测", command=self._on_backtest).grid(row=0, column=8, padx=8)

        self.bt_metrics = ttk.Treeview(self.tab_backtest, columns=("k", "v"), show="headings", height=6)
        self.bt_metrics.heading("k", text="指标")
        self.bt_metrics.heading("v", text="数值")
        self.bt_metrics.column("k", width=160)
        self.bt_metrics.column("v", width=160)
        self.bt_metrics.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.bt_log = ttk.Treeview(self.tab_backtest,
                                   columns=("date", "side", "price", "shares", "amount", "pnl"),
                                   show="headings", height=6)
        for c, t in [("date", "日期"), ("side", "方向"), ("price", "价格"),
                     ("shares", "股数"), ("amount", "金额"), ("pnl", "盈亏")]:
            self.bt_log.heading(c, text=t)
            self.bt_log.column(c, width=110)
        self.bt_log.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.bt_frame = ttk.Frame(self.tab_backtest)
        self.bt_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=6, pady=4)

    def _on_backtest(self):
        if self.df is None or self.signals is None:
            messagebox.showwarning("提示", "请先在『行情/信号』页加载行情并计算信号")
            return
        try:
            bt = Backtest(self.df,
                          initial_cash=float(self.cash_var.get()),
                          commission=float(self.comm_var.get()),
                          slippage=float(self.slip_var.get()),
                          stake=float(self.stake_var.get()))
            res = bt.run(self.signals)
            self.bt_result = res
            self._fill_metrics(res["metrics"])
            self._fill_log(self.bt_log, res["trades"])
            fig = plot_equity(res["equity"], res["benchmark"])
            self._show_fig(self.bt_frame, fig)
        except Exception as e:
            messagebox.showerror("错误", f"回测失败: {e}")

    def _fill_metrics(self, m: dict):
        self.bt_metrics.delete(*self.bt_metrics.get_children())
        order = ["起始资金", "结束资金", "总收益率", "年化收益", "最大回撤",
                 "夏普比率", "卡玛比率", "交易次数", "平仓次数", "胜率",
                 "平均盈利", "平均亏损"]
        for k in order:
            if k in m:
                v = m[k]
                if isinstance(v, float):
                    if "率" in k or "回撤" in k:
                        v = fmt_pct(v)
                    elif "资金" in k or "盈利" in k or "亏损" in k or "金额" in k:
                        v = fmt_money(v)
                    else:
                        v = f"{v:.4f}"
                self.bt_metrics.insert("", tk.END, values=(k, v))

    def _fill_log(self, tree, trades):
        tree.delete(*tree.get_children())
        for t in trades:
            tree.insert("", tk.END, values=(
                str(t["date"])[:10], t["side"], fmt_money(t["price"]),
                f"{t['shares']:.2f}", fmt_money(t["amount"]),
                "" if t["pnl"] is None else fmt_money(t["pnl"]),
            ))

    # ---------- 模拟交易 ----------
    def _build_paper(self):
        ctrl = ttk.Frame(self.tab_paper)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(ctrl, text="运行全程模拟", command=self._on_paper_run).grid(row=0, column=0, padx=4)
        ttk.Button(ctrl, text="下一交易日 ▶", command=self._on_paper_step).grid(row=0, column=1, padx=4)
        ttk.Button(ctrl, text="重置", command=self._on_paper_reset).grid(row=0, column=2, padx=4)

        self.paper_status = tk.StringVar(value="尚未运行模拟")
        ttk.Label(ctrl, textvariable=self.paper_status, foreground="purple").grid(row=0, column=3, padx=10)

        self.paper_tree = ttk.Treeview(self.tab_paper,
                                       columns=("date", "side", "price", "shares", "amount", "pnl"),
                                       show="headings", height=8)
        for c, t in [("date", "日期"), ("side", "方向"), ("price", "价格"),
                     ("shares", "股数"), ("amount", "金额"), ("pnl", "盈亏")]:
            self.paper_tree.heading(c, text=t)
            self.paper_tree.column(c, width=110)
        self.paper_tree.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.paper_frame = ttk.Frame(self.tab_paper)
        self.paper_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=6, pady=4)

    def _paper_reset_state(self):
        code = self._ensure_data()
        self.paper = PaperTrader(
            initial_cash=float(self.cash_var.get()),
            commission=float(self.comm_var.get()),
            slippage=float(self.slip_var.get()),
            stake=float(self.stake_var.get()),
        )
        self.sim_index = 0

    def _on_paper_reset(self):
        try:
            self._paper_reset_state()
            self.paper_status.set("已重置模拟账户")
            self.paper_tree.delete(*self.paper_tree.get_children())
            self._show_fig(self.paper_frame, plot_equity(
                pd.Series([self.paper.initial_cash], name="equity")))
        except Exception as e:
            messagebox.showerror("错误", f"重置失败: {e}")

    def _ensure_signals(self):
        if self.signals is None:
            self._on_compute()

    def _on_paper_run(self):
        try:
            self._ensure_data()
            self._ensure_signals()
            self._paper_reset_state()
            for i in range(len(self.df)):
                self.paper.step(self.df["date"].iloc[i], self.df["close"].iloc[i],
                                float(self.signals.iloc[i]))
            self.sim_index = len(self.df)
            self._refresh_paper_view()
        except Exception as e:
            messagebox.showerror("错误", f"模拟失败: {e}")

    def _on_paper_step(self):
        try:
            if self.paper is None:
                self._paper_reset_state()
            self._ensure_signals()
            if self.sim_index >= len(self.df):
                self.paper_status.set("已到行情末尾")
                return
            row = self.df.iloc[self.sim_index]
            self.paper.step(row["date"], row["close"], float(self.signals.iloc[self.sim_index]))
            self.sim_index += 1
            self._refresh_paper_view()
        except Exception as e:
            messagebox.showerror("错误", f"步进失败: {e}")

    def _refresh_paper_view(self):
        summ = self.paper.summary()
        eq = summ["equity"]
        last = eq.iloc[-1]
        pos = self.paper.position
        self.paper_status.set(
            f"日期 {str(self.df['date'].iloc[min(self.sim_index, len(self.df)-1)])[:10]} | "
            f"持仓比例 {pos:.0%} | 现金 {fmt_money(self.paper.cash)} | "
            f"市值 {fmt_money(last)} | 成交 {len(summ['trades'])} 笔")
        self._fill_log(self.paper_tree, summ["trades"])
        fig = plot_equity(eq)
        self._show_fig(self.paper_frame, fig)

    # ---------- 说明 ----------
    def _build_about(self):
        # ===== 上半部分:AI 行情点评(远程大模型,经统一网关) =====
        ai_frame = ttk.LabelFrame(self.tab_about, text="AI 行情点评 (远程大模型)")
        ai_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        top = ttk.Frame(ai_frame)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(top, text="标的:").pack(side=tk.LEFT)
        lab = ttk.Label(top, textvariable=self.stock_var, width=24,
                        relief=tk.SUNKEN, anchor=tk.W)
        lab.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="生成 AI 点评", command=self._on_ai_commentary).pack(side=tk.LEFT, padx=6)
        self.ai_status_var = tk.StringVar(value=self._gateway_status_text())
        ttk.Label(top, textvariable=self.ai_status_var, foreground="gray").pack(side=tk.LEFT, padx=6)

        self.ai_text = scrolledtext.ScrolledText(ai_frame, wrap=tk.WORD, height=8)
        self.ai_text.pack(fill=tk.BOTH, expand=0, padx=6, pady=(0, 6))
        self.ai_text.insert(tk.END,
            "点击『生成 AI 点评』,由远程大模型基于该股票近期行情(及已有回测绩效)生成客观短线点评。\n"
            "需先在本机配置 .env(见 .env.example):AI_GATEWAY_BASE_URL / AI_GATEWAY_API_KEY / AI_GATEWAY_MODEL。")
        self.ai_text.config(state=tk.DISABLED)

        # ===== 下半部分:使用说明(静态,不可编辑) =====
        help_frame = ttk.LabelFrame(self.tab_about, text="使用说明")
        help_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=8, pady=(4, 8))
        txt = scrolledtext.ScrolledText(help_frame, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=1, padx=6, pady=6)
        txt.insert(tk.END, ABOUT_CONTENT)
        txt.config(state=tk.DISABLED)

    def _gateway_status_text(self) -> str:
        try:
            gw = get_gateway()
            return "远程AI: 已配置 ✓" if gw.remote is not None else "远程AI: 未配置(见 .env.example)"
        except Exception:
            return "远程AI: 状态未知"

    def _on_ai_commentary(self):
        # 确保已加载所选标的行情(无则先拉取)
        try:
            self._ensure_data()
        except Exception as e:
            messagebox.showerror("错误", f"加载行情失败: {e}")
            return
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("提示", "行情为空,无法生成点评")
            return

        # 优先用已有回测绩效;否则传空 dict(点评主要基于近期价量)
        metrics = {}
        if self.bt_result is not None:
            metrics = self.bt_result.get("metrics", {}) or {}

        self.ai_status_var.set("远程AI: 生成中...")
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert(tk.END, "正在请求远程大模型,请稍候...")
        self.ai_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

        def worker():
            try:
                comment = analysis.market_commentary(self.df, metrics)
                ok, text = True, comment
            except RuntimeError as e:
                ok, text = False, (
                    f"无法生成 AI 点评:{e}\n\n"
                    "请在项目根目录创建 .env 并填入(参考 .env.example):\n"
                    "AI_GATEWAY_BASE_URL=https://你的兼容接口/v1\n"
                    "AI_GATEWAY_API_KEY=sk-xxxx\n"
                    "AI_GATEWAY_MODEL=gpt-4o-mini\n")
            except Exception as e:
                ok, text = False, f"AI 点评生成出错:{e}"
            self.root.after(0, self._set_ai_commentary, text, ok)

        threading.Thread(target=worker, daemon=True).start()

    def _set_ai_commentary(self, text: str, ok: bool):
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert(tk.END, text)
        self.ai_text.config(state=tk.DISABLED)
        self.ai_status_var.set(self._gateway_status_text())

    # ---------- 工具 ----------
    def _show_fig(self, parent, fig):
        for w in list(parent.winfo_children()):
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print("无法启动图形界面(可能无显示器环境):", e)
        print("请在本机桌面环境运行: python run.py")
        return
    app = QuantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
