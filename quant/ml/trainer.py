"""机器学习训练(数据饥渴版):分类器 + 回归器双模型、超参交叉验证寻优、
置信度门控的 walk-forward 预测(严格防前视)、以及跨股票联合(pooled)训练。

关键设计 —— 如何"用大量数据把模型训强"且不做假:
1. 同时训两个模型:
   - 分类器:预测 h 日后上涨概率(用于「该不该做」的置信度门控)。
   - 回归器:预测 h 日前向收益(用于「预期空间」判断)。
   两者都只在 [0, t) 的历史上训练,只用第 t 日特征预测,绝不用未来。
2. 超参搜索:在训练集上用「时序切分」做交叉验证选最优
   (n_estimators / max_depth / learning_rate),避免凭手感拍参数、也避免过拟合。
3. 置信度门控:只有「上涨概率 > 门槛 且 预测前向收益 > 门槛」才持仓,
   否则空仓——只做高确定性交易,这是实盘里最现实的"提高胜率"手段。
4. pooled 模式:把全市场多只股票的特征按「时间截断」拼接训练(特征都是比率,
   量纲一致可拼),每次预测第 t 日时只用 date < t 的全部股票历史,无前视。
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from quant.ml.features import CLS_TARGET, FEATURE_COLS, REG_TARGET, build_features

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(_ROOT, "models")

# 超参搜索网格(在训练集上做时序交叉验证选优)
PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.02},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.02},
]


def _clean(feats: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    if dropna:
        return feats.dropna(subset=FEATURE_COLS + [CLS_TARGET, REG_TARGET])
    return feats


def _new_clf(p):
    return GradientBoostingClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=0.8, random_state=42)


def _new_reg(p):
    return GradientBoostingRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=0.8, random_state=42)


def _best_params(X, y, make_fn, scoring):
    """时序交叉验证选最优超参。"""
    tscv = TimeSeriesSplit(n_splits=3)
    best, best_s = None, -np.inf
    for p in PARAM_GRID:
        try:
            s = cross_val_score(make_fn(p), X, y, cv=tscv, scoring=scoring, n_jobs=1)
            m = s.mean()
        except Exception:
            continue
        if m > best_s:
            best_s, best = m, p
    return best or PARAM_GRID[0], (None if np.isnan(best_s) else float(best_s))


def train_models(df: pd.DataFrame, symbol: str, horizon: int = 5,
                 model_path: str | None = None, test_size: float = 0.2) -> dict:
    """在全部历史上按时间切分训练/测试,训练双模型并保存,返回评估指标。

    切分方式:前 (1-test_size) 用于训练 + 超参搜索,后 test_size 作为样本外测试集,
    报告里的指标就是这段"没见过的数据"上的表现。
    """
    feats = _clean(build_features(df))
    X = feats[FEATURE_COLS]
    yc, yr = feats[CLS_TARGET], feats[REG_TARGET]
    n = len(feats)
    split = int(n * (1 - test_size))

    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    yc_tr, yc_te = yc.iloc[:split], yc.iloc[split:]
    yr_tr, yr_te = yr.iloc[:split], yr.iloc[split:]

    # 分类器:时序 CV 选超参
    clf_p, clf_cv = _best_params(Xtr, yc_tr, _new_clf, "roc_auc")
    clf = _new_clf(clf_p).fit(Xtr, yc_tr)
    # 回归器:时序 CV 选超参
    reg_p, reg_cv = _best_params(Xtr, yr_tr, _new_reg, "r2")
    reg = _new_reg(reg_p).fit(Xtr, yr_tr)

    # 样本外评估
    pred_c = clf.predict(Xte)
    acc = accuracy_score(yc_te, pred_c)
    auc = (roc_auc_score(yc_te, clf.predict_proba(Xte)[:, 1])
           if yc_te.nunique() > 1 else float("nan"))
    pred_r = reg.predict(Xte)
    r2 = r2_score(yr_te, pred_r)
    mae = mean_absolute_error(yr_te, pred_r)

    os.makedirs(MODELS_DIR, exist_ok=True)
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{symbol}.joblib")
    joblib.dump({"clf": clf, "reg": reg, "horizon": horizon,
                 "clf_params": clf_p, "reg_params": reg_p}, model_path)

    return {
        "model_path": model_path,
        "horizon": horizon,
        "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
        "clf_cv_auc": clf_cv, "reg_cv_r2": reg_cv,
        "test_accuracy": round(float(acc), 4),
        "test_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "test_r2": round(float(r2), 4),
        "test_mae": round(float(mae), 6),
        "clf_params": clf_p, "reg_params": reg_p,
        "feature_importances": dict(zip(
            FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_])),
    }


def _train_pair_on(feats_slice: pd.DataFrame):
    """在给定切片上训练一对模型,返回 (clf, reg) 或 None。"""
    sub = feats_slice.dropna(subset=FEATURE_COLS + [CLS_TARGET, REG_TARGET])
    if len(sub) < 50 or sub[CLS_TARGET].nunique() < 2:
        return None
    X, yc, yr = sub[FEATURE_COLS], sub[CLS_TARGET], sub[REG_TARGET]
    clf = _new_clf(PARAM_GRID[0]).fit(X, yc)
    reg = _new_reg(PARAM_GRID[0]).fit(X, yr)
    return clf, reg


def walk_forward_signals(df: pd.DataFrame, horizon: int = 5, lookback: int = 60,
                         window: int = 750, retrain_every: int = 20,
                         prob_threshold: float = 0.55,
                         return_threshold: float = 0.0) -> pd.Series:
    """逐日 walk-forward:用 [lo, t) 历史训练双模型,预测第 t 日,生成 0/1 仓位。

    门控:上涨概率 > prob_threshold 且 预测前向收益 > return_threshold 才持仓。
    严格防前视:第 t 日信号由 [0, t) 历史得到,只用第 t 日特征预测。
    """
    feats = build_features(df)
    n = len(df)
    position = pd.Series(0, index=df.index)
    if n < lookback + horizon + 2:
        return position

    clf, reg, trained_at = None, None, -9999
    for t in range(lookback, n - horizon):
        lo = max(0, t - window)
        if t - trained_at >= retrain_every or t == lookback:
            pair = _train_pair_on(feats.iloc[lo:t])
            if pair is None:
                continue
            clf, reg = pair
            trained_at = t
        if clf is None:
            continue
        Xt = feats.iloc[[t]][FEATURE_COLS]
        if Xt.isna().any().any():
            continue
        prob_up = clf.predict_proba(Xt)[0, 1]
        pred_ret = float(reg.predict(Xt)[0])
        if prob_up > prob_threshold and pred_ret > return_threshold:
            position.iloc[t] = 1
    return position


# ———————————————— 跨股票联合(pooled)训练 ————————————————
def _pool_frame(dfs: list) -> pd.DataFrame:
    """把多只股票特征拼接(带 symbol/date 标记),用于联合训练。"""
    frames = []
    for sym, d in dfs:
        f = build_features(d).copy()
        f["symbol"] = sym
        f["date"] = d["date"].values
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def train_pooled(dfs: list, model_path: str | None = None,
                 test_size: float = 0.15) -> dict:
    """把全市场多只股票的特征拼接起来,训一个"大盘模型"(数据量最大化)。

    切分:按时间把最后 test_size 比例作为样本外测试(用 date 排序),避免用未来数据。
    """
    pool = _pool_frame(dfs)
    pool = pool.dropna(subset=FEATURE_COLS + [CLS_TARGET, REG_TARGET])
    pool = pool.sort_values("date").reset_index(drop=True)
    n = len(pool)
    split = int(n * (1 - test_size))
    tr, te = pool.iloc[:split], pool.iloc[split:]

    clf_p, _ = _best_params(tr[FEATURE_COLS], tr[CLS_TARGET], _new_clf, "roc_auc")
    reg_p, _ = _best_params(tr[FEATURE_COLS], tr[REG_TARGET], _new_reg, "r2")
    clf = _new_clf(clf_p).fit(tr[FEATURE_COLS], tr[CLS_TARGET])
    reg = _new_reg(reg_p).fit(tr[FEATURE_COLS], tr[REG_TARGET])

    pred_c = clf.predict(te[FEATURE_COLS])
    acc = accuracy_score(te[CLS_TARGET], pred_c)
    auc = (roc_auc_score(te[CLS_TARGET], clf.predict_proba(te[FEATURE_COLS])[:, 1])
           if te[CLS_TARGET].nunique() > 1 else float("nan"))
    pred_r = reg.predict(te[FEATURE_COLS])
    r2 = r2_score(te[REG_TARGET], pred_r)

    os.makedirs(MODELS_DIR, exist_ok=True)
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "pooled.joblib")
    joblib.dump({"clf": clf, "reg": reg, "pooled": True}, model_path)
    return {
        "model_path": model_path, "pooled": True,
        "train_n": int(len(tr)), "test_n": int(len(te)),
        "test_accuracy": round(float(acc), 4),
        "test_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "test_r2": round(float(r2), 4),
        "feature_importances": dict(zip(
            FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_])),
    }


def walk_forward_pooled(df: pd.DataFrame, dfs: list, model_path: str,
                        horizon: int = 5, lookback: int = 60, window: int = 750,
                        retrain_every: int = 60, prob_threshold: float = 0.55,
                        return_threshold: float = 0.0,
                        max_train_rows: int = 8000) -> pd.Series:
    """用全市场数据做 walk-forward:每次重训时只用 date < 第 t 日 的历史,
    并取最近 max_train_rows 条(滚动窗口),既严格防前视,又把单步成本封顶。

    "用全部数据"且防前视:预测某只股票第 t 日时,训练集包含所有股票在 t 日之前
    的数据(不含任何未来信息);滚动上限让它既能吃海量历史、又不会越跑越慢。
    """
    # 特征只算一次(此前放在循环里,是主要的性能瓶颈)
    pool_all = _pool_frame(dfs).sort_values("date").reset_index(drop=True)
    feats_df = build_features(df)
    target_dates = df["date"].values
    n = len(df)
    position = pd.Series(0, index=df.index)
    if n < lookback + horizon + 2:
        return position

    model = None
    trained_at = -9999
    for t in range(lookback, n - horizon):
        if t - trained_at >= retrain_every or t == lookback or model is None:
            cut = pool_all[pool_all["date"] < target_dates[t]].tail(max_train_rows)
            pair = _train_pair_on(cut)
            if pair is None:
                continue
            model = pair
            trained_at = t
        clf, reg = model
        Xt = feats_df.iloc[[t]][FEATURE_COLS]
        if Xt.isna().any().any():
            continue
        prob_up = clf.predict_proba(Xt)[0, 1]
        pred_ret = float(reg.predict(Xt)[0])
        if prob_up > prob_threshold and pred_ret > return_threshold:
            position.iloc[t] = 1
    return position
