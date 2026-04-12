from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data import config
from model.features import build_supervised_frame, make_xy_next_day
from utils.metrics import mae, mape, rmse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ModelName = Literal[
    "naive",
    "ma_7",
    "sarima",
    "ets",
    "random_forest",
    "gradient_boosting",
]


@dataclass
class ForecastResult:
    history: pd.Series
    forecast_index: pd.DatetimeIndex
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    model_name: str = ""


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    last = float(series.dropna().iloc[-1])
    return np.full(horizon, last)


def moving_average_forecast(series: pd.Series, window: int, horizon: int) -> np.ndarray:
    s = series.dropna()
    if len(s) < window:
        w = max(1, len(s))
    else:
        w = window
    last_mean = float(s.iloc[-w:].mean())
    return np.full(horizon, last_mean)


def sarima_forecast(series: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    s = series.dropna().astype(float)
    if len(s) < 30:
        p = naive_forecast(s, horizon)
        return p, None, None
    try:
        mod = SARIMAX(
            s,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean.values
        conf = fc.conf_int(alpha=0.2)
        return mean.astype(float), conf.iloc[:, 0].values.astype(float), conf.iloc[:, 1].values.astype(float)
    except Exception:
        try:
            mod = SARIMAX(s, order=(1, 1, 1), enforce_stationarity=False)
            res = mod.fit(disp=False)
            fc = res.get_forecast(steps=horizon)
            mean = fc.predicted_mean.values
            conf = fc.conf_int(alpha=0.2)
            return mean.astype(float), conf.iloc[:, 0].values.astype(float), conf.iloc[:, 1].values.astype(float)
        except Exception:
            p = naive_forecast(s, horizon)
            return p, None, None


def ets_forecast(series: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    s = series.dropna().astype(float)
    if len(s) < 14:
        p = naive_forecast(s, horizon)
        return p, None, None
    try:
        mod = ExponentialSmoothing(
            s,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        res = mod.fit(optimized=True)
        fc = res.forecast(horizon)
        pred = np.asarray(fc, dtype=float)
        resid_std = float(np.std(res.resid.dropna())) if len(res.resid.dropna()) else 0.0
        z = 1.28
        lo = pred - z * resid_std * np.sqrt(np.arange(1, horizon + 1))
        hi = pred + z * resid_std * np.sqrt(np.arange(1, horizon + 1))
        return pred, lo.astype(float), hi.astype(float)
    except Exception:
        p = naive_forecast(s, horizon)
        return p, None, None


def _fit_sklearn_regressor(name: str):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
    return GradientBoostingRegressor(random_state=42, max_depth=4, n_estimators=150, learning_rate=0.08)


def train_ml_model(df_daily: pd.DataFrame, target_col: str, model_name: str):
    sup = build_supervised_frame(df_daily, target_col)
    X, y = make_xy_next_day(sup, target_col)
    feature_columns = X.columns.tolist()
    reg = _fit_sklearn_regressor(model_name)
    reg.fit(X, y)
    return reg, feature_columns, target_col


def _row_features_at_end(
    df_block: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    sup = build_supervised_frame(df_block, target_col)
    last = sup.iloc[[-1]].reindex(columns=feature_columns)
    if last.isna().any().any() and len(sup) >= 2:
        last = last.fillna(sup.iloc[-2])
    return last


def ml_recursive_forecast(
    df_daily: pd.DataFrame,
    target_col: str,
    model,
    feature_columns: list[str],
    horizon: int,
    exog_cols: list[str] | None = None,
) -> np.ndarray:
    """Multi-step by appending point predictions and updating lags (exog forward-filled)."""
    exog_cols = exog_cols or [
        config.APPREHENDED,
        config.CBP_CUSTODY,
        config.TRANSFERS,
        config.DISCHARGED,
    ]
    block = df_daily.copy()
    preds = []
    for _ in range(horizon):
        row = _row_features_at_end(block, target_col, feature_columns)
        if row.isna().any().any():
            row = row.ffill(axis=1).bfill(axis=1)
        y_hat = float(model.predict(row.values)[0])
        preds.append(y_hat)
        next_idx = block.index[-1] + pd.Timedelta(days=1)
        new_row = {target_col: y_hat}
        for c in exog_cols:
            if c in block.columns:
                new_row[c] = float(block[c].iloc[-1])
        block = pd.concat([block, pd.DataFrame([new_row], index=[next_idx])])
        block.index = pd.DatetimeIndex(block.index)
        block = block.sort_index()
    return np.array(preds, dtype=float)


def run_model_forecast(
    df_daily: pd.DataFrame,
    target_col: str,
    model_name: ModelName,
    horizon: int,
    ml_bundle: tuple | None = None,
) -> ForecastResult:
    s = df_daily[target_col]
    idx = pd.date_range(df_daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if model_name == "naive":
        p = naive_forecast(s, horizon)
        return ForecastResult(s, idx, p, model_name=model_name)
    if model_name == "ma_7":
        p = moving_average_forecast(s, 7, horizon)
        return ForecastResult(s, idx, p, model_name=model_name)
    if model_name == "sarima":
        p, lo, hi = sarima_forecast(s, horizon)
        return ForecastResult(s, idx, p, lo, hi, model_name=model_name)
    if model_name == "ets":
        p, lo, hi = ets_forecast(s, horizon)
        return ForecastResult(s, idx, p, lo, hi, model_name=model_name)
    if model_name in ("random_forest", "gradient_boosting"):
        if ml_bundle is None:
            model, cols, _ = train_ml_model(df_daily, target_col, model_name)
        else:
            model, cols, _ = ml_bundle
        p = ml_recursive_forecast(df_daily, target_col, model, cols, horizon)
        resid_scale = float(s.diff().dropna().std() or 1.0) * 1.5
        z = 1.28
        steps = np.arange(1, horizon + 1, dtype=float)
        lo = p - z * resid_scale * np.sqrt(steps)
        hi = p + z * resid_scale * np.sqrt(steps)
        return ForecastResult(s, idx, p, lo, hi, model_name=model_name)
    raise ValueError(model_name)


def walk_forward_scores(
    df_daily: pd.DataFrame,
    target_col: str,
    model_name: ModelName,
    min_train: int = 400,
    step: int = 5,
    horizons: tuple[int, ...] = (1, 7, 14),
) -> dict:
    """Expanding-window pseudo backtest; returns metrics by horizon."""
    s = df_daily[target_col]
    n = len(s)
    ml_model = None
    ml_cols = None

    by_h = {h: {"pred": [], "true": []} for h in horizons}

    for start in range(min_train, n - max(horizons), step):
        train = df_daily.iloc[:start]
        for h in horizons:
            if start + h >= n:
                continue
            actual = float(s.iloc[start + h - 1])
            try:
                if model_name in ("random_forest", "gradient_boosting"):
                    if ml_model is None or start % (step * 10) == 0:
                        ml_model, ml_cols, _ = train_ml_model(train, target_col, model_name)
                    fc = run_model_forecast(train, target_col, model_name, h, ml_bundle=(ml_model, ml_cols, target_col))
                    pred = float(fc.point[h - 1])
                else:
                    fc = run_model_forecast(train, target_col, model_name, h)
                    pred = float(fc.point[h - 1])
            except Exception:
                pred = float(s.iloc[start - 1])
            by_h[h]["pred"].append(pred)
            by_h[h]["true"].append(actual)

    summary = {}
    for h in horizons:
        if not by_h[h]["pred"]:
            summary[h] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n": 0}
            continue
        yt = np.array(by_h[h]["true"])
        yp = np.array(by_h[h]["pred"])
        summary[h] = {
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "mape": mape(yt, yp),
            "n": len(yt),
        }
    return summary


def decomposition_plotly_df(series: pd.Series):
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = series.dropna().astype(float)
    if len(s) < 28:
        return None
    try:
        dec = seasonal_decompose(s, model="additive", period=7, extrapolate_trend="freq")
        df = pd.DataFrame(
            {
                "observed": dec.observed,
                "trend": dec.trend,
                "seasonal": dec.seasonal,
                "resid": dec.resid,
            }
        )
        return df
    except Exception:
        return None


def kpi_bundle(
    df_daily: pd.DataFrame,
    target_col: str,
    model_name: ModelName,
    breach_level: float | None = None,
    walk_forward: dict | None = None,
):
    wf = walk_forward
    if wf is None:
        wf = walk_forward_scores(df_daily, target_col, model_name, min_train=min(400, len(df_daily) // 2))
    h1 = wf.get(1, {})
    mape_1 = h1.get("mape", np.nan)
    acc_pct = float(max(0.0, 100.0 - mape_1)) if not np.isnan(mape_1) else np.nan
    surge_days = None
    try:
        s = df_daily[target_col]
        thresh = float(s.quantile(0.9))
        fc = run_model_forecast(df_daily, target_col, "sarima", 30)
        above = np.where(fc.point > thresh)[0]
        surge_days = int(above[0] + 1) if len(above) else None
    except Exception:
        surge_days = None
    breach_prob = None
    if breach_level is not None:
        try:
            fc = run_model_forecast(df_daily, target_col, model_name, 14)
            if fc.lower is not None and fc.upper is not None:
                breach_prob = float(np.mean(fc.point > breach_level))
            else:
                breach_prob = float(np.mean(fc.point > breach_level))
        except Exception:
            breach_prob = None
    stability = None
    try:
        errs = []
        s = df_daily[target_col]
        for end in range(200, len(df_daily), 50):
            sub = df_daily.iloc[:end]
            fc = run_model_forecast(sub, target_col, model_name, 1)
            errs.append(abs(fc.point[0] - float(s.iloc[end])) / max(abs(float(s.iloc[end])), 1.0))
        stability = float(1.0 / (1.0 + np.std(errs))) if errs else None
    except Exception:
        stability = None
    return {
        "forecast_accuracy_pct": acc_pct,
        "walk_forward": wf,
        "surge_lead_time_days": surge_days,
        "capacity_breach_prob_14d": breach_prob,
        "forecast_stability_index": stability,
    }
