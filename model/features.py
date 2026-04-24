from __future__ import annotations

import pandas as pd

from data import config
from data.loader import net_pressure


def build_supervised_frame(
    df: pd.DataFrame,
    target_col: str,
    lags: tuple[int, ...] = (1, 7, 14),
    roll_windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    """Row-level features for sklearn models; target is same-day value (use shifted target externally)."""
    # Keep only modeling-relevant columns so training does not use every raw column.
    keep_cols = [target_col]
    exog = [
        config.APPREHENDED,
        config.CBP_CUSTODY,
        config.TRANSFERS,
        config.DISCHARGED,
    ]
    keep_cols.extend([c for c in exog if c in df.columns and c != target_col])
    d = df[keep_cols].copy()
    for lag in lags:
        d[f"lag_{lag}"] = d[target_col].shift(lag)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = d[target_col].rolling(w, min_periods=1).mean()
        d[f"roll_var_{w}"] = d[target_col].rolling(w, min_periods=1).var().fillna(0)
    np_ = net_pressure(d)
    d["net_pressure"] = np_
    for lag in (1, 7):
        d[f"net_pressure_lag_{lag}"] = np_.shift(lag)
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    for c in exog:
        if c in d.columns:
            d[f"{c}_lag1"] = d[c].shift(1)
    return d


def make_xy_next_day(supervised: pd.DataFrame, target_col: str):
    """Predict t+1 using information through end of day t (includes same-day level)."""
    y = supervised[target_col].shift(-1)
    X = supervised
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]
