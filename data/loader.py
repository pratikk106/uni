from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from data import config


def _strip_number(x) -> float:
    if pd.isna(x) or x == "":
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = re.sub(r",", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df = df[df[config.DATE].notna() & (df[config.DATE].astype(str).str.strip() != "")]
    df[config.DATE] = pd.to_datetime(df[config.DATE], errors="coerce")
    df = df.dropna(subset=[config.DATE])
    for c in [
        config.APPREHENDED,
        config.CBP_CUSTODY,
        config.TRANSFERS,
        config.HHS_CARE,
        config.DISCHARGED,
    ]:
        if c in df.columns:
            df[c] = df[c].map(_strip_number)
    df = df.sort_values(config.DATE).set_index(config.DATE)
    df = df[~df.index.duplicated(keep="last")]
    return df


def to_daily_continuous(df: pd.DataFrame, interpolate: bool = True) -> pd.DataFrame:
    """Reindex to calendar days; optional time interpolation for missing days."""
    if df.empty:
        return df
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    out = df.reindex(full)
    numeric = out.select_dtypes(include=[np.number]).columns
    if interpolate:
        out[numeric] = out[numeric].interpolate(method="time", limit_direction="both")
    out[numeric] = out[numeric].ffill().bfill()
    out.index.name = "Date"
    return out


def load_and_prepare(csv_path: str | Path, interpolate: bool = True) -> pd.DataFrame:
    raw = load_raw_csv(csv_path)
    return to_daily_continuous(raw, interpolate=interpolate)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    return out


def net_pressure(df: pd.DataFrame) -> pd.Series:
    """Transfers into HHS minus discharges (imbalance signal)."""
    return df[config.TRANSFERS].fillna(0) - df[config.DISCHARGED].fillna(0)
