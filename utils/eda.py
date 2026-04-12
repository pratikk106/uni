"""Exploratory summaries and automated insight text for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import config


def core_numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        config.APPREHENDED,
        config.CBP_CUSTODY,
        config.TRANSFERS,
        config.HHS_CARE,
        config.DISCHARGED,
    ]
    return [c for c in cols if c in df.columns]


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    cols = core_numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T.round(2)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = core_numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr().round(3)


def build_insight_bullets(df: pd.DataFrame) -> list[str]:
    """Short, stakeholder-oriented bullets with concrete numbers from the loaded frame."""
    bullets: list[str] = []
    if df.empty:
        return ["No rows available after preparation."]

    idx = df.index
    bullets.append(
        f"**Coverage:** {len(df):,} daily observations from **{idx.min().date()}** to **{idx.max().date()}** "
        f"(missing calendar days were interpolated for continuity)."
    )

    hhs = df[config.HHS_CARE].dropna()
    if len(hhs) >= 30:
        last = float(hhs.iloc[-1])
        mean_all = float(hhs.mean())
        ma30 = float(hhs.iloc[-30:].mean())
        d30 = last - float(hhs.iloc[-30])
        bullets.append(
            f"**HHS care load:** latest **{last:,.0f}** children vs. full-sample mean **{mean_all:,.0f}**; "
            f"trailing 30-day average **{ma30:,.0f}**; change over last 30 days **{d30:+,.0f}**."
        )

    dis = df[config.DISCHARGED].dropna()
    if len(dis) >= 14:
        bullets.append(
            f"**Discharges (placements):** trailing 14-day mean **{dis.iloc[-14:].mean():.1f}** per day "
            f"(std **{dis.iloc[-14:].std():.1f}**), indicating short-term placement throughput variability."
        )

    tr = df[config.TRANSFERS].dropna()
    dch = df[config.DISCHARGED].fillna(0)
    if len(tr) >= 14 and len(dch) >= 14:
        net = (tr - dch).iloc[-14:]
        bullets.append(
            f"**Flow pressure (transfers − discharges), last 14 days:** mean **{net.mean():+.1f}** per day "
            f"(positive ⇒ more net inflow to HHS if other flows are stable)."
        )

    corr_cols = core_numeric_columns(df)
    if len(corr_cols) >= 2:
        c = df[corr_cols].corr()
        if config.TRANSFERS in c.columns and config.HHS_CARE in c.index:
            r = float(c.loc[config.HHS_CARE, config.TRANSFERS])
            if not np.isnan(r):
                bullets.append(
                    f"**Association:** same-period correlation (HHS care vs. daily transfers) ≈ **{r:.2f}** "
                    "(descriptive only; not causal)."
                )

    bullets.append(
        "**Forecasting note:** models use strict time-based validation; use confidence bands as **rough** "
        "uncertainty, not operational guarantees—especially under policy or border shocks."
    )
    return bullets
