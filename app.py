"""
Streamlit dashboard: UAC care load & discharge demand forecasting.
Run: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import config
from data.loader import load_and_prepare, net_pressure
from utils.eda import build_insight_bullets, correlation_matrix, summary_statistics
from model.forecasting import (
    ModelName,
    decomposition_plotly_df,
    kpi_bundle,
    run_model_forecast,
    train_ml_model,
    walk_forward_scores,
)

CSV_DEFAULT = ROOT / "HHS_Unaccompanied_Alien_Children_Program.csv"

MODEL_LABELS: dict[str, str] = {
    "naive": "Naïve (last value)",
    "ma_7": "Moving average (7d)",
    "sarima": "SARIMA (weekly seasonality)",
    "ets": "Exponential smoothing (Holt–Winters)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


@st.cache_data(show_spinner=False)
def get_prepared_data(csv_path: str, data_fingerprint: str) -> pd.DataFrame:
    """`data_fingerprint` must change when the file contents change (upload or disk edit)."""
    _ = data_fingerprint
    return load_and_prepare(csv_path, interpolate=True)


@st.cache_data(show_spinner=False)
def cached_walk_forward(path: str, target: str, model: str, data_fingerprint: str) -> dict:
    df = get_prepared_data(path, data_fingerprint)
    return walk_forward_scores(df, target, model, min_train=min(400, max(100, len(df) // 2)))


def fig_forecast(
    hist: pd.Series,
    fc_dates: pd.DatetimeIndex,
    point: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            name="History",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=point,
            name="Forecast",
            line=dict(color="#dc2626", width=2, dash="dash"),
        )
    )
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=list(fc_dates) + list(fc_dates)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself",
                fillcolor="rgba(220,38,38,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="~80% interval",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=48, r=24, t=56, b=48),
    )
    fig.update_yaxes(title_text="Children")
    fig.update_xaxes(title_text="Date")
    return fig


def main():
    st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
    st.title("Predictive forecasting: care load & placement demand")
    st.caption(
        "Short-term forecasts for children in HHS care and discharges, with model comparison "
        "and approximate uncertainty bands where available."
    )

    using_upload = False
    upload_display_name: str | None = None
    data_fingerprint = ""

    with st.sidebar:
        st.header("Data")
        upload = st.file_uploader("CSV (optional)", type=["csv"])
        if upload is not None:
            raw = upload.getvalue()
            up_path = ROOT / "_streamlit_upload.csv"
            up_path.write_bytes(raw)
            csv_path = str(up_path)
            using_upload = True
            upload_display_name = upload.name
            data_fingerprint = f"upload:{hashlib.sha256(raw).hexdigest()}"
            st.success("Uploaded file is active", icon="✅")
            st.caption(f"**{upload.name}** · {len(raw):,} bytes saved")
        else:
            csv_path = str(CSV_DEFAULT)
            p = Path(csv_path)
            if p.exists():
                data_fingerprint = f"default:{p.stat().st_mtime_ns}:{p.stat().st_size}"
            else:
                data_fingerprint = "default:missing"
            st.info("Using default project CSV", icon="📁")
            st.caption(f"`{CSV_DEFAULT.name}`")
        if not Path(csv_path).exists():
            st.error(f"CSV not found: {csv_path}")
            st.stop()

        st.header("Controls")
        horizon = st.slider("Forecast horizon (days)", 1, 42, 14)
        breach_level = st.number_input(
            "Capacity stress threshold (HHS care count)",
            min_value=0,
            value=int(2600),
            step=50,
            help="Used for breach probability KPI.",
        )
        st.subheader("Models for scenario comparison")
        compare_a: ModelName = st.selectbox(
            "Scenario A",
            list(MODEL_LABELS.keys()),
            index=2,
            format_func=lambda k: MODEL_LABELS[k],
        )
        compare_b: ModelName = st.selectbox(
            "Scenario B",
            list(MODEL_LABELS.keys()),
            index=5,
            format_func=lambda k: MODEL_LABELS[k],
        )

    df = get_prepared_data(csv_path, data_fingerprint)

    st.divider()
    ind1, ind2, ind3, ind4 = st.columns(4)
    if using_upload:
        ind1.markdown(":green[**Data source**]  \nUploaded CSV")
        ind2.markdown(f"**File**  \n`{upload_display_name}`")
    else:
        ind1.markdown(":blue[**Data source**]  \nBundled default")
        ind2.markdown(f"**File**  \n`{CSV_DEFAULT.name}`")
    ind3.metric("Rows loaded", f"{len(df):,}")
    ind4.metric("Date span", f"{df.index.min().date()} → {df.index.max().date()}")
    st.caption(
        "Charts and models below use this dataset. Change the upload or replace the default CSV on disk, "
        "then the app refreshes automatically."
    )
    st.divider()

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "EDA & insights",
            "Forecasts",
            "Model comparison",
            "Decomposition & flow",
            "Evaluation & KPIs",
        ]
    )

    with tab0:
        st.subheader("Exploratory data analysis (live)")
        st.caption("Statistics and charts update from the dataset selected in the sidebar.")

        cols_num = [
            c
            for c in [
                config.HHS_CARE,
                config.TRANSFERS,
                config.DISCHARGED,
                config.CBP_CUSTODY,
                config.APPREHENDED,
            ]
            if c in df.columns
        ]

        m1, m2, m3 = st.columns(3)
        m1.metric("Observations (days)", f"{len(df):,}")
        m2.metric("Start date", str(df.index.min().date()))
        m3.metric("End date", str(df.index.max().date()))

        st.markdown("#### Summary statistics")
        st.dataframe(summary_statistics(df), use_container_width=True)

        st.markdown("#### Key series (standardized for comparison)")
        zdf = pd.DataFrame(index=df.index)
        for c in cols_num:
            s = df[c].astype(float)
            std = float(s.std()) or 1.0
            zdf[c[:22] + "…" if len(c) > 22 else c] = (s - s.mean()) / std
        figz = go.Figure()
        for c in zdf.columns:
            figz.add_trace(go.Scatter(x=zdf.index, y=zdf[c], name=c, mode="lines", opacity=0.85))
        figz.update_layout(
            height=400,
            hovermode="x unified",
            yaxis_title="Z-score",
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(t=48),
        )
        st.plotly_chart(figz, use_container_width=True)

        st.markdown("#### Correlation matrix (numeric program columns)")
        cm = correlation_matrix(df)
        if not cm.empty:
            fig_h = go.Figure(
                data=go.Heatmap(
                    z=cm.values,
                    x=[x[:18] + "…" if len(x) > 18 else x for x in cm.columns],
                    y=[y[:18] + "…" if len(y) > 18 else y for y in cm.index],
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="r"),
                )
            )
            fig_h.update_layout(height=420, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Not enough numeric columns for a correlation matrix.")

        st.markdown("#### Automated insights (from current data)")
        for b in build_insight_bullets(df):
            st.markdown(f"- {b}")

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Children in HHS care")
            model_hhs: ModelName = st.selectbox(
                "Model (HHS care)",
                list(MODEL_LABELS.keys()),
                index=2,
                format_func=lambda k: MODEL_LABELS[k],
                key="m_hhs",
            )
        with col2:
            st.subheader("Discharges (placement demand)")
            model_dis: ModelName = st.selectbox(
                "Model (discharges)",
                list(MODEL_LABELS.keys()),
                index=2,
                format_func=lambda k: MODEL_LABELS[k],
                key="m_dis",
            )

        ml_hhs = None
        ml_dis = None
        if model_hhs in ("random_forest", "gradient_boosting"):
            ml_hhs = train_ml_model(df, config.HHS_CARE, model_hhs)
        if model_dis in ("random_forest", "gradient_boosting"):
            ml_dis = train_ml_model(df, config.DISCHARGED, model_dis)

        fc_hhs = run_model_forecast(
            df, config.HHS_CARE, model_hhs, horizon, ml_bundle=ml_hhs
        )
        fc_dis = run_model_forecast(
            df, config.DISCHARGED, model_dis, horizon, ml_bundle=ml_dis
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                fig_forecast(
                    fc_hhs.history.tail(365),
                    fc_hhs.forecast_index,
                    fc_hhs.point,
                    fc_hhs.lower,
                    fc_hhs.upper,
                    "HHS care load — last 365 days + forecast",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                fig_forecast(
                    fc_dis.history.tail(365),
                    fc_dis.forecast_index,
                    fc_dis.point,
                    fc_dis.lower,
                    fc_dis.upper,
                    "Discharges — last 365 days + forecast",
                ),
                use_container_width=True,
            )

        imbalance = (
            pd.Series(fc_dis.point, index=fc_dis.forecast_index, name="discharge_fc")
            .to_frame()
        )
        # transfers forward-filled from last known
        last_t = float(df[config.TRANSFERS].iloc[-1])
        imbalance["transfer_assumption"] = last_t
        imbalance["net_pressure_fc"] = imbalance["transfer_assumption"] - imbalance["discharge_fc"]
        st.subheader("Imbalance indicator (forecast period)")
        st.caption("Transfers assumed flat at last observed day; discharge uses model forecast.")
        st.dataframe(imbalance.round(1), use_container_width=True)

    with tab2:
        st.subheader("Scenario comparison (HHS care)")
        mla = train_ml_model(df, config.HHS_CARE, compare_a) if compare_a in (
            "random_forest",
            "gradient_boosting",
        ) else None
        mlb = train_ml_model(df, config.HHS_CARE, compare_b) if compare_b in (
            "random_forest",
            "gradient_boosting",
        ) else None
        fa = run_model_forecast(df, config.HHS_CARE, compare_a, horizon, ml_bundle=mla)
        fb = run_model_forecast(df, config.HHS_CARE, compare_b, horizon, ml_bundle=mlb)

        figc = go.Figure()
        h = fa.history.tail(365)
        figc.add_trace(
            go.Scatter(x=h.index, y=h.values, name="History", line=dict(color="#64748b"))
        )
        figc.add_trace(
            go.Scatter(
                x=fa.forecast_index,
                y=fa.point,
                name=MODEL_LABELS[compare_a],
                line=dict(color="#2563eb", dash="dash"),
            )
        )
        figc.add_trace(
            go.Scatter(
                x=fb.forecast_index,
                y=fb.point,
                name=MODEL_LABELS[compare_b],
                line=dict(color="#ea580c", dash="dot"),
            )
        )
        if fa.lower is not None and fa.upper is not None:
            figc.add_trace(
                go.Scatter(
                    x=list(fa.forecast_index) + list(fa.forecast_index)[::-1],
                    y=list(fa.upper) + list(fa.lower)[::-1],
                    fill="toself",
                    fillcolor="rgba(37,99,235,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="A: interval",
                    showlegend=True,
                )
            )
        figc.update_layout(
            title="HHS care — two scenarios",
            height=440,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(figc, use_container_width=True)

    with tab3:
        st.subheader("Seasonal decomposition (additive, 7-day period)")
        dec_df = decomposition_plotly_df(df[config.HHS_CARE])
        if dec_df is not None:
            figd = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for i, col in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
                figd.add_trace(
                    go.Scatter(x=dec_df.index, y=dec_df[col], name=col, showlegend=False),
                    row=i,
                    col=1,
                )
            figd.update_layout(height=720, title_text="HHS care load decomposition")
            st.plotly_chart(figd, use_container_width=True)
        else:
            st.info("Not enough data for decomposition.")

        st.subheader("Net pressure: transfers − discharges")
        np_ = net_pressure(df)
        fign = go.Figure()
        fign.add_trace(go.Scatter(x=np_.index, y=np_.values, name="Net pressure", line=dict(color="#0f766e")))
        fign.update_layout(height=360, title="Historical net inflow pressure")
        st.plotly_chart(fign, use_container_width=True)

    with tab4:
        st.subheader("Walk-forward validation (time-based)")
        eval_model: ModelName = st.selectbox(
            "Model to evaluate",
            list(MODEL_LABELS.keys()),
            index=2,
            format_func=lambda k: MODEL_LABELS[k],
            key="eval_m",
        )
        wf = cached_walk_forward(csv_path, config.HHS_CARE, eval_model, data_fingerprint)
        rows = []
        for h, m in sorted(wf.items()):
            rows.append(
                {
                    "Horizon (days)": h,
                    "MAE": round(m["mae"], 2),
                    "RMSE": round(m["rmse"], 2),
                    "MAPE %": round(m["mape"], 2),
                    "n": m["n"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        kpis = kpi_bundle(
            df,
            config.HHS_CARE,
            eval_model,
            breach_level=float(breach_level),
            walk_forward=wf,
        )
        k1, k2, k3, k4 = st.columns(4)
        acc = kpis["forecast_accuracy_pct"]
        k1.metric(
            "Forecast accuracy (%)",
            f"{acc:.1f}" if acc is not None and not (isinstance(acc, float) and np.isnan(acc)) else "—",
        )
        k2.metric(
            "Surge lead time (days)",
            f"{kpis['surge_lead_time_days']}" if kpis["surge_lead_time_days"] is not None else "—",
            help="Days until SARIMA forecast crosses historical 90th percentile.",
        )
        k3.metric(
            "Breach prob. (14d)",
            f"{kpis['capacity_breach_prob_14d']:.2f}" if kpis["capacity_breach_prob_14d"] is not None else "—",
        )
        k4.metric(
            "Stability index",
            f"{kpis['forecast_stability_index']:.3f}" if kpis["forecast_stability_index"] is not None else "—",
        )


if __name__ == "__main__":
    main()
