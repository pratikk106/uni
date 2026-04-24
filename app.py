"""
Streamlit dashboard: UAC care load & discharge demand forecasting.
Run: streamlit run app.py
"""

from __future__ import annotations

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
from data.loader import load_and_prepare, load_raw_csv, net_pressure
from utils.eda import build_insight_bullets, correlation_matrix, summary_statistics
from model.forecasting import (
    ModelName,
    decomposition_plotly_df,
    run_model_forecast,
    train_ml_model,
    walk_forward_scores,
)

CSV_DEFAULT_NAME = "HHS_Unaccompanied_Alien_Children_Program (1).csv"
CSV_DEFAULT_LOCATIONS = [ROOT, ROOT.parent, ROOT.parent.parent]


def resolve_default_csv() -> Path:
    for folder in CSV_DEFAULT_LOCATIONS:
        candidate = folder / CSV_DEFAULT_NAME
        if candidate.exists():
            return candidate
    return ROOT / CSV_DEFAULT_NAME


CSV_DEFAULT = resolve_default_csv()

MODEL_LABELS: dict[str, str] = {
    "naive": "Naïve (last value)",
    "ma_7": "Moving average (7d)",
    "sarima": "SARIMA (weekly seasonality)",
    "ets": "Exponential smoothing (Holt–Winters)",
}


@st.cache_data(show_spinner=False)
def get_prepared_data(csv_path: str, data_fingerprint: str, prep_mode: str) -> pd.DataFrame:
    """`data_fingerprint` must change when the file contents change (upload or disk edit)."""
    _ = data_fingerprint
    if prep_mode == "Continuous daily + interpolation":
        return load_and_prepare(csv_path, interpolate=True)
    # Default: use only cleaned raw-valid rows.
    return load_raw_csv(csv_path)


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


@st.cache_data(show_spinner=False)
def cached_model_scores(path: str, target: str, model: str, data_fingerprint: str) -> dict:
    df = get_prepared_data(path, data_fingerprint, prep_mode="Raw valid rows (no interpolation)")
    return walk_forward_scores(df, target, model, min_train=min(300, max(120, len(df) // 2)))


def main():
    st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
    st.title("Predictive forecasting: care load & placement demand")
    st.caption(
        "Short-term forecasts for children in HHS care and discharges, with model comparison "
        "and approximate uncertainty bands where available."
    )

    csv_path = str(CSV_DEFAULT)
    p = Path(csv_path)
    if p.exists():
        data_fingerprint = f"default:{p.stat().st_mtime_ns}:{p.stat().st_size}"
    else:
        data_fingerprint = "default:missing"

    with st.sidebar:
        st.header("Data")
        st.info("Using fixed project CSV", icon="📁")
        st.caption(f"`{CSV_DEFAULT.name}`")
        if not Path(csv_path).exists():
            st.error(f"CSV not found: {csv_path}")
            st.stop()

        st.header("Methodology mode")
        prep_mode = st.selectbox(
            "Time-series preparation",
            ["Raw valid rows (no interpolation)", "Continuous daily + interpolation"],
            index=0,
            help="Raw valid rows uses only cleaned dated rows. Continuous daily expands missing dates and interpolates values.",
        )

        st.header("Controls")
        horizon = st.slider("Forecast horizon (days)", 1, 42, 14)
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
            index=3,
            format_func=lambda k: MODEL_LABELS[k],
        )

    # Data coverage counters for transparent preprocessing diagnostics.
    try:
        raw_csv_rows = int(len(pd.read_csv(csv_path)))
    except Exception:
        raw_csv_rows = 0
    try:
        valid_dated_rows = int(len(load_raw_csv(csv_path)))
    except Exception:
        valid_dated_rows = 0
    df = get_prepared_data(csv_path, data_fingerprint, prep_mode)
    modeled_days = int(len(df))

    st.divider()
    ind1, ind2, ind3, ind4 = st.columns(4)
    ind1.markdown(":blue[**Data source**]  \nFixed dataset")
    ind2.markdown(f"**File**  \n`{CSV_DEFAULT.name}`")
    ind3.metric("Rows loaded", f"{modeled_days:,}")
    ind4.metric("Date span", f"{df.index.min().date()} → {df.index.max().date()}")
    st.caption("Charts and models below use this fixed dataset file and the selected methodology mode.")
    st.markdown("#### Data pipeline counts")
    c1, c2 = st.columns(2)
    c1.metric("Valid dated rows", f"{valid_dated_rows:,}")
    c2.metric("Final modeling rows", f"{modeled_days:,}")
    st.caption(f"Active preparation mode: `{prep_mode}`")
    st.divider()

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "EDA & insights",
            "Forecasts",
            "Model comparison",
            "Decomposition & flow",
            "Early warnings",
            "ML vs statistical",
            "Methodology mode",
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
        m1.metric("Observations (rows)", f"{len(df):,}")
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

        fc_hhs = run_model_forecast(
            df, config.HHS_CARE, model_hhs, horizon, ml_bundle=None
        )
        fc_dis = run_model_forecast(
            df, config.DISCHARGED, model_dis, horizon, ml_bundle=None
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
        fa = run_model_forecast(df, config.HHS_CARE, compare_a, horizon, ml_bundle=None)
        fb = run_model_forecast(df, config.HHS_CARE, compare_b, horizon, ml_bundle=None)

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
        st.subheader("Early-warning indicators (non-training)")
        st.caption(
            "Rule-based alerts from short-term forecasts to support proactive shelter and staffing decisions."
        )

        w1, w2 = st.columns(2)
        with w1:
            warning_model: ModelName = st.selectbox(
                "Warning forecast model",
                list(MODEL_LABELS.keys()),
                index=2,
                format_func=lambda k: MODEL_LABELS[k],
                key="warn_model",
            )
        with w2:
            breach_level = st.number_input(
                "Capacity stress threshold (HHS care count)",
                min_value=0,
                value=int(2600),
                step=50,
                key="warn_breach_threshold",
            )

        fc_hhs_warn = run_model_forecast(df, config.HHS_CARE, warning_model, horizon, ml_bundle=None)
        fc_dis_warn = run_model_forecast(df, config.DISCHARGED, warning_model, horizon, ml_bundle=None)

        hhs_fc = pd.Series(fc_hhs_warn.point, index=fc_hhs_warn.forecast_index, name="hhs_care_fc")
        dis_fc = pd.Series(fc_dis_warn.point, index=fc_dis_warn.forecast_index, name="discharge_fc")
        transfer_assumption = float(df[config.TRANSFERS].iloc[-1])
        net_pressure_fc = transfer_assumption - dis_fc

        above = hhs_fc[hhs_fc > float(breach_level)]
        breach_prob_14d = float((hhs_fc > float(breach_level)).mean())
        days_to_breach = int((above.index[0] - hhs_fc.index[0]).days + 1) if not above.empty else None
        net_inflow_share = float((net_pressure_fc > 0).mean())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Breach risk (horizon)",
            f"{100 * breach_prob_14d:.1f}%",
            help="Share of forecast days above the capacity threshold.",
        )
        k2.metric(
            "Lead time to breach",
            f"{days_to_breach} days" if days_to_breach is not None else "No breach in horizon",
            help="Days until first forecasted threshold crossing.",
        )
        k3.metric(
            "Net inflow risk days",
            f"{100 * net_inflow_share:.1f}%",
            help="Share of forecast days where transfers exceed discharges.",
        )
        k4.metric(
            "Transfer assumption",
            f"{transfer_assumption:,.0f}/day",
            help="Held constant at the latest observed transfers value.",
        )

        warn_df = pd.DataFrame(
            {
                "HHS care forecast": hhs_fc.round(1),
                "Discharge forecast": dis_fc.round(1),
                "Transfer assumption": transfer_assumption,
                "Net pressure (transfer - discharge)": net_pressure_fc.round(1),
                "Capacity breach": (hhs_fc > float(breach_level)),
            }
        )
        st.dataframe(warn_df, use_container_width=True)

    with tab5:
        st.subheader("ML vs statistical comparison (optional)")
        st.caption(
            "This section is isolated from the main flow and trains on the same 720 valid rows currently loaded."
        )
        run_ml_compare = st.button(
            "Run ML comparison now",
            key="run_ml_compare_btn",
            help="Runs training and walk-forward metrics only when you click.",
        )
        compare_labels = {
            "naive": "Naïve (last value)",
            "ma_7": "Moving average (7d)",
            "sarima": "SARIMA (weekly seasonality)",
            "ets": "Exponential smoothing (Holt–Winters)",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
        }
        c1, c2, c3 = st.columns(3)
        with c1:
            cmp_target = st.selectbox(
                "Comparison target",
                [config.HHS_CARE, config.DISCHARGED],
                index=0,
                key="cmp_target",
            )
        with c2:
            cmp_a = st.selectbox(
                "Model A",
                list(compare_labels.keys()),
                index=2,
                format_func=lambda k: compare_labels[k],
                key="cmp_a",
            )
        with c3:
            cmp_b = st.selectbox(
                "Model B",
                list(compare_labels.keys()),
                index=4,
                format_func=lambda k: compare_labels[k],
                key="cmp_b",
            )

        if not run_ml_compare:
            st.info("Click **Run ML comparison now** to generate charts and metrics.")
        else:
            ml_a = (
                train_ml_model(df, cmp_target, cmp_a)
                if cmp_a in ("random_forest", "gradient_boosting")
                else None
            )
            ml_b = (
                train_ml_model(df, cmp_target, cmp_b)
                if cmp_b in ("random_forest", "gradient_boosting")
                else None
            )
            fa = run_model_forecast(df, cmp_target, cmp_a, horizon, ml_bundle=ml_a)
            fb = run_model_forecast(df, cmp_target, cmp_b, horizon, ml_bundle=ml_b)

            figm = go.Figure()
            h = fa.history.tail(365)
            figm.add_trace(go.Scatter(x=h.index, y=h.values, name="History", line=dict(color="#64748b")))
            figm.add_trace(
                go.Scatter(
                    x=fa.forecast_index,
                    y=fa.point,
                    name=compare_labels[cmp_a],
                    line=dict(color="#2563eb", dash="dash"),
                )
            )
            figm.add_trace(
                go.Scatter(
                    x=fb.forecast_index,
                    y=fb.point,
                    name=compare_labels[cmp_b],
                    line=dict(color="#ea580c", dash="dot"),
                )
            )
            figm.update_layout(height=420, hovermode="x unified", title="Forecast comparison")
            st.plotly_chart(figm, use_container_width=True)

            s_a = cached_model_scores(csv_path, cmp_target, cmp_a, data_fingerprint)
            s_b = cached_model_scores(csv_path, cmp_target, cmp_b, data_fingerprint)
            rows = []
            for h_ in (1, 7, 14):
                ma = s_a.get(h_, {})
                mb = s_b.get(h_, {})
                rows.append(
                    {
                        "Horizon (days)": h_,
                        "A model": compare_labels[cmp_a],
                        "A MAE": round(ma.get("mae", np.nan), 2),
                        "A RMSE": round(ma.get("rmse", np.nan), 2),
                        "A MAPE %": round(ma.get("mape", np.nan), 2),
                        "B model": compare_labels[cmp_b],
                        "B MAE": round(mb.get("mae", np.nan), 2),
                        "B RMSE": round(mb.get("rmse", np.nan), 2),
                        "B MAPE %": round(mb.get("mape", np.nan), 2),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab6:
        st.subheader("Methodology mode")
        st.caption("Separate section to control and explain time-series preparation behavior.")
        i1, i2 = st.columns(2)
        i1.metric("Current mode", prep_mode)
        i2.metric("Modeling rows in this mode", f"{modeled_days:,}")
        st.markdown("#### Dataset used in this mode")
        data_dict = pd.DataFrame(
            [
                {"Column": "Date", "Description": "Reporting date"},
                {
                    "Column": "Children apprehended and placed in CBP custody*",
                    "Description": "Daily intake volume",
                },
                {"Column": "Children in CBP custody", "Description": "Active CBP care load"},
                {
                    "Column": "Children transferred out of CBP custody",
                    "Description": "Flow into HHS system",
                },
                {"Column": "Children in HHS Care", "Description": "Active HHS care load"},
                {
                    "Column": "Children discharged from HHS Care",
                    "Description": "Successful sponsor placements",
                },
            ]
        )
        st.dataframe(data_dict, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("CSV file", CSV_DEFAULT.name)
        c2.metric("Valid dated rows", f"{valid_dated_rows:,}")
        c3.metric("Rows after selected mode", f"{modeled_days:,}")
        st.markdown(
            "- **Raw valid rows (no interpolation):** uses only cleaned dated observations from the CSV.\n"
            "- **Continuous daily + interpolation:** reindexes to daily calendar continuity and interpolates missing days."
        )

if __name__ == "__main__":
    main()
