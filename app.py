from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from src.data_prep import DISPLAY_NAMES, load_uac_data
from src.forecasting import MODEL_REGISTRY, make_forecast


st.set_page_config(page_title="Care Load and Placement Demand Forecasting", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_uac_data()


@st.cache_data
def get_forecast(target: str, horizon: int, model_name: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = make_forecast(get_data(), target=target, horizon=horizon, model_name=model_name)
    return result.forecast, result.metrics, result.backtest


def breach_probability(forecast: float, upper_95: float, capacity: float) -> float:
    sigma = max((upper_95 - forecast) / 1.96, 1e-6)
    z = (capacity - forecast) / sigma
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def line_chart(history: pd.DataFrame, forecast: pd.DataFrame, target: str) -> None:
    if go is None:
        chart_df = pd.concat(
            [
                history.reset_index(names="date").tail(180)[["date", target]].rename(columns={target: "Historical"}),
                forecast[["date", "forecast"]].rename(columns={"forecast": "Forecast"}),
            ],
            axis=0,
        )
        st.line_chart(chart_df, x="date")
        return

    fig = go.Figure()
    recent = history.tail(180)
    fig.add_trace(go.Scatter(x=recent.index, y=recent[target], name="Historical", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast"], name="Forecast", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"].iloc[::-1]]),
            y=pd.concat([forecast["upper_95"], forecast["lower_95"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(49, 130, 189, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% interval",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title=DISPLAY_NAMES[target],
        legend_orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)


df = get_data()

st.title("Predictive Forecasting of Care Load and Placement Demand")
st.caption(
    "An interactive forecasting dashboard for estimating future care load, discharge demand, "
    "and potential capacity stress."
)

with st.sidebar:
    st.header("Forecast Controls")
    horizon = st.slider("Forecast horizon in days", 7, 60, 30, step=1)
    target = st.radio(
        "Select forecast target",
        ["hhs_care_load", "hhs_discharges"],
        format_func=lambda x: DISPLAY_NAMES[x],
    )
    model_options = ["Automatically select best model"] + list(MODEL_REGISTRY.keys())
    model_choice = st.selectbox("Forecasting model", model_options)
    selected_model = None if model_choice == "Automatically select best model" else model_choice
    default_capacity = int(max(df["hhs_care_load"].tail(90).max() * 1.1, df["hhs_care_load"].iloc[-1] + 250))
    capacity = st.number_input("Care capacity threshold", min_value=0, value=default_capacity, step=50)

forecast, metrics, backtest = get_forecast(target, horizon, selected_model)
active_model = forecast["model"].iloc[0]
latest = df.iloc[-1]
last_forecast = forecast.iloc[-1]
prob = breach_probability(last_forecast["forecast"], last_forecast["upper_95"], capacity)
breach_rows = forecast[forecast["upper_95"] >= capacity]
lead_time = int(breach_rows["horizon_day"].iloc[0]) if not breach_rows.empty else None

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Latest care load", f"{latest['hhs_care_load']:,.0f}")
kpi2.metric(f"{horizon}-day forecast", f"{last_forecast['forecast']:,.0f}")
kpi3.metric("Capacity breach probability", f"{prob:.1%}")
kpi4.metric("Surge lead time", f"{lead_time} days" if lead_time else "No breach")

tab_forecast, tab_discharge, tab_compare, tab_scenario = st.tabs(
    ["Care Load Forecast", "Discharge Demand Forecast", "Model Comparison", "Capacity Scenario"]
)

with tab_forecast:
    st.subheader(DISPLAY_NAMES[target])
    line_chart(df, forecast, target)
    st.caption(
        f"Active model: {active_model}. The shaded area shows an empirical 95% forecast interval "
        "based on holdout residual variance."
    )

with tab_discharge:
    discharge_forecast, discharge_metrics, _ = get_forecast("hhs_discharges", horizon, None)
    st.subheader("Discharge Demand Forecast")
    line_chart(df, discharge_forecast, "hhs_discharges")
    st.dataframe(discharge_forecast.round(2), use_container_width=True, hide_index=True)

with tab_compare:
    st.subheader("Model Evaluation on the Final 90 Days")
    display_metrics = metrics[["model", "MAE", "RMSE", "MAPE", "Forecast Accuracy (%)", "Residual Std"]].copy()
    st.dataframe(display_metrics.round(2), use_container_width=True, hide_index=True)
    if go is not None:
        fig = go.Figure()
        for model in backtest["model"].unique():
            subset = backtest[backtest["model"] == model]
            fig.add_trace(go.Scatter(x=subset["date"], y=subset["predicted"], mode="lines", name=model))
        actual = backtest[backtest["model"] == backtest["model"].iloc[0]]
        fig.add_trace(go.Scatter(x=actual["date"], y=actual["actual"], mode="lines", name="Actual", line=dict(width=3)))
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=20, b=20), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(backtest.pivot(index="date", columns="model", values="predicted"))

with tab_scenario:
    st.subheader("Capacity Stress Scenario")
    scenario = forecast.copy()
    scenario["capacity"] = capacity
    scenario["net_gap_vs_capacity"] = capacity - scenario["forecast"]
    scenario["breach_probability"] = [
        breach_probability(row.forecast, row.upper_95, capacity) for row in scenario.itertuples(index=False)
    ]
    st.dataframe(
        scenario[["date", "forecast", "lower_95", "upper_95", "capacity", "net_gap_vs_capacity", "breach_probability"]].round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Scenario CSV",
        scenario.to_csv(index=False).encode("utf-8"),
        file_name="uac_forecast_scenario.csv",
        mime="text/csv",
    )
