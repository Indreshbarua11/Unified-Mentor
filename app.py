from __future__ import annotations

import math

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from src.data_prep import load_uac_data
from src.forecasting import MODEL_REGISTRY, make_forecast


TARGET_LABELS = {
    "hhs_care_load": "Care Load",
    "hhs_discharges": "Discharge Demand",
}


st.set_page_config(page_title="Care Load Forecasting App", layout="wide")


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
        yaxis_title=TARGET_LABELS[target],
        legend_orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)


df = get_data()

st.title("Predictive Forecasting of Care Load and Placement Demand")
st.write(
    "This simple app forecasts future care load and discharge demand. It helps users understand "
    "expected demand, compare forecasting models, and check whether the forecast may cross a selected capacity limit."
)

with st.sidebar:
    st.header("Settings")
    target = st.radio(
        "What do you want to forecast?",
        ["hhs_care_load", "hhs_discharges"],
        format_func=lambda x: TARGET_LABELS[x],
    )
    horizon = st.slider("Forecast days", 7, 60, 30, step=1)
    model_options = ["Automatically select best model"] + list(MODEL_REGISTRY.keys())
    model_choice = st.selectbox("Model", model_options)
    selected_model = None if model_choice == "Automatically select best model" else model_choice
    default_capacity = int(max(df["hhs_care_load"].tail(90).max() * 1.1, df["hhs_care_load"].iloc[-1] + 250))
    capacity = st.number_input("Capacity limit", min_value=0, value=default_capacity, step=50)

forecast, metrics, backtest = get_forecast(target, horizon, selected_model)
active_model = forecast["model"].iloc[0]
latest = df.iloc[-1]
last_forecast = forecast.iloc[-1]
prob = breach_probability(last_forecast["forecast"], last_forecast["upper_95"], capacity)
breach_rows = forecast[forecast["upper_95"] >= capacity]
lead_time = int(breach_rows["horizon_day"].iloc[0]) if not breach_rows.empty else None

st.subheader("Project Overview")
overview_1, overview_2, overview_3 = st.columns(3)
overview_1.metric("Data start date", df.index.min().strftime("%Y-%m-%d"))
overview_2.metric("Latest date", df.index.max().strftime("%Y-%m-%d"))
overview_3.metric("Selected model", active_model)

st.subheader("Key Forecast Results")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Latest care load", f"{latest['hhs_care_load']:,.0f}")
kpi2.metric(f"{horizon}-day forecast", f"{last_forecast['forecast']:,.0f}")
kpi3.metric("Capacity risk", f"{prob:.1%}")
kpi4.metric("First risk day", f"Day {lead_time}" if lead_time else "No risk")

st.subheader(f"{TARGET_LABELS[target]} Forecast")
line_chart(df, forecast, target)
st.caption(
    f"The chart uses the {active_model} model. The shaded area shows the estimated 95% forecast range."
)

st.subheader("Forecast Table")
forecast_table = forecast[["date", "forecast", "lower_95", "upper_95"]].copy()
forecast_table.columns = ["Date", "Forecast", "Lower estimate", "Upper estimate"]
st.dataframe(forecast_table.round(2), use_container_width=True, hide_index=True)

st.subheader("Model Comparison")
display_metrics = metrics[["model", "MAE", "RMSE", "MAPE", "Forecast Accuracy (%)"]].copy()
display_metrics.columns = ["Model", "MAE", "RMSE", "MAPE", "Forecast Accuracy (%)"]
st.dataframe(display_metrics.round(2), use_container_width=True, hide_index=True)

st.subheader("Capacity Check")
scenario = forecast.copy()
scenario["Capacity limit"] = capacity
scenario["Gap to capacity"] = capacity - scenario["forecast"]
scenario["Risk probability"] = [
    breach_probability(row.forecast, row.upper_95, capacity) for row in scenario.itertuples(index=False)
]
scenario_table = scenario[["date", "forecast", "Capacity limit", "Gap to capacity", "Risk probability"]].copy()
scenario_table.columns = ["Date", "Forecast", "Capacity limit", "Gap to capacity", "Risk probability"]
st.dataframe(scenario_table.round(3), use_container_width=True, hide_index=True)

st.download_button(
    "Download Forecast CSV",
    scenario.to_csv(index=False).encode("utf-8"),
    file_name="care_load_forecast.csv",
    mime="text/csv",
)
