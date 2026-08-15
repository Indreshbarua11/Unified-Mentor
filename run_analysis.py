from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_prep import DISPLAY_NAMES, load_uac_data
from src.forecasting import build_default_outputs


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"


def _fmt(value: float) -> str:
    return f"{value:,.1f}"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_csv(index=False), encoding="utf-8")


def write_reports(outputs: dict[str, pd.DataFrame]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    df = outputs["clean_daily"].copy()
    metrics = outputs["metrics"]
    forecasts = outputs["forecasts"]
    latest = df.iloc[-1]
    best_care = metrics[metrics["target"] == "hhs_care_load"].iloc[0]
    best_discharge = metrics[metrics["target"] == "hhs_discharges"].iloc[0]
    care_30 = forecasts[(forecasts["target"] == "hhs_care_load") & (forecasts["horizon_day"] == 30)].iloc[0]
    discharge_30 = forecasts[(forecasts["target"] == "hhs_discharges") & (forecasts["horizon_day"] == 30)].iloc[0]

    executive_summary = f"""# Executive Summary

## Predictive Forecasting of Care Load & Placement Demand

The UAC daily dataset was cleaned, regularized to a daily calendar, and used to compare baseline, statistical-style, and feature-based forecasting models for short-term HHS planning.

## Current Operating Picture

- Data coverage: {df['date'].min().date()} to {df['date'].max().date()}
- Latest reported HHS care load: {_fmt(latest['hhs_care_load'])} children
- Latest reported discharges: {_fmt(latest['hhs_discharges'])} children
- Latest net pressure: {_fmt(latest['net_pressure'])} transfers minus discharges

## Forecast Signals

- Best care-load model in the 90-day holdout: {best_care['model']} with MAE {_fmt(best_care['MAE'])}
- Best discharge-demand model in the 90-day holdout: {best_discharge['model']} with MAE {_fmt(best_discharge['MAE'])}
- 30-day HHS care-load forecast: {_fmt(care_30['forecast'])} children, 95% interval {_fmt(care_30['lower_95'])} to {_fmt(care_30['upper_95'])}
- 30-day discharge-demand forecast: {_fmt(discharge_30['forecast'])} children/day, 95% interval {_fmt(discharge_30['lower_95'])} to {_fmt(discharge_30['upper_95'])}

## Planning Recommendations

- Use the care-load forecast as the primary shelter and staffing trigger.
- Use the discharge-demand forecast to plan case management, sponsor vetting, legal support, transport, and post-release coordination.
- Review breach probability daily against a configurable operating capacity threshold.
- Treat prediction intervals as the operational planning range, not just the point forecast.
"""

    research_paper = f"""# Predictive Forecasting of Care Load & Placement Demand

## Background

HHS care operations for unaccompanied children face volatile intake, transfer, and discharge patterns. Descriptive reporting explains historical movement, but operational leaders need forward-looking indicators for shelter utilization, medical staffing, caseworker load, and placement throughput.

## Data and Preparation

The source file contains daily observations for {', '.join(DISPLAY_NAMES.values())}. Blank trailing rows were removed, dates were parsed, duplicate dates were de-duplicated, and the series was reindexed to daily frequency. Missing dates were interpolated to support continuous forecasting while preserving a `was_reported` indicator.

## Feature Engineering

The modeling table includes lag features at 1, 7, 14, and 28 days; rolling means and standard deviations over 7, 14, and 28 days; net pressure defined as transfers to HHS minus discharges; and calendar effects including day of week, weekend, and month cyclic encodings.

## Modeling Approach

Models compared include naive persistence, 7-day moving average, seasonal naive, exponential smoothing, trend regression, and feature regression. The project uses a strict time-based 90-day holdout and reports MAE, RMSE, MAPE, forecast accuracy, and residual standard deviation for uncertainty bands.

## Results

The best 90-day holdout care-load model is **{best_care['model']}**, with MAE {_fmt(best_care['MAE'])}, RMSE {_fmt(best_care['RMSE'])}, and MAPE {_fmt(best_care['MAPE'])}%. The best discharge-demand model is **{best_discharge['model']}**, with MAE {_fmt(best_discharge['MAE'])}, RMSE {_fmt(best_discharge['RMSE'])}, and MAPE {_fmt(best_discharge['MAPE'])}%.

## KPIs

- Forecast accuracy: computed as 100 minus MAPE on the time-based holdout.
- Surge lead time: first forecast horizon where demand exceeds the selected capacity threshold.
- Capacity breach probability: normal approximation using model residual variance.
- Forecast stability index: suggested production metric tracking model-to-model and day-to-day forecast changes.

## Recommendations

Operational dashboards should refresh daily, compare multiple models, and expose uncertainty intervals. Capacity decisions should be tied to the upper forecast band when the cost of under-preparation is high.
"""

    (REPORT_DIR / "executive_summary.md").write_text(executive_summary, encoding="utf-8")
    (REPORT_DIR / "research_paper.md").write_text(research_paper, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = build_default_outputs(horizon=30)
    for name, frame in outputs.items():
        write_csv(frame, OUTPUT_DIR / f"{name}.csv")
    write_reports(outputs)
    print("Wrote outputs/clean_daily.csv, outputs/forecasts.csv, outputs/metrics.csv, outputs/backtest.csv")
    print("Wrote reports/executive_summary.md and reports/research_paper.md")


if __name__ == "__main__":
    main()
