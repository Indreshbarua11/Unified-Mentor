# Predictive Forecasting of Care Load and Placement Demand

This project converts the United States Department of Health and Human Services daily reporting dataset for unaccompanied children into short-term operational forecasts for care load, discharge demand, and capacity stress risk.

## Project Structure

- `data/raw/HHS_Unaccompanied_Alien_Children_Program.csv` - source dataset used by the project
- `src/data_prep.py` - cleaning, daily continuity, and feature engineering
- `src/forecasting.py` - baseline, statistical-style, and feature-based forecasting models
- `run_analysis.py` - reproducible batch pipeline for metrics, forecasts, and reports
- `app.py` - Streamlit dashboard for model comparison and scenario analysis
- `outputs/` - generated clean data, forecasts, backtests, and metrics
- `reports/` - generated executive summary and research paper

## Quick Start

```bash
pip install -r requirements.txt
python run_analysis.py
streamlit run app.py
```

## Methods Included

- Daily time-series preparation with blank-row removal, date parsing, duplicate handling, and interpolation for missing dates
- Lag features at 1, 7, 14, and 28 days
- Rolling mean and volatility over 7, 14, and 28 days
- Net pressure signal: transfers to Health and Human Services care minus discharges from care
- Calendar effects: day of week, month, weekend, and cyclic encodings
- Strict 90-day time-based holdout evaluation
- Forecast uncertainty bands from holdout residual variance
- Capacity breach probability under a configurable care-load threshold

## Dashboard Modules

- Future care load forecast chart
- Discharge demand forecast panel
- Model selection and comparison
- Confidence interval visualization
- Forecast horizon selector
- Scenario comparison against capacity threshold
