# Predictive Forecasting of Care Load and Placement Demand

## Background

Care operations for unaccompanied children face volatile intake, transfer, and discharge patterns. Descriptive reporting explains historical movement, but operational leaders need forward-looking indicators for shelter utilization, medical staffing, caseworker load, and placement throughput.

## Data and Preparation

The source file contains daily observations for children apprehended and placed in United States Customs and Border Protection custody, children in Customs and Border Protection custody, children transferred out of Customs and Border Protection custody, children in Health and Human Services care, children discharged from Health and Human Services care, and transfers minus discharges. Blank trailing rows were removed, dates were parsed, duplicate dates were de-duplicated, and the series was reindexed to daily frequency. Missing dates were interpolated to support continuous forecasting while preserving a `was_reported` indicator.

## Feature Engineering

The modeling table includes lag features at 1, 7, 14, and 28 days; rolling means and standard deviations over 7, 14, and 28 days; net pressure defined as transfers to Health and Human Services care minus discharges; and calendar effects including day of week, weekend, and month cyclic encodings.

## Modeling Approach

Models compared include naive persistence, a 7-day moving average, seasonal naive forecasting, exponential smoothing, trend regression, and feature regression. The project uses a strict time-based 90-day holdout and reports mean absolute error, root mean squared error, mean absolute percentage error, forecast accuracy, and residual standard deviation for uncertainty bands.

## Results

The best 90-day holdout care-load model is **Random Forest Regressor**, with mean absolute error of 51.1, root mean squared error of 65.2, and mean absolute percentage error of 2.2%. The best discharge-demand model is **Feature Regression**, with mean absolute error of 3.2, root mean squared error of 4.0, and mean absolute percentage error of 47.3%.

## KPIs

- Forecast accuracy: computed as 100 minus mean absolute percentage error on the time-based holdout.
- Surge lead time: first forecast horizon where demand exceeds the selected capacity threshold.
- Capacity breach probability: normal approximation using model residual variance.
- Forecast stability index: suggested production metric tracking model-to-model and day-to-day forecast changes.

## Recommendations

Operational dashboards should refresh daily, compare multiple models, and expose uncertainty intervals. Capacity decisions should be tied to the upper forecast band when the cost of under-preparation is high.
