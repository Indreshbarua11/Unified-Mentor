# Executive Summary

## Predictive Forecasting of Care Load and Placement Demand

The daily dataset for unaccompanied children was cleaned, regularized to a daily calendar, and used to compare baseline, statistical-style, and feature-based forecasting models for short-term care planning.

## Current Operating Picture

- Data coverage: 2023-01-12 to 2025-12-21
- Latest reported care load: 2,484.0 children
- Latest reported discharges: 14.0 children
- Latest net pressure: -3.0 transfers minus discharges

## Forecast Signals

- Best care-load model in the 90-day holdout: Random Forest Regressor with mean absolute error of 51.1
- Best discharge-demand model in the 90-day holdout: Feature Regression with mean absolute error of 3.2
- 30-day care-load forecast: 2,552.1 children, 95% interval 1,882.4 to 3,221.9
- 30-day discharge-demand forecast: 13.6 children/day, 95% interval 0.0 to 56.3

## Planning Recommendations

- Use the care-load forecast as the primary shelter and staffing trigger.
- Use the discharge-demand forecast to plan case management, sponsor vetting, legal support, transport, and post-release coordination.
- Review breach probability daily against a configurable operating capacity threshold.
- Treat prediction intervals as the operational planning range, not just the point forecast.
