from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data_prep import add_forecast_features, load_uac_data

try:  # Optional production models, activated when requirements.txt is installed.
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None
    RandomForestRegressor = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover
    ExponentialSmoothing = None
    SARIMAX = None


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    backtest: pd.DataFrame


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def _linear_fit_predict(train_y: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(train_y), dtype=float)
    x_mean = x.mean()
    y_mean = train_y.mean()
    slope = np.sum((x - x_mean) * (train_y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-9)
    intercept = y_mean - slope * x_mean
    future_x = np.arange(len(train_y), len(train_y) + horizon, dtype=float)
    return intercept + slope * future_x


def _exp_smoothing(train_y: np.ndarray, horizon: int, alpha: float = 0.35) -> np.ndarray:
    level = train_y[0]
    for value in train_y[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.repeat(level, horizon)


def _moving_average(train_y: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    return np.repeat(np.mean(train_y[-window:]), horizon)


def _seasonal_naive(train_y: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    if len(train_y) < season:
        return np.repeat(train_y[-1], horizon)
    pattern = train_y[-season:]
    return np.array([pattern[i % season] for i in range(horizon)], dtype=float)


def _persistence(train_y: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(train_y[-1], horizon)


def _ridge_regression_forecast(history: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    featured = add_forecast_features(history, target)
    feature_cols = [c for c in featured.columns if c != target and featured[c].dtype.kind in "ifbu"]
    x = featured[feature_cols].to_numpy(dtype=float)
    y = featured[target].to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1
    x_scaled = (x - x_mean) / x_std
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    lam = 2.0
    penalty = np.eye(x_design.shape[1]) * lam
    penalty[0, 0] = 0
    beta = np.linalg.pinv(x_design.T @ x_design + penalty) @ x_design.T @ y

    simulated = history.copy()
    preds: list[float] = []
    for _ in range(horizon):
        next_date = simulated.index.max() + pd.Timedelta(days=1)
        next_row = simulated.iloc[-1:].copy()
        next_row.index = [next_date]
        next_row["day_of_week"] = next_date.dayofweek
        next_row["month"] = next_date.month
        next_row["is_weekend"] = int(next_date.dayofweek in (5, 6))
        simulated = pd.concat([simulated, next_row])
        feat_next = add_forecast_features(simulated, target).iloc[-1:][feature_cols]
        pred = (np.column_stack([np.ones(1), (feat_next.to_numpy(dtype=float) - x_mean) / x_std]) @ beta).item()
        pred = max(pred, 0)
        simulated.loc[next_date, target] = pred
        preds.append(pred)
    return np.array(preds)


def _feature_model_forecast(history: pd.DataFrame, target: str, horizon: int, estimator) -> np.ndarray:
    featured = add_forecast_features(history, target)
    feature_cols = [c for c in featured.columns if c != target and featured[c].dtype.kind in "ifbu"]
    model = estimator()
    model.fit(featured[feature_cols], featured[target])

    simulated = history.copy()
    preds: list[float] = []
    for _ in range(horizon):
        next_date = simulated.index.max() + pd.Timedelta(days=1)
        next_row = simulated.iloc[-1:].copy()
        next_row.index = [next_date]
        next_row["day_of_week"] = next_date.dayofweek
        next_row["month"] = next_date.month
        next_row["is_weekend"] = int(next_date.dayofweek in (5, 6))
        simulated = pd.concat([simulated, next_row])
        feat_next = add_forecast_features(simulated, target).iloc[-1:][feature_cols]
        pred = max(float(model.predict(feat_next)[0]), 0)
        simulated.loc[next_date, target] = pred
        preds.append(pred)
    return np.array(preds)


def _random_forest_forecast(history: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    if RandomForestRegressor is None:
        raise RuntimeError("scikit-learn is required for Random Forest Regressor.")
    return _feature_model_forecast(
        history,
        target,
        horizon,
        lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1),
    )


def _gradient_boosting_forecast(history: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    if GradientBoostingRegressor is None:
        raise RuntimeError("scikit-learn is required for Gradient Boosting Regressor.")
    return _feature_model_forecast(
        history,
        target,
        horizon,
        lambda: GradientBoostingRegressor(random_state=42, learning_rate=0.04, n_estimators=250, max_depth=2),
    )


def _ets_forecast(history: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels is required for ETS.")
    series = history[target].astype(float)
    seasonal = "add" if len(series) >= 21 else None
    seasonal_periods = 7 if seasonal else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.maximum(model.forecast(horizon).to_numpy(dtype=float), 0)


def _sarima_forecast(history: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    if SARIMAX is None:
        raise RuntimeError("statsmodels is required for SARIMA.")
    series = history[target].astype(float)
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return np.maximum(model.forecast(horizon).to_numpy(dtype=float), 0)


MODEL_REGISTRY: dict[str, Callable[[pd.DataFrame, str, int], np.ndarray]] = {
    "Naive persistence": lambda df, target, horizon: _persistence(df[target].to_numpy(dtype=float), horizon),
    "7-day moving average": lambda df, target, horizon: _moving_average(df[target].to_numpy(dtype=float), horizon),
    "Seasonal naive": lambda df, target, horizon: _seasonal_naive(df[target].to_numpy(dtype=float), horizon),
    "Exponential smoothing": lambda df, target, horizon: _exp_smoothing(df[target].to_numpy(dtype=float), horizon),
    "Trend regression": lambda df, target, horizon: _linear_fit_predict(df[target].to_numpy(dtype=float), horizon),
    "Feature regression": _ridge_regression_forecast,
}

if ExponentialSmoothing is not None:
    MODEL_REGISTRY["ETS (statsmodels)"] = _ets_forecast
if SARIMAX is not None:
    MODEL_REGISTRY["SARIMA (statsmodels)"] = _sarima_forecast
if RandomForestRegressor is not None:
    MODEL_REGISTRY["Random Forest Regressor"] = _random_forest_forecast
if GradientBoostingRegressor is not None:
    MODEL_REGISTRY["Gradient Boosting Regressor"] = _gradient_boosting_forecast


def evaluate_models(df: pd.DataFrame, target: str, test_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    rows = []
    backtest_frames = []

    for model_name, model_fn in MODEL_REGISTRY.items():
        preds = model_fn(train, target, len(test))
        actual = test[target].to_numpy(dtype=float)
        residual = actual - preds
        rows.append(
            {
                "target": target,
                "model": model_name,
                "MAE": mae(actual, preds),
                "RMSE": rmse(actual, preds),
                "MAPE": mape(actual, preds),
                "Forecast Accuracy (%)": max(0.0, 100 - mape(actual, preds)),
                "Residual Std": float(np.std(residual, ddof=1)),
            }
        )
        backtest_frames.append(
            pd.DataFrame(
                {
                    "date": test.index,
                    "target": target,
                    "model": model_name,
                    "actual": actual,
                    "predicted": preds,
                    "error": residual,
                }
            )
        )

    metrics = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    return metrics, pd.concat(backtest_frames, ignore_index=True)


def make_forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = 30,
    model_name: str | None = None,
    test_days: int = 90,
) -> ForecastResult:
    metrics, backtest = evaluate_models(df, target, test_days=test_days)
    selected = model_name or str(metrics.iloc[0]["model"])
    model_fn = MODEL_REGISTRY[selected]
    point = model_fn(df, target, horizon)

    residual_std = float(metrics.loc[metrics["model"] == selected, "Residual Std"].iloc[0])
    steps = np.arange(1, horizon + 1)
    interval = 1.96 * residual_std * np.sqrt(steps / max(steps[0], 1))
    dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame(
        {
            "date": dates,
            "target": target,
            "model": selected,
            "horizon_day": steps,
            "forecast": np.maximum(point, 0),
            "lower_95": np.maximum(point - interval, 0),
            "upper_95": np.maximum(point + interval, 0),
        }
    )
    return ForecastResult(forecast=forecast, metrics=metrics, backtest=backtest)


def build_default_outputs(horizon: int = 30) -> dict[str, pd.DataFrame]:
    df = load_uac_data()
    outputs: dict[str, pd.DataFrame] = {"clean_daily": df.reset_index(names="date")}
    forecast_frames = []
    metrics_frames = []
    backtest_frames = []
    for target in ("hhs_care_load", "hhs_discharges"):
        result = make_forecast(df, target, horizon=horizon)
        forecast_frames.append(result.forecast)
        metrics_frames.append(result.metrics)
        backtest_frames.append(result.backtest)
    outputs["forecasts"] = pd.concat(forecast_frames, ignore_index=True)
    outputs["metrics"] = pd.concat(metrics_frames, ignore_index=True)
    outputs["backtest"] = pd.concat(backtest_frames, ignore_index=True)
    return outputs
