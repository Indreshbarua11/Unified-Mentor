from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HHS_Unaccompanied_Alien_Children_Program.csv"

COLUMN_MAP = {
    "Children apprehended and placed in CBP custody*": "cbp_apprehensions",
    "Children in CBP custody": "cbp_care_load",
    "Children transferred out of CBP custody": "transfers_to_hhs",
    "Children in HHS Care": "hhs_care_load",
    "Children discharged from HHS Care": "hhs_discharges",
}

DISPLAY_NAMES = {
    "cbp_apprehensions": "Children apprehended and placed in CBP custody",
    "cbp_care_load": "Children in CBP custody",
    "transfers_to_hhs": "Children transferred out of CBP custody",
    "hhs_care_load": "Children in HHS Care",
    "hhs_discharges": "Children discharged from HHS Care",
    "net_pressure": "Transfers minus discharges",
}


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def load_uac_data(path: str | Path = RAW_DATA_PATH, interpolate_missing: bool = True) -> pd.DataFrame:
    """Load, clean, sort, and optionally regularize the UAC daily time series."""
    df = pd.read_csv(path)
    df = df.dropna(how="all").copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).rename(columns=COLUMN_MAP)

    for col in COLUMN_MAP.values():
        df[col] = _to_number(df[col])

    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    df = df.set_index("Date").asfreq("D")
    df["was_reported"] = df["hhs_care_load"].notna()

    if interpolate_missing:
        numeric_cols = list(COLUMN_MAP.values())
        df[numeric_cols] = df[numeric_cols].interpolate("time").ffill().bfill()
        flow_cols = ["cbp_apprehensions", "cbp_care_load", "transfers_to_hhs", "hhs_discharges"]
        df[flow_cols] = df[flow_cols].clip(lower=0)

    df["net_pressure"] = df["transfers_to_hhs"] - df["hhs_discharges"]
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_forecast_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Create leakage-safe lag, rolling, flow, and calendar features."""
    out = df.copy()
    for lag in (1, 7, 14, 28):
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)

    shifted = out[target].shift(1)
    for window in (7, 14, 28):
        out[f"{target}_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{target}_roll_std_{window}"] = shifted.rolling(window).std()

    for col in ("transfers_to_hhs", "hhs_discharges", "net_pressure", "cbp_care_load"):
        out[f"{col}_lag_1"] = out[col].shift(1)
        out[f"{col}_roll_mean_7"] = out[col].shift(1).rolling(7).mean()

    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out.dropna()


def train_test_split_time(df: pd.DataFrame, test_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_days:
        raise ValueError("Not enough observations for requested test window.")
    return df.iloc[:-test_days].copy(), df.iloc[-test_days:].copy()
