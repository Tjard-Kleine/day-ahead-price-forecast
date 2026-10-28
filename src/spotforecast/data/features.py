import numpy as np
import pandas as pd

DERIVED_FEATURES = ["Spot_roll_168", "solar_util"]
TIME_FEATURES = ["hour_sin", "hour_cos"]


def get_time_features(datetime_series: pd.Series, dtype: type=np.float32) -> np.ndarray:
    """Cyclic encoding of the hour of day, one row per timestamp in the order of TIME_FEATURES."""
    ds = pd.to_datetime(datetime_series)
    hour = ds.dt.hour.to_numpy().astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24).astype(dtype)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(dtype)

    return np.stack([hour_sin, hour_cos], axis=1).astype(dtype)


def add_derived_features_(df: pd.DataFrame) -> None:
    """Adds the DERIVED_FEATURES columns to the hourly data in place."""
    if "Spot" in df.columns:
        df.loc[:, "Spot_roll_168"] = df["Spot"].rolling(window=168, min_periods=1).mean()

    if "solar" in df.columns and "solar capacity" in df.columns:
        df.loc[:, "solar_util"] = df["solar"] / (df["solar capacity"] + 1e-6)


def with_derived_features(features: list[str]) -> list[str]:
    """The data columns that get normalized: the configured features followed by the derived ones."""
    return features + [_c for _c in DERIVED_FEATURES if _c not in features]


def get_input_feature_names(features: list[str]) -> list[str]:
    """Column order of the history window that is fed to the model."""
    return with_derived_features(features) + TIME_FEATURES
