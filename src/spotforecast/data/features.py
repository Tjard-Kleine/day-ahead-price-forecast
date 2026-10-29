import re

import holidays
import numpy as np
import pandas as pd

DERIVED_FEATURES = ["Spot_roll_168", "solar_util"]
TIME_FEATURES = ["hour_sin", "hour_cos"]

# computed from the date and already in [0, 1], so they are not scaled; holiday is the share of the population that
# has a public holiday on that day
CALENDAR_FEATURES = ["saturday", "sunday", "holiday"]
# computed from other columns of the hourly data: name -> (columns it needs, function of the data)
COMPUTED_FEATURES = {
    # the demand that wind and solar leave to the other power plants, in the merit order it sets the price
    "residual load": (["conso", "onshore wind", "offshore wind", "solar"],
                      lambda df: df["conso"] - df["onshore wind"] - df["offshore wind"] - df["solar"]),
}
# "<column>_lag<h>" is the column shifted by h hours, for values that are not known up to the end of the history,
# e.g. "natural gas_lag12": at 12:00 on D-1 the generation is only measured until about 12:00
LAG_PATTERN = re.compile(r"^(?P<column>.+)_lag(?P<hours>\d+)$")

# approximate population of the German states in 2023, in millions, weights the regional public holidays
STATE_POPULATION = {"BW": 11.3, "BY": 13.4, "BE": 3.8, "BB": 2.6, "HB": 0.7, "HH": 1.9, "HE": 6.4, "MV": 1.6,
                    "NI": 8.1, "NW": 18.1, "RP": 4.2, "SL": 1.0, "SN": 4.1, "ST": 2.2, "SH": 3.0, "TH": 2.1}


def get_time_features(datetime_series: pd.Series, dtype: type=np.float32) -> np.ndarray:
    """Cyclic encoding of the hour of day, one row per timestamp in the order of TIME_FEATURES."""
    ds = pd.to_datetime(datetime_series)
    hour = ds.dt.hour.to_numpy().astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24).astype(dtype)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(dtype)

    return np.stack([hour_sin, hour_cos], axis=1).astype(dtype)


def get_calendar_features(datetime_series: pd.Series) -> pd.DataFrame:
    """The CALENDAR_FEATURES of the local date of every timestamp."""
    ds = pd.to_datetime(datetime_series)
    days = ds.dt.date
    years = range(ds.dt.year.min(), ds.dt.year.max() + 1)
    state_holidays = {state: holidays.Germany(subdiv=state, years=years) for state in STATE_POPULATION}
    total_population = sum(STATE_POPULATION.values())

    def holiday_share(day) -> float:
        # Christmas Eve and New Year's Eve are no public holidays, but the demand drops as on one
        if (day.month, day.day) in [(12, 24), (12, 31)]:
            return 1.0
        return sum(pop for state, pop in STATE_POPULATION.items() if day in state_holidays[state]) / total_population

    shares = {day: holiday_share(day) for day in days.unique()}
    return pd.DataFrame({"saturday": (ds.dt.dayofweek == 5).astype(float),
                         "sunday": (ds.dt.dayofweek == 6).astype(float),
                         "holiday": days.map(shares)})


def add_derived_features_(df: pd.DataFrame) -> None:
    """Adds the DERIVED_FEATURES columns to the hourly data in place."""
    if "Spot" in df.columns:
        df.loc[:, "Spot_roll_168"] = df["Spot"].rolling(window=168, min_periods=1).mean()

    if "solar" in df.columns and "solar capacity" in df.columns:
        df.loc[:, "solar_util"] = df["solar"] / (df["solar capacity"] + 1e-6)


def add_columns_(df: pd.DataFrame, names: list[str], datetime_col: str) -> None:
    """Adds the calendar, computed and lagged columns among names to the hourly data in place."""
    calendar = [_c for _c in CALENDAR_FEATURES if _c in names]
    if calendar:
        df[calendar] = get_calendar_features(df[datetime_col])[calendar]
    for name in names:
        lag = LAG_PATTERN.match(name)
        if name in COMPUTED_FEATURES:
            df[name] = COMPUTED_FEATURES[name][1](df)
        elif lag is not None:
            if lag["column"] not in df.columns:
                add_columns_(df, [lag["column"]], datetime_col)
            # the first hours have no earlier value and repeat the first one, they only appear in the first history
            df[name] = df[lag["column"]].shift(int(lag["hours"])).bfill()


def source_columns(names: list[str]) -> list[str]:
    """The columns of the raw data that are needed for the given feature names, without duplicates."""
    columns = []
    for name in names:
        lag = LAG_PATTERN.match(name)
        if name in CALENDAR_FEATURES or name in DERIVED_FEATURES:
            needed = []
        elif name in COMPUTED_FEATURES:
            needed = COMPUTED_FEATURES[name][0]
        elif lag is not None:
            needed = source_columns([lag["column"]])
        else:
            needed = [name]
        columns += [_c for _c in needed if _c not in columns]
    return columns


def add_feature_groups(features: list[str], targets: list[str], groups: dict, names) -> tuple[list[str], list[str]]:
    """The features and targets extended by the history features and future covariates of the named groups, which are
    defined in configs/feature_selection.toml."""
    for name in names:
        features = features + [_f for _f in groups[name]["features"] if _f not in features]
        targets = targets + [_f for _f in groups[name]["future"] if _f not in targets]
    return features, targets


def with_derived_features(features: list[str]) -> list[str]:
    """The data columns that get normalized: the configured features followed by the derived ones."""
    return features + [_c for _c in DERIVED_FEATURES if _c not in features]


def get_input_feature_names(features: list[str]) -> list[str]:
    """Column order of the history window that is fed to the model."""
    return with_derived_features(features) + TIME_FEATURES
