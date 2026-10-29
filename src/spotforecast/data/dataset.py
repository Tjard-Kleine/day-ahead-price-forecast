import pandas as pd
import numpy as np
import torch
from typing import Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import spotforecast.data.preprocessing as datap
from spotforecast.config import PROJECT_ROOT
from spotforecast.data.features import (CALENDAR_FEATURES, add_columns_, add_derived_features_, get_input_feature_names,
                                        get_time_features, source_columns, with_derived_features)


class EnergyDataset(Dataset):

    def __init__(self, features, target, window, horizon, target_starts, dates: Optional[pd.DatetimeIndex]=None,
                 target_scaler: Optional[StandardScaler]=None):
        self.features = torch.tensor(features, dtype=torch.float32) if not torch.is_tensor(features) else features
        self.target = torch.tensor(target, dtype=torch.float32) if not torch.is_tensor(target) else target
        if len(self.target) != len(self.features):
            raise ValueError(f"Expected features and targets to have equal length and the same indices!")
        self.window = window
        self.horizon = horizon
        # timestamp of every row, e.g. to find the forecast day of a window
        self.dates = dates
        # undoes the target normalization, e.g. to report errors in EUR/MWh
        self.target_scaler = target_scaler

        self.set_usable_idxs(target_starts)

    def set_usable_idxs(self, target_starts):
        """The window returned for an index starts at that index, then continues for self.window many steps for the
        input and ends with another self.horizon steps for the target features. Only the windows whose targets begin
        at one of the target_starts rows and that fit completely into the data are used."""
        target_starts = np.asarray(target_starts)
        _fits = (target_starts >= self.window) & (target_starts + self.horizon <= len(self.features))
        self.usable_idxs = (target_starts[_fits] - self.window).astype(np.int32)

    def forecast_starts(self) -> pd.DatetimeIndex:
        """Timestamp of the first forecast hour of every window."""
        return self.dates[self.usable_idxs + self.window]

    def __len__(self):
        return len(self.usable_idxs)

    def __getitem__(self, idx):
        _idx = self.usable_idxs[idx]
        x = self.features[_idx: _idx + self.window]
        y = self.target[_idx + self.window: _idx + self.window + self.horizon]
        return x, y


def load_from_parquet(path: str, input_cols: list[str], output_cols: Optional[list[str]]=None, additional_cols:Optional[list[str]]=None):
    """Loads a dataset from a parquet file.

    :param str path: the path to the dataset
    :param list[str] input_cols: the list of columns to use for the input
    :param list[str] additional_cols: list of columns that will be used intermediately
    """
    _load_cols = additional_cols + input_cols if additional_cols is not None else input_cols
    if output_cols is not None:
        _load_cols = _load_cols + [_oc for _oc in output_cols if _oc not in _load_cols]

    return pd.read_parquet(path=path, columns=_load_cols)


def before_cutoff(dates: pd.Series, cutoff_datetime: str) -> pd.Series:
    """Marks the timestamps before the cutoff, which is read in the timezone of the dates if it has none."""
    cutoff = datetime.fromisoformat(cutoff_datetime)
    timezone = dates[0].tz
    if timezone is not None and cutoff.tzinfo is None:
        cutoff = timezone.localize(cutoff)
    return dates < cutoff


def fit_trimmed_scaler(data: pd.DataFrame, columns: list[str], train_map: pd.Series) -> StandardScaler:
    """Scaler with the trimmed mean and std of the training rows. The calendar columns already lie in [0, 1] and are
    kept as they are, their trimmed std would be 0."""
    _train = data.loc[train_map, columns]
    scaler = StandardScaler().fit(_train.to_numpy(dtype=np.float32))
    _means = _train.apply(datap.trimmed_func, axis=0, function = np.mean, keep=0.95).to_numpy(dtype=np.float64)
    _stds = _train.apply(datap.trimmed_func, axis=0, function = np.std, keep=0.95).to_numpy(dtype=np.float64)
    _unscaled = np.isin(columns, CALENDAR_FEATURES)
    scaler.mean_ = np.where(_unscaled, 0.0, _means)
    scaler.scale_ = np.where(_unscaled, 1.0, _stds)
    return scaler


def get_normalized_data(data: pd.DataFrame, features: list[str], targets: list[str], datetime_col: str,
                        valid_cutoff_datetime: str, test_cutoff_datetime: str):
    # assuming the dataset is ordered, the splits end at the number of rows before their cutoff
    _train_map = before_cutoff(data[datetime_col], valid_cutoff_datetime)
    _train_end = int(_train_map.sum())
    _val_end = int(before_cutoff(data[datetime_col], test_cutoff_datetime).sum())

    scaler_X = fit_trimmed_scaler(data, features, _train_map)
    scaler_y = fit_trimmed_scaler(data, targets, _train_map)

    X_scaled = scaler_X.transform(data[features].to_numpy(dtype=np.float32))
    y_scaled = scaler_y.transform(data[targets].to_numpy(dtype=np.float32))

    return X_scaled, y_scaled, scaler_y, _train_end, _val_end


def load_preprocessed_data(path: str, columns: list[str], preprocess_kwargs: dict) -> pd.DataFrame:
    """Hourly data with the given columns, including the derived, calendar, computed and lagged ones. The
    preprocess_kwargs are those of the data config without the split cutoffs."""
    _datetime_col = preprocess_kwargs.get("datetime_col", "Date")
    _source_cols = source_columns(columns)
    df = load_from_parquet(path=path, input_cols=_source_cols, additional_cols=[_datetime_col])
    df = datap.preprocess_ts_data(data=df,
                                  input_cols=_source_cols,
                                  **preprocess_kwargs
                                  )

    add_derived_features_(df)
    add_columns_(df, columns, _datetime_col)
    return df


def get_preprocessed_ts_data(path: str, features: list[str], targets: list[str], window_size: int, forecast_horizon: int, preprocess_kwargs: dict):
    _targets = [targets] if not isinstance(targets, list) else targets
    _datetime_col = preprocess_kwargs.get("datetime_col", "Date")
    _valid_cutoff_datetime = preprocess_kwargs.pop("valid_cutoff_datetime")
    _test_cutoff_datetime = preprocess_kwargs.pop("test_cutoff_datetime")

    _data_cols = features + [_c for _c in _targets if _c not in features]  # all data cols must be normalized
    df = load_preprocessed_data(path, _data_cols, preprocess_kwargs)
    print(f"Using features: {get_input_feature_names(features)}, using targets: {_targets}")

    return build_datasets(df, features, _targets, window_size, forecast_horizon, _datetime_col,
                          _valid_cutoff_datetime, _test_cutoff_datetime)


def build_datasets(df: pd.DataFrame, features: list[str], targets: list[str], window_size: int, forecast_horizon: int,
                   datetime_col: str, valid_cutoff_datetime: str, test_cutoff_datetime: str
                   ) -> tuple[EnergyDataset, EnergyDataset, EnergyDataset]:
    """Scales the hourly data with the statistics of the training split and cuts it into the windows of the
    train, validation and test split. df may hold more columns than needed, e.g. for several feature sets."""
    _features = with_derived_features(features)
    X_scaled, y_scaled, scaler_y, _train_end, _val_end = get_normalized_data(data=df,
                                                                             features=_features,
                                                                             targets=targets,
                                                                             datetime_col=datetime_col,
                                                                             valid_cutoff_datetime=valid_cutoff_datetime,
                                                                             test_cutoff_datetime=test_cutoff_datetime)

    # the cyclic time features are already in [-1, 1] and are appended without scaling
    X_scaled = np.c_[X_scaled, get_time_features(df.loc[:, datetime_col], dtype=np.float32)]

    # day-ahead task: one forecast per day whose targets start at local midnight, the history ends right before.
    # The two days per year with a clock change (23 or 25 hours) are skipped, so every forecast day has 24 hours
    _dates = pd.DatetimeIndex(df[datetime_col])
    _day = pd.Series(_dates.date)
    _hours_of_day = _day.map(_day.value_counts()).to_numpy()
    _day_starts = np.flatnonzero((_dates.hour == 0) & (_hours_of_day == 24))

    def _split(start: int, end: Optional[int]) -> EnergyDataset:
        # the history window may reach back into the previous split, the forecast targets all lie in [start, end)
        _start = max(start - window_size, 0)
        _target_starts = _day_starts[_day_starts >= start] - _start
        return EnergyDataset(X_scaled[_start:end], y_scaled[_start:end], window_size, forecast_horizon, _target_starts,
                             dates=_dates[_start:end], target_scaler=scaler_y)

    train_ds = _split(0, _train_end)
    valid_ds = _split(_train_end, _val_end)
    test_ds = _split(_val_end, None)

    return train_ds, valid_ds, test_ds


def load_datasets(config: dict, data_config: dict) -> tuple[EnergyDataset, EnergyDataset, EnergyDataset]:
    """Builds the train, validation and test datasets from the model config and the data config."""
    return get_preprocessed_ts_data(
        path=PROJECT_ROOT / data_config["path"],
        features=config["features"],
        targets=config["targets"],
        window_size=config["forecast_history"],
        forecast_horizon=config["forecast_horizon"],
        preprocess_kwargs=dict(data_config["preprocessing"]),
    )


def get_feature_dims(ds: EnergyDataset, num_targets: int=1) -> tuple[int, int]:
    """Returns the number of history features and of future covariates per time step."""
    x, y = ds[0]
    return x.shape[-1], y.shape[-1] - num_targets


def default_batch_transform(x: torch.Tensor, y: torch.Tensor, device: torch.device, train: bool):
    x, y = x.to(device), y.to(device)
    if y.shape[-1] == 1: y = y.squeeze(-1)  # y has shape (batch_size, 1)
    return x, y


def split_history_future_batch_transform(x: torch.Tensor, y: torch.Tensor, device: torch.device, train: bool, num_targets: int=1):
    """Turns a batch into the model input (history, future covariates) and the forecast target.

    The first num_targets columns of y are forecast, the remaining columns are covariates that are known for the
    forecast horizon (e.g. weather forecasts) and are passed to the model as input."""

    with torch.no_grad():
        _future = y[:, :, num_targets:]
        _y = y[:, :, :num_targets]
        if _y.shape[-1] == 1: _y = _y.squeeze(-1)
    _x = (x.requires_grad_(train).to(device), _future.requires_grad_(train).to(device))
    _y = _y.requires_grad_(train).to(device)
    return _x, _y
