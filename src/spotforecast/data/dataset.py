import pandas as pd
import numpy as np
import torch
from typing import Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import spotforecast.data.preprocessing as datap
from spotforecast.config import PROJECT_ROOT
from spotforecast.data.features import add_derived_features_, get_input_feature_names, get_time_features, with_derived_features


class EnergyDataset(Dataset):

    def __init__(self, features, target, window, horizon, stride):
        self.features = torch.tensor(features, dtype=torch.float32) if not torch.is_tensor(features) else features
        self.target = torch.tensor(target, dtype=torch.float32) if not torch.is_tensor(target) else target
        if len(self.target) != len(self.features):
            raise ValueError(f"Expected features and targets to have equal length and the same indices!")
        self.window = window
        self.horizon = horizon
        self.stride = stride

        self.set_usable_idxs()

    def set_usable_idxs(self):
        """The window returned for an index starts at that index, then continues for self.window many steps for the
        input and ends with another self.horizon steps for the target features."""
        self.usable_idxs = np.arange(stop=len(self.features) - self.window - self.horizon + 1, step=self.stride, dtype=np.int32)

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


def get_normalized_data(data: pd.DataFrame, features: list[str], targets: list[str], datetime_col: str, valid_cutoff_datetime: str):
    data_X = data[features].to_numpy(dtype=np.float32)
    data_y = data[targets].to_numpy(dtype=np.float32)

    # calculate the index map
    valid_dt = datetime.fromisoformat(valid_cutoff_datetime)
    timezone = data[datetime_col][0].tz
    if timezone is not None:
        valid_dt_copy = valid_dt
        if valid_dt_copy.tzinfo is None:
            train_test_split_local = timezone.localize(valid_dt_copy)
        else:
            train_test_split_local = valid_dt_copy
        _train_map = data[datetime_col] < train_test_split_local
    else:
        _train_map = data[datetime_col] < valid_dt

    # assuming the dataset is ordered, we can calculate the index
    _train_end = int(_train_map.sum()) + 1
    _val_end = None

    scaler_X = StandardScaler().fit(data_X[:_train_end])
    _means_X = data.loc[_train_map, features].apply(datap.trimmed_func, axis=0, function = np.mean, keep=0.95)
    _stds_X = data.loc[_train_map, features].apply(datap.trimmed_func, axis=0, function = np.std, keep=0.95)
    scaler_X.mean_ = _means_X.to_numpy(dtype=np.float64)
    scaler_X.scale_ = _stds_X.to_numpy(dtype=np.float64)

    scaler_y = StandardScaler().fit(data_y[:_train_end])
    _means_y = data.loc[_train_map, targets].apply(datap.trimmed_func, axis=0, function = np.mean, keep=0.95)
    _stds_y = data.loc[_train_map, targets].apply(datap.trimmed_func, axis=0, function = np.std, keep=0.95)
    scaler_y.mean_ = _means_y.to_numpy(dtype=np.float64)
    scaler_y.scale_ = _stds_y.to_numpy(dtype=np.float64)

    X_scaled = scaler_X.transform(data_X)
    y_scaled = scaler_y.transform(data_y)

    return X_scaled, y_scaled, _train_end, _val_end


def get_preprocessed_ts_data(path: str, features: list[str], targets: list[str], window_size: int, forecast_horizon: int, stride: int, preprocess_kwargs: dict):
    _targets = [targets] if not isinstance(targets, list) else targets
    _datetime_col = preprocess_kwargs.get("datetime_col", "Date")
    df = load_from_parquet(path=path, input_cols=features, output_cols=_targets, additional_cols=[_datetime_col])

    _valid_cutoff_datetime = preprocess_kwargs.pop("valid_cutoff_datetime") if preprocess_kwargs.get("valid_cutoff_datetime", None) else None

    _data_cols = features + [_c for _c in _targets if _c not in features]  # all data cols must be normalized
    df = datap.preprocess_ts_data(data=df,
                                  input_cols=_data_cols,
                                  **preprocess_kwargs
                                  )

    add_derived_features_(df)
    _features = with_derived_features(features)
    print(f"Using features: {get_input_feature_names(features)}, using targets: {_targets}")

    X_scaled, y_scaled, _train_end, _val_end = get_normalized_data(data=df,
                                                                   features=_features,
                                                                   targets=_targets,
                                                                   datetime_col=_datetime_col,
                                                                   valid_cutoff_datetime=_valid_cutoff_datetime)

    # the cyclic time features are already in [-1, 1] and are appended without scaling
    X_scaled = np.c_[X_scaled, get_time_features(df.loc[:, _datetime_col], dtype=np.float32)]

    train_ds = EnergyDataset(X_scaled[:_train_end], y_scaled[:_train_end], window_size,
                             forecast_horizon, stride)
    valid_ds = EnergyDataset(X_scaled[_train_end:_val_end], y_scaled[_train_end:_val_end], window_size,
                             forecast_horizon, stride)

    return train_ds, valid_ds


def load_datasets(config: dict, data_config: dict) -> tuple[EnergyDataset, EnergyDataset]:
    """Builds the train and validation datasets from the model config and the data config."""
    return get_preprocessed_ts_data(
        path=PROJECT_ROOT / data_config["path"],
        features=config["features"],
        targets=config["targets"],
        window_size=config["forecast_history"],
        forecast_horizon=config["forecast_horizon"],
        stride=config["stride"],
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
