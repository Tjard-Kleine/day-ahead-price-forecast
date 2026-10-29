"""Simple reference forecasts on the scaled data. A model is only useful if it beats them."""
import numpy as np
from sklearn.linear_model import RidgeCV

from spotforecast.data.dataset import EnergyDataset


def stack_windows(ds: EnergyDataset) -> tuple[np.ndarray, np.ndarray]:
    """All windows of a dataset as arrays of shape (n, window, n_features) and (n, horizon, n_targets)."""
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    Y = np.stack([ds[i][1].numpy() for i in range(len(ds))])
    return X, Y


def naive_forecasts(X: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Forecasts of the spot price (first history feature) that only repeat known values. The history ends at
    midnight before the forecast day D, so its last 24 hours are the prices of day D-1."""
    window = X.shape[1]
    j = np.arange(horizon)
    return {
        "mean": np.zeros((len(X), horizon), dtype=X.dtype),  # the training mean is 0 after scaling
        "persistence": np.repeat(X[:, -1:, 0], horizon, axis=1),
        "day D-1": X[:, window - 24 + (j % 24), 0],
        "day D-7": X[:, window - 168 + j, 0],
    }


def ridge_features(X: np.ndarray, Y: np.ndarray, n_lags: int=168) -> np.ndarray:
    """Spot price of the last n_lags hours, all features of the last hour and the future covariates."""
    return np.c_[X[:, -n_lags:, 0], X[:, -1, :], Y[:, :, 1:].reshape(len(Y), -1)]


def fit_ridge(train_ds: EnergyDataset) -> RidgeCV:
    """Linear model that forecasts all hours of the horizon at once, regularization chosen on the training data."""
    X, Y = stack_windows(train_ds)
    return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(ridge_features(X, Y), Y[:, :, 0])
