"""Reference forecasts on the scaled data, from repeating known prices to Ridge and LightGBM. A model is only useful if
it beats them."""
import lightgbm as lgb
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
        "day D-1": day_before(X, horizon),
        "day D-7": X[:, window - 168 + j, 0],
    }


def ridge_features(X: np.ndarray, Y: np.ndarray, n_lags: int=168) -> np.ndarray:
    """Spot price of the last n_lags hours, all features of the last hour and the future covariates."""
    return np.c_[X[:, -n_lags:, 0], X[:, -1, :], Y[:, :, 1:].reshape(len(Y), -1)]


def day_before(X: np.ndarray, horizon: int=24) -> np.ndarray:
    """The scaled spot prices of the same hours on day D-1, the last 24 hours of the history, shape (n, horizon). A
    relative target is the change against them, so a model carries the price level of D-1 forward and only forecasts
    the deviation from it."""
    return X[:, -24 + np.arange(horizon) % 24, 0]


def fit_ridge(train_ds: EnergyDataset, relative: bool=False) -> RidgeCV:
    """Linear model that forecasts all hours of the horizon at once, regularization chosen on the training data. With
    relative, it forecasts the change against day D-1 (predict_ridge adds it back)."""
    X, Y = stack_windows(train_ds)
    target = Y[:, :, 0] - day_before(X, Y.shape[1]) if relative else Y[:, :, 0]
    return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(ridge_features(X, Y), target)


def predict_ridge(model: RidgeCV, X: np.ndarray, Y: np.ndarray, relative: bool=False) -> np.ndarray:
    """The scaled spot price forecasts of Ridge, of shape (n, horizon)."""
    pred = model.predict(ridge_features(X, Y))
    return pred + day_before(X, Y.shape[1]) if relative else pred


def lgbm_table(X: np.ndarray, Y: np.ndarray, n_lags: int=168) -> tuple[np.ndarray, np.ndarray]:
    """One row per day and hour: the spot price of the last n_lags hours and all features of the last history hour
    (the same for every hour of a day), then the hour, the price of the same hour on D-1 and D-7 and the future
    covariates of that hour. A tree cannot pick the matching lag for an hour by itself, so those are added."""
    n, horizon = Y.shape[:2]
    hours = np.arange(horizon)
    day_part = np.repeat(np.c_[X[:, -n_lags:, 0], X[:, -1, :]], horizon, axis=0)
    hour_part = np.c_[np.tile(hours, n), X[:, -24 + hours, 0].reshape(-1), X[:, -168 + hours, 0].reshape(-1)]
    return np.c_[day_part, hour_part, Y[:, :, 1:].reshape(n * horizon, -1)], Y[:, :, 0].reshape(-1)


# the settings of the feature selection; num_leaves, min_child_samples, reg_lambda and objective are LightGBM's defaults
LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
               "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 0.0,
               "objective": "regression"}


def fit_lgbm(X: np.ndarray, Y: np.ndarray, seed: int=1, relative: bool=False, **params) -> lgb.LGBMRegressor:
    """Gradient boosted trees on the stacked windows of the training split, one model for all hours (lgbm_table). The
    params override LGBM_PARAMS, e.g. the [params] of configs/lgbm_v2.toml. With relative, the trees forecast the change
    against day D-1 (predict_lgbm adds it back): a tree only outputs means of training targets, so it cannot reach a
    price level that it has not seen, but it can reach a change that it has seen."""
    model = lgb.LGBMRegressor(**{**LGBM_PARAMS, **params}, random_state=seed, verbose=-1)
    table, target = lgbm_table(X, Y)
    if relative:
        target = target - day_before(X, Y.shape[1]).reshape(-1)
    return model.fit(table, target)


def predict_lgbm(model: lgb.LGBMRegressor, X: np.ndarray, Y: np.ndarray, relative: bool=False) -> np.ndarray:
    """The scaled spot price forecasts of LightGBM, of shape (n, horizon)."""
    pred = model.predict(lgbm_table(X, Y)[0]).reshape(len(X), Y.shape[1])
    return pred + day_before(X, Y.shape[1]) if relative else pred
