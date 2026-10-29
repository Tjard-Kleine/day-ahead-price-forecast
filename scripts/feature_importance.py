"""Which inputs carry information about the spot prices of the forecast day?

A random forest forecasts the 24 prices of day D from the last history step (23:00 on D-1) and the future covariates
of day D. The importance of an input is how much the validation MAE in EUR/MWh grows when the input is shuffled across
the days; the 24 hourly values of a future covariate are shuffled together. The forest only sees the last history step,
so it ranks the inputs, it is no forecast model.

    python scripts/feature_importance.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from spotforecast.config import load_config
from spotforecast.data.dataset import load_datasets
from spotforecast.data.features import get_input_feature_names
from spotforecast.models.baselines import stack_windows


def to_table(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inputs: all features of the last history step and the future covariates of every hour. Target: the 24 spot prices."""
    return np.c_[X[:, -1, :], Y[:, :, 1:].reshape(len(Y), -1)], Y[:, :, 0]


def group_permutation_importance(model, X: np.ndarray, y: np.ndarray, groups: dict[str, list[int]], scale: float,
                                 n_repeats: int=10, seed: int=42) -> tuple[float, pd.DataFrame]:
    """MAE in EUR/MWh and its increase when the columns of a group are shuffled together across the samples."""
    rng = np.random.default_rng(seed)
    base_mae = np.abs(model.predict(X) - y).mean() * scale
    importances = {}
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, cols] = X[rng.permutation(len(X))][:, cols]
            increases.append(np.abs(model.predict(X_perm) - y).mean() * scale - base_mae)
        importances[name] = increases
    return base_mae, pd.DataFrame(importances)


def main():
    config = load_config("lstm")
    data_config = load_config("data")

    print("Loading preprocessed data...")
    train_ds, valid_ds, _ = load_datasets(config, data_config)  # the test split is not touched
    X_train, y_train = to_table(*stack_windows(train_ds))
    X_val, y_val = to_table(*stack_windows(valid_ds))

    hist_names = get_input_feature_names(config['features'])
    future_names = config['targets'][1:]
    # the hour features are the same in every window, because the history always ends at 23:00
    constant = [name for i, name in enumerate(hist_names) if X_train[:, i].min() == X_train[:, i].max()]
    if constant:
        print(f"Skipping the inputs that never change: {constant}")
    groups = {f"{name} at 23:00 D-1": [i] for i, name in enumerate(hist_names) if name not in constant}
    for k, name in enumerate(future_names):
        # the future covariates are flattened hour by hour, so covariate k of hour h is in column h * n_future + k
        groups[f"{name} day D"] = [len(hist_names) + h * len(future_names) + k for h in range(config['forecast_horizon'])]

    df_train = pd.DataFrame(X_train[:, :len(hist_names)], columns=hist_names).drop(columns=constant)
    corr_matrix = df_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated = [column for column in upper.columns if any(upper[column] > 0.95)]
    if correlated:
        print(f"Warning: these features are strongly correlated (>0.95) with another one, their importance is split:\n{correlated}")

    print(f"Training forest on {X_train.shape[0]} days with {X_train.shape[1]} input columns.")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("Calculating permutation importance on the validation split...")
    base_mae, importances = group_permutation_importance(rf, X_val, y_val, groups, scale=valid_ds.target_scaler.scale_[0])
    order = importances.mean().sort_values().index
    print(f"Forest validation MAE {base_mae:.1f} EUR/MWh, increase in EUR/MWh when an input is shuffled:")
    print(importances.mean()[order[::-1]].round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(importances[order].to_numpy(), vert=False)
    ax.set_yticklabels(order)
    ax.set_title("Increase of the validation MAE when an input is shuffled")
    ax.set_xlabel("EUR/MWh")
    ax.axvline(x=0, color="k", linestyle="--")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
