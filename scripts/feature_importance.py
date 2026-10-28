import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from spotforecast.config import load_config
from spotforecast.data.dataset import load_datasets
from spotforecast.data.features import get_input_feature_names


def main():
    config = load_config("lstm")
    data_config = load_config("data")

    print("Loading preprocessed data...")
    train_ds, valid_ds = load_datasets(config, data_config)

    def ds_to_numpy(ds):
        X_list, y_list = [], []
        step = 10
        for i in range(0, len(ds), step):
            x, y = ds[i]
            X_list.append(x[-1, :].numpy())
            y_list.append(y[0].numpy())
        return np.array(X_list), np.array(y_list)

    X_train, y_train = ds_to_numpy(train_ds)
    X_val, y_val = ds_to_numpy(valid_ds)
    X_train[np.isinf(X_train)] = np.nan
    X_val[np.isinf(X_val)] = np.nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    feature_names = get_input_feature_names(config['features'])

    if X_train.shape[1] > len(feature_names):
        print(f"Achtung: Datensatz hat {X_train.shape[1]} Spalten, aber nur {len(feature_names)} Namen in Config.")
        missing_count = X_train.shape[1] - len(feature_names)
        feature_names += [f"Extra_{i}" for i in range(missing_count)]

    df_train = pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]])
    corr_matrix = df_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    if to_drop:
        print(f"WARNUNG: Folgende Features sind extrem stark korreliert (>0.95) und verfälschen die Importance:\n{to_drop}")

    print(f"Training Forest on {X_train.shape[0]} points with {X_train.shape[1]} features.")

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("Calculating Permutation Importance...")
    result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()

    top_n = 30
    sorted_idx = sorted_idx[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title(f"Top {top_n} Feature Importances (Permutation on Val Set)")
    ax.axvline(x=0, color="k", linestyle="--")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
