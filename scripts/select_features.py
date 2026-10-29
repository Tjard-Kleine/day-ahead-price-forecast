"""Which groups of features improve the forecast of the 24 prices of day D?

Every combination of the groups in configs/feature_selection.toml is added to its base features and future covariates.
Ridge and LightGBM are trained on everything before a fold and evaluated on the fold (rolling
origin), so a group has to help in several quarters and not only in the volatile Q4 2024. Both models fit in seconds;
the LSTM only confirms the best combinations afterwards.

    python scripts/select_features.py              # about 11 min on a laptop, writes logs/feature_selection.csv
    python scripts/select_features.py --summary    # only prints the summary of the saved results again
    python scripts/select_features.py --config lstm_v2 --selection feature_selection_v2    # model v2, SMARD data
"""
import argparse
import contextlib
import io
import itertools
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import build_datasets, load_preprocessed_data
from spotforecast.data.features import add_feature_groups
from spotforecast.models.baselines import ridge_features, stack_windows

RESULTS = PROJECT_ROOT / "logs" / "feature_selection.csv"


def lgbm_table(X: np.ndarray, Y: np.ndarray, n_lags: int=168) -> tuple[np.ndarray, np.ndarray]:
    """One row per day and hour: the spot price of the last n_lags hours and all features of the last history hour
    (the same for every hour of a day), then the hour, the price of the same hour on D-1 and D-7 and the future
    covariates of that hour. A tree cannot pick the matching lag for an hour by itself, so those are added."""
    n, horizon = Y.shape[:2]
    hours = np.arange(horizon)
    day_part = np.repeat(np.c_[X[:, -n_lags:, 0], X[:, -1, :]], horizon, axis=0)
    hour_part = np.c_[np.tile(hours, n), X[:, -24 + hours, 0].reshape(-1), X[:, -168 + hours, 0].reshape(-1)]
    return np.c_[day_part, hour_part, Y[:, :, 1:].reshape(n * horizon, -1)], Y[:, :, 0].reshape(-1)


def fold_maes(df: pd.DataFrame, config: dict, features: list[str], targets: list[str], folds: list[list[str]],
              datetime_col: str) -> dict[str, list[float]]:
    """Validation MAE in EUR/MWh of Ridge and LightGBM in every fold."""
    maes = {"ridge": [], "lgbm": []}
    for start, end in folds:
        train_ds, valid_ds, _ = build_datasets(df, features, targets, config["forecast_history"],
                                               config["forecast_horizon"], datetime_col, start, end,
                                               config.get("exclude_days", []))
        (X_train, Y_train), (X_val, Y_val) = stack_windows(train_ds), stack_windows(valid_ds)
        scale = valid_ds.target_scaler.scale_[0]

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(ridge_features(X_train, Y_train), Y_train[:, :, 0])
        pred = ridge.predict(ridge_features(X_val, Y_val))
        maes["ridge"].append(float(np.abs(pred - Y_val[:, :, 0]).mean() * scale))

        lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=1, verbose=-1)
        lgbm.fit(*lgbm_table(X_train, Y_train))
        X_val_table, y_val_table = lgbm_table(X_val, Y_val)
        maes["lgbm"].append(float(np.abs(lgbm.predict(X_val_table) - y_val_table).mean() * scale))
    return maes


def forward_selection(results: pd.DataFrame, model: str, group_names: list[str], min_better_folds: int,
                      min_gain: float) -> list[str]:
    """Adds one group at a time, the one that lowers the mean MAE the most, as long as it lowers it by at least
    min_gain and is better in at least min_better_folds folds. Every combination is already evaluated, so this only
    walks through the results."""
    folds = [c for c in results.columns if c.startswith(f"{model}_fold")]
    chosen = []
    while True:
        current = results.loc["+".join(chosen) or "-", folds]
        candidates = {}
        for group in group_names:
            if group not in chosen:
                combo = "+".join(g for g in group_names if g in chosen + [group])
                fold_gain = current.to_numpy() - results.loc[combo, folds].to_numpy()
                if fold_gain.mean() >= min_gain and (fold_gain > 0).sum() >= min_better_folds:
                    candidates[group] = fold_gain.mean()
        if not candidates:
            return chosen
        best = max(candidates, key=candidates.get)
        print(f"  + {best:15s} mean MAE {results.loc['+'.join(chosen) or '-', f'{model}_mean']:.2f} -> "
              f"{results.loc['+'.join(g for g in group_names if g in chosen + [best]), f'{model}_mean']:.2f} EUR/MWh")
        chosen.append(best)
        chosen.sort(key=group_names.index)


def evaluate_combinations(config: dict, data_config: dict, selection: dict, results_path: Path) -> pd.DataFrame:
    """Fold MAEs of Ridge and LightGBM for every combination of the groups, one row per combination. The results are
    saved to results_path after every combination, and a combination that is already there is not evaluated again, so
    an interrupted run continues where it stopped (delete the file after changing groups or folds)."""
    groups = selection["groups"]
    group_names = list(groups)
    preprocess_kwargs = dict(data_config["preprocessing"])
    preprocess_kwargs.pop("valid_cutoff_datetime")
    preprocess_kwargs.pop("test_cutoff_datetime")
    datetime_col = preprocess_kwargs.get("datetime_col", "Date")
    all_features, all_targets = add_feature_groups(selection["base_features"], selection["base_targets"], groups,
                                                   group_names)
    columns = list(dict.fromkeys(all_features + all_targets))
    print("Loading and preprocessing the data once for all groups...")
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        df = load_preprocessed_data(PROJECT_ROOT / data_config["path"], columns, preprocess_kwargs)

    combos = [c for r in range(len(group_names) + 1) for c in itertools.combinations(group_names, r)]
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done = {row["groups"] for row in rows}
    if done:
        print(f"{len(done)} of {len(combos)} combinations are already in {results_path.name}, they are skipped")
    results_path.parent.mkdir(exist_ok=True)
    started = time.time()
    for i, combo in enumerate(combos, 1):
        if ("+".join(combo) or "-") in done:
            continue
        features, targets = add_feature_groups(selection["base_features"], selection["base_targets"], groups, combo)
        maes = fold_maes(df, config, features, targets, selection["folds"], datetime_col)
        row = {"groups": "+".join(combo) or "-"}
        for model, values in maes.items():
            row.update({f"{model}_fold{k + 1}": v for k, v in enumerate(values)})
            row[f"{model}_mean"] = float(np.mean(values))
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"[{i}/{len(combos)}, {time.time() - started:.0f} s] {row['groups']:60s} "
              f"Ridge {row['ridge_mean']:.2f}, LightGBM {row['lgbm_mean']:.2f} EUR/MWh", flush=True)
    return pd.DataFrame(rows).set_index("groups")


def main():
    parser = argparse.ArgumentParser(description="Evaluates every combination of the feature groups with Ridge and LightGBM.")
    parser.add_argument("--summary", action="store_true", help="only summarize the saved results again")
    parser.add_argument("--config", default="lstm", help="the model config, it gives the data and the history length")
    parser.add_argument("--selection", default="feature_selection",
                        help="the config of the groups and folds; the results go to logs/<selection>.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    selection = load_config(args.selection)
    group_names = list(selection["groups"])
    results_path = RESULTS.with_name(f"{args.selection}.csv")

    if args.summary:
        results = pd.read_csv(results_path, index_col="groups")
    else:
        results = evaluate_combinations(config, load_data_config(config), selection, results_path)
        results.to_csv(results_path)
    combos = [c for r in range(len(group_names) + 1) for c in itertools.combinations(group_names, r)]

    print(f"\nMean validation MAE over {len(selection['folds'])} folds in EUR/MWh, results in "
          f"{results_path.relative_to(PROJECT_ROOT)}")
    for model in ["ridge", "lgbm"]:
        folds = [c for c in results.columns if c.startswith(f"{model}_fold")]
        print(f"\n{model}: without groups {results.loc['-', f'{model}_mean']:.2f}, best combinations:")
        print(results.sort_values(f"{model}_mean")[folds + [f"{model}_mean"]].head(5).round(2).to_string())

        # how much a group helps on average when it is added to any combination without it
        print(f"{model}: effect of adding a group to any other combination")
        for group in group_names:
            without = [c for c in combos if group not in c]
            gains = np.array([results.loc["+".join(c) or "-", folds].to_numpy() -
                              results.loc["+".join(g for g in group_names if g in c + (group,)), folds].to_numpy()
                              for c in without])
            print(f"  {group:15s} mean MAE {-gains.mean():+.2f} EUR/MWh, better in {(gains > 0).mean():.0%} of the folds")

        print(f"{model}: forward selection (a group needs to lower the mean MAE by at least {selection['min_gain']} "
              f"EUR/MWh and to be better in at least {selection['min_better_folds']} folds)")
        chosen = forward_selection(results, model, group_names, selection["min_better_folds"], selection["min_gain"])
        print(f"  selected: {chosen or 'no group'}")


if __name__ == '__main__':
    main()
