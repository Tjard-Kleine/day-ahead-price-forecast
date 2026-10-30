"""Step P4 of the roadmap: Ridge, LightGBM and their mean on the validation year, with several feature sets and two
targets, the price level and the change against day D-1.

Every model is fitted on the training split and evaluated on the whole validation year, like
python -m spotforecast.evaluate. LightGBM has the settings of the feature selection (LGBM_PARAMS) and one seed; its
seeds differ by about 0.05 EUR/MWh. The LSTM is trained on Colab with the same feature groups
(python -m spotforecast.train --config lstm_v3 --selection feature_selection_v3 --groups ...).

    python scripts/screen_models.py    # about 3 minutes -> logs/screen_models_v3.csv
"""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import build_datasets, data_paths, load_preprocessed_data
from spotforecast.data.features import add_feature_groups
from spotforecast.evaluate import compute_metrics
from spotforecast.models.baselines import fit_lgbm, fit_ridge, predict_lgbm, predict_ridge, stack_windows

# the groups of the selection config that are added to its base
FEATURE_SETS = {
    "base": [],
    "base + fuel": ["fuel"],
    "base + neighbours": ["neighbour_prices", "austria"],
    "base + neighbours + fuel": ["neighbour_prices", "austria", "fuel"],
}
TARGETS = {"level": False, "change vs D-1": True}


def main():
    parser = argparse.ArgumentParser(description="Ridge, LightGBM and their mean on the validation year with several "
                                                 "feature sets and targets.")
    parser.add_argument("--config", default="lstm_v3", help="the model config of the data, window and horizon")
    parser.add_argument("--selection", default="feature_selection_v3", help="the config of the feature groups")
    parser.add_argument("--output", default="logs/screen_models_v3.csv")
    args = parser.parse_args()

    config, selection = load_config(args.config), load_config(args.selection)
    data_config = load_data_config(config)
    kwargs = dict(data_config["preprocessing"])
    valid_cutoff, test_cutoff = kwargs.pop("valid_cutoff_datetime"), kwargs.pop("test_cutoff_datetime")
    sets = {name: add_feature_groups(selection["base_features"], selection["base_targets"], selection["groups"], groups)
            for name, groups in FEATURE_SETS.items()}
    print("Loading the data...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        df = load_preprocessed_data(data_paths(data_config),
                                    list(dict.fromkeys(c for f, t in sets.values() for c in f + t)), kwargs)

    rows = []
    for name, (features, targets) in sets.items():
        train_ds, valid_ds, _ = build_datasets(df, features, targets, config["forecast_history"],
                                               config["forecast_horizon"], kwargs.get("datetime_col", "Date"),
                                               valid_cutoff, test_cutoff, config.get("exclude_days", []))
        (X_train, Y_train), (X, Y) = stack_windows(train_ds), stack_windows(valid_ds)
        y = Y[:, :, 0]
        quarters = [f"{d.year} Q{d.quarter}" for d in valid_ds.forecast_starts()]
        for target, relative in TARGETS.items():
            preds = {"ridge": predict_ridge(fit_ridge(train_ds, relative), X, Y, relative),
                     "lightgbm": predict_lgbm(fit_lgbm(X_train, Y_train, relative=relative), X, Y, relative)}
            # equal fixed weights, so that nothing is fitted on the validation year
            preds["mean of both"] = (preds["ridge"] + preds["lightgbm"]) / 2
            for model, pred in preds.items():
                metrics = compute_metrics(pred, y, valid_ds)
                daily = pd.Series((np.abs(pred - y) * valid_ds.target_scaler.scale_[0]).mean(axis=1))
                rows.append({"features": name, "target": target, "model": model, "mae": metrics["mae_eur"],
                             "rmse": metrics["rmse_eur"], **daily.groupby(quarters).mean().to_dict()})
                print(f"{name:26s} {target:14s} {model:13s} MAE {metrics['mae_eur']:6.2f}  RMSE {metrics['rmse_eur']:6.2f}",
                      flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(PROJECT_ROOT / args.output, index=False)
    print("\nMAE on the validation year in EUR/MWh:")
    print(table.pivot_table(index=["features", "model"], columns="target", values="mae", sort=False).round(2).to_string())
    print(f"\nMAE per quarter in EUR/MWh, written to {args.output}:")
    print(table.drop(columns="rmse").set_index(["features", "target", "model"]).round(1).to_string())


if __name__ == '__main__':
    main()
