"""Summarizes logs/experiments.csv: one row per configuration with the mean and std of its runs over the seeds.

Only the runs on the data of the model config (configs/lstm.toml by default) and only the config values that differ
between the runs are shown. The features are shown as the groups of configs/feature_selection.toml a run used, plus
the names outside of these groups that it added (+) or left out (-).

    python scripts/compare_runs.py                    # runs on the course data of phase 1
    python scripts/compare_runs.py --config lstm_v2   # runs on the SMARD data of phase 2
"""
import argparse

import pandas as pd

from spotforecast.config import load_config
from spotforecast.train import EXPERIMENT_LOG

# columns that differ between the runs of one configuration
RUN_COLUMNS = ["time", "run", "commit", "seed", "best_epoch"]
METRICS = ["val_mae_eur", "val_rmse_eur", "train_mae_eur", "val_future_gain_eur", "best_epoch"]
# runs from before an option existed were trained like its default
OPTION_DEFAULTS = {"loss": "mse", "huber_delta": 1.0, "early_stopping_patience": 0, "zero_init_output": False,
                   "exclude_days": "none", "data": "data"}


def feature_label(features: str, targets: str, selection: dict) -> str:
    """The feature groups that a run used, e.g. "calendar+residual_load", from its "|"-joined features and targets."""
    features, targets = features.split("|"), targets.split("|")
    groups = selection["groups"]
    used = [name for name, group in groups.items()
            if set(group["features"]) <= set(features) and set(group["future"]) <= set(targets)]
    known = set(selection["base_features"] + selection["base_targets"])
    known |= {f for name in used for f in groups[name]["features"] + groups[name]["future"]}
    added = [f"+{f}" for f in dict.fromkeys(features + targets) if f not in known]
    missing = [f"-{f}" for f in dict.fromkeys(selection["base_features"] + selection["base_targets"])
               if f not in features + targets]
    return " ".join(["+".join(used) or "base"] + added + missing)


def main():
    parser = argparse.ArgumentParser(description="Summarizes logs/experiments.csv per configuration.")
    parser.add_argument("--config", default="lstm",
                        help="the model config: only runs on its data are shown, their features relative to it")
    parser.add_argument("--selection", default="feature_selection", help="the config of the feature groups")
    args = parser.parse_args()

    base = load_config(args.config)
    selection = load_config(args.selection)
    log = pd.read_csv(EXPERIMENT_LOG)
    log = log[log["forecast_horizon"] == base["forecast_horizon"]].copy()
    for key, default in OPTION_DEFAULTS.items():
        if key in log.columns:
            log[key] = [default if pd.isna(value) else value for value in log[key]]
    # runs on other data have other validation splits and cannot be compared
    if "data" in log.columns:
        log = log[log["data"] == base.get("data", "data")]
    log["features"] = [feature_label(f, t, selection) for f, t in zip(log["features"], log["targets"])]
    log = log.drop(columns="targets")

    config_cols = [c for c in log.columns if c not in RUN_COLUMNS and not c.startswith(("train_", "val_"))]
    keys = log[[c for c in config_cols if log[c].astype(str).nunique() > 1]].astype(str)
    if keys.empty:
        keys = pd.DataFrame({"config": "all"}, index=log.index)
    grouped = log.groupby([keys[c] for c in keys.columns])
    summary = grouped[[m for m in METRICS if m in log.columns]].agg(["mean", "std"]).round(2)
    summary[("runs", "")] = grouped.size()
    print(f"{len(log)} runs of {EXPERIMENT_LOG.name}, errors in EUR/MWh on the validation split\n")
    print(summary.sort_values(("val_mae_eur", "mean")).to_string())


if __name__ == '__main__':
    main()
