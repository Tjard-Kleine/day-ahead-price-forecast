"""Summarizes logs/experiments.csv: one row per configuration with the mean and std of its runs over the seeds.

Only the runs of the current task (forecast horizon of configs/lstm.toml) and only the config values that differ
between the runs are shown; feature lists as the names added (+) or missing (-) compared to configs/lstm.toml.

    python scripts/compare_runs.py
"""
import pandas as pd

from spotforecast.config import load_config
from spotforecast.train import EXPERIMENT_LOG

# columns that differ between the runs of one configuration
RUN_COLUMNS = ["time", "run", "commit", "seed", "best_epoch"]
METRICS = ["val_mae_eur", "val_rmse_eur", "train_mae_eur", "val_future_gain_eur", "best_epoch"]
# runs from before an option existed were trained like its default
OPTION_DEFAULTS = {"loss": "mse", "huber_delta": 1.0, "early_stopping_patience": 0, "zero_init_output": False}


def difference(value: str, base: list[str]) -> str:
    """The names of a "|"-joined list that are added to or missing from the base list."""
    names = value.split("|")
    changes = [f"+{n}" for n in names if n not in base] + [f"-{n}" for n in base if n not in names]
    return " ".join(changes) or "as config"


def main():
    base = load_config("lstm")
    log = pd.read_csv(EXPERIMENT_LOG)
    log = log[log["forecast_horizon"] == base["forecast_horizon"]]
    log = log.fillna({k: v for k, v in OPTION_DEFAULTS.items() if k in log.columns})
    for key in ["features", "targets"]:
        log[key] = log[key].map(lambda value: difference(value, base[key]))

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
