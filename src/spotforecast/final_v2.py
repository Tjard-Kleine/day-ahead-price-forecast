"""Step V6 of the roadmap: trains the final model v2 on train + validation and evaluates it once on the test period.

Model v2 was fixed on the validation year: LightGBM of configs/lgbm_v2.toml, the mean of the forecasts of its seeds.
Averaging it with the LSTM seeds and Ridge was only 0.3 EUR/MWh better there, less than the 0.5 of the decision rule.
LightGBM trains a fixed number of trees, so it is refitted on train + validation without any other change; Ridge is
refitted on the same days. The results are written to reports/test_results_v2.md. The test period is evaluated only
once: when that file exists, the command only shows it.

    python -m spotforecast.final_v2 --dry-run    # the same procedure on train -> validation, the test period is not touched
    python -m spotforecast.final_v2              # once, at the very end
"""
import argparse
import contextlib
import io
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import EnergyDataset, load_datasets
from spotforecast.evaluate import MODEL_DIR, diebold_mariano
from spotforecast.final import error_table, markdown_table
from spotforecast.models.baselines import (LGBM_PARAMS, fit_lgbm, fit_ridge, naive_forecasts, predict_lgbm,
                                           ridge_features, stack_windows)
from spotforecast.train import git_commit

CONFIG = "lgbm_v2"
# not models/final/, which holds the LSTMs of model v1
FINAL_DIR = MODEL_DIR / "final_v2"
RESULTS = PROJECT_ROOT / "reports" / "test_results_v2.md"
FINAL = "final model v2: lightgbm mean of the seeds"


def quarterly_mae(forecasts: dict[str, np.ndarray], y: np.ndarray, ds: EnergyDataset) -> pd.DataFrame:
    """The MAE in EUR/MWh of the forecasts (rows) in every quarter of the forecast days (columns)."""
    quarters = [f"{d.year} Q{d.quarter}" for d in ds.forecast_starts()]
    daily = pd.DataFrame({name: (np.abs(pred - y) * ds.target_scaler.scale_[0]).mean(axis=1)
                          for name, pred in forecasts.items()})
    grouped = daily.groupby(quarters)
    table = grouped.mean()
    table.index = [f"{q} ({n} days)" for q, n in grouped.size().items()]
    return table.T


def commit_note() -> str:
    """The checked out commit, marked when the working tree has changes that are not committed."""
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        dirty = ""
    return (git_commit() or "unknown") + (" with changes that were not committed yet" if dirty else "")


def main():
    parser = argparse.ArgumentParser(description="Trains the final model v2 on train + validation and evaluates it "
                                                 "once on the test period.")
    parser.add_argument("--dry-run", action="store_true",
                        help="train on the training split and evaluate on the validation year, the test period is not "
                             "touched and nothing is saved")
    parser.add_argument("--force", action="store_true", help="evaluate the test period again although it already was")
    args = parser.parse_args()

    if not args.dry_run and RESULTS.exists() and not args.force:
        print(f"The test period was already evaluated, these are the results of {RESULTS.relative_to(PROJECT_ROOT)}:\n")
        print(RESULTS.read_text(encoding="utf-8"))
        return

    config = load_config(CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        train_ds, valid_ds, test_ds = load_datasets(config, load_data_config(config))
    if args.dry_run:
        fit_ds, eval_ds, split = train_ds, valid_ds, "validation year (dry run)"
    else:
        fit_ds, eval_ds, split = ConcatDataset([train_ds, valid_ds]), test_ds, "test period"
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(FINAL_DIR / f"final_{config['model_name']}.json", "w") as f:
            json.dump({"config": config}, f, indent=2)
    print(f"Training {len(config['seeds'])} LightGBM models on {len(fit_ds)} days...")

    X_fit, Y_fit = stack_windows(fit_ds)
    X, Y = stack_windows(eval_ds)
    y = Y[:, :, 0]
    forecasts = naive_forecasts(X, y.shape[1])
    forecasts["ridge"] = fit_ridge(fit_ds).predict(ridge_features(X, Y))
    seeds = []
    for seed in config["seeds"]:
        model = fit_lgbm(X_fit, Y_fit, seed, **config.get("params", {}))
        seeds.append(predict_lgbm(model, X, Y))
        forecasts[f"lightgbm, seed {seed}"] = seeds[-1]
        if not args.dry_run:
            model.booster_.save_model(str(FINAL_DIR / f"final_{config['model_name']}_seed{seed}.txt"))
        print(f"  seed {seed} trained")
    forecasts[FINAL] = np.mean(seeds, axis=0)

    table = error_table(forecasts, y, eval_ds)
    quarters = quarterly_mae({k: forecasts[k] for k in [FINAL, "ridge", "day D-1"]}, y, eval_ds)
    days = eval_ds.forecast_starts()
    tests = "\n".join(f"- vs {other}: statistic {stat:.2f}, p-value {p:.4f}"
                      for other in ["ridge", "day D-1"]
                      for stat, p in [diebold_mariano(forecasts[FINAL], forecasts[other], y)])
    quarter_hours = "" if args.dry_run else ("\nSince 1 October 2025 the auction sets quarter-hourly prices; the hourly "
                                             "price is their mean.")
    report = f"""# {"Dry run of the final procedure of model v2" if args.dry_run else "Test results of model v2"}

Day-ahead forecast of the German spot price with the SMARD data on the {split}: {days[0]:%Y-%m-%d} to
{days[-1]:%Y-%m-%d}, {len(days)} days, evaluated on {datetime.now():%Y-%m-%d} with commit {commit_note()}.

The final model was fixed in step V6 of ROADMAP.md on the validation year: LightGBM of `configs/lgbm_v2.toml`, the
mean of the forecasts of the seeds {", ".join(map(str, config["seeds"]))}, trained on {len(fit_ds)} days; Ridge on the
same days and features. The LightGBM settings: {json.dumps({**LGBM_PARAMS, **config.get("params", {})})}. Only what is known at 12:00 on D-1 is used: the prices of D-1 in Germany and in the coupled
neighbouring markets and the day-ahead forecasts of load, wind and solar for Germany and Austria (whether these are
published before 12:00 is still to be checked, see ROADMAP.md).{quarter_hours}

Errors in EUR/MWh (mse_z: in the scaled space; median_ae: the median absolute error; within_10/20: the share of hours
with an error below 10 and 20 EUR/MWh):

{markdown_table(table)}

MAE in EUR/MWh per quarter:

{markdown_table(quarters)}

Diebold-Mariano test on the daily MAE of the final model (one-sided; a negative statistic and a small p-value mean that
the final model is more accurate):

{tests}
"""
    print("\n" + report)
    if not args.dry_run:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(report, encoding="utf-8")
        print(f"written to {RESULTS.relative_to(PROJECT_ROOT)}, the final models are in "
              f"{FINAL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
