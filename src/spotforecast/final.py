"""Step 7 of the roadmap: trains the final model on train + validation and evaluates it once on the test split.

The final model was fixed in step 6 on the validation split: the mean of the forecasts of the LSTM of configs/lstm.toml
with the seeds FINAL_SEEDS and of Ridge, 50/50. Without a validation split there is no early stopping, so every LSTM
trains for the median best epoch of the validation runs of this config (logs/experiments.csv) and keeps its last epoch;
Ridge is refitted on the same data. The results are written to reports/test_results.md. The test split is evaluated
only once: when that file exists, the command only shows it.

    python -m spotforecast.final --dry-run    # the same procedure on train -> validation, the test split is not touched
    python -m spotforecast.final              # once, at the very end
"""
import argparse
import contextlib
import copy
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import get_feature_dims, load_datasets
from spotforecast.evaluate import MODEL_DIR, compute_metrics, diebold_mariano, predict, seed_runs
from spotforecast.models.baselines import fit_ridge, naive_forecasts, ridge_features, stack_windows
from spotforecast.train import EXPERIMENT_LOG, git_commit, train

FINAL_SEEDS = [1, 2, 3]
# a subfolder, so that the final models, which have seen the validation split, are not mixed with the validation runs
FINAL_DIR = MODEL_DIR / "final"
RESULTS = PROJECT_ROOT / "reports" / "test_results.md"
FINAL = "final model: lstm mean + ridge"


def final_epochs(config: dict) -> int:
    """The median best epoch of the validation runs of the config in models/ and logs/experiments.csv."""
    log = pd.read_csv(EXPERIMENT_LOG).set_index("run")
    best_epochs = log.loc[log.index.intersection(list(seed_runs(config))), "best_epoch"]
    if best_epochs.empty:
        raise RuntimeError("No validation runs of configs/lstm.toml found in models/ and logs/experiments.csv")
    return int(round(best_epochs.median()))


def error_table(forecasts: dict[str, np.ndarray], y: np.ndarray, ds) -> pd.DataFrame:
    """The metrics of every forecast plus the distribution of its absolute errors in EUR/MWh."""
    rows = {}
    for name, pred in forecasts.items():
        abs_err = np.abs(pred - y) * ds.target_scaler.scale_[0]
        rows[name] = {**compute_metrics(pred, y, ds), "median_ae_eur": np.median(abs_err),
                      "within_10_eur": (abs_err < 10).mean(), "within_20_eur": (abs_err < 20).mean()}
    return pd.DataFrame(rows).T


def markdown_table(table: pd.DataFrame) -> str:
    def cell(column: str, value: float) -> str:
        if column.startswith("within"):
            return f"{value:.0%}"
        return f"{value:.3f}" if column == "mse_z" else f"{value:.1f}"
    lines = ["| forecast | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(cell(c, row[c]) for c in table.columns) + " |" for name, row in table.iterrows()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Trains the final model on train + validation and evaluates it once "
                                                 "on the test split.")
    parser.add_argument("--dry-run", action="store_true",
                        help="train on the training split and evaluate on the validation split, the test split is not "
                             "touched and nothing is saved")
    parser.add_argument("--epochs", type=int, help="overrides the number of epochs, e.g. for a quick --dry-run")
    parser.add_argument("--force", action="store_true", help="evaluate the test split again although it already was")
    args = parser.parse_args()

    if not args.dry_run and RESULTS.exists() and not args.force:
        print(f"The test split was already evaluated, these are the results of {RESULTS.relative_to(PROJECT_ROOT)}:\n")
        print(RESULTS.read_text(encoding="utf-8"))
        return

    config = load_config("lstm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or final_epochs(config)
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        train_ds, valid_ds, test_ds = load_datasets(config, load_config("data"))
    if args.dry_run:
        fit_ds, eval_ds, split = train_ds, valid_ds, "validation split (dry run)"
    else:
        fit_ds, eval_ds, split = ConcatDataset([train_ds, valid_ds]), test_ds, "test split"
    print(f"Training {len(FINAL_SEEDS)} LSTMs for {epochs} epochs on {len(fit_ds)} days, device {device}...")

    X, Y = stack_windows(eval_ds)
    y = Y[:, :, 0]
    forecasts = naive_forecasts(X, y.shape[1])
    forecasts["ridge"] = fit_ridge(fit_ds).predict(ridge_features(X, Y))
    lstm = []
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) if args.dry_run else FINAL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        for seed in FINAL_SEEDS:
            seed_config = copy.deepcopy(config)
            seed_config.update(seed=seed, epochs=epochs, early_stopping_patience=0)
            model_path = model_dir / f"final_{config['model_name']}_seed{seed}.pth"
            with open(model_path.with_suffix(".json"), "w") as f:
                json.dump({"config": seed_config, "feature_dims": list(get_feature_dims(train_ds))}, f, indent=2)
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = train(seed_config, fit_ds, None, device, model_path)
            lstm.append(predict(model, eval_ds, device)[0])
            forecasts[f"lstm, seed {seed}"] = lstm[-1]
            print(f"  seed {seed} trained")
    forecasts["lstm, mean of the seeds"] = np.mean(lstm, axis=0)
    forecasts[FINAL] = (forecasts["lstm, mean of the seeds"] + forecasts["ridge"]) / 2

    table = error_table(forecasts, y, eval_ds)
    days = eval_ds.forecast_starts()
    tests = "\n".join(f"- vs {other}: statistic {stat:.2f}, p-value {p:.4f}"
                      for other in ["ridge", "lstm, mean of the seeds", "day D-1"]
                      for stat, p in [diebold_mariano(forecasts[FINAL], forecasts[other], y)])
    epochs_origin = "set with --epochs" if args.epochs else "the median best epoch of their validation runs"
    report = f"""# {"Dry run of the final procedure" if args.dry_run else "Test results"}

Day-ahead forecast of the German spot price on the {split}: {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, {len(days)} days,
evaluated on {datetime.now():%Y-%m-%d} with commit {git_commit() or 'unknown'}.

The final model was fixed in step 6 of ROADMAP.md on the validation split: the mean of the forecasts of the LSTM of
`configs/lstm.toml` with the seeds {", ".join(map(str, FINAL_SEEDS))} and of Ridge, 50/50. The LSTMs were trained on
{len(fit_ds)} days for {epochs} epochs each ({epochs_origin}), Ridge on the same days.
The inputs of day D (weather, wind, solar, residual load) are measured values used as perfect forecasts, so with real
forecasts the errors would be larger.

Errors in EUR/MWh (mse_z: in the scaled space of the training; median_ae: the median absolute error; within_10/20:
the share of hours with an error below 10 and 20 EUR/MWh):

{markdown_table(table)}

Diebold-Mariano test on the daily MAE of the final model (one-sided; a negative statistic and a small p-value mean that
the final model is more accurate):

{tests}
"""
    print("\n" + report)
    if not args.dry_run:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(report, encoding="utf-8")
        print(f"written to {RESULTS.relative_to(PROJECT_ROOT)}, the final models are in {FINAL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
