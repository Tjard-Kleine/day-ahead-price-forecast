"""Step P6 of the roadmap: the single evaluation of model v3 on the test period, as in live operation.

Model v3 was fixed on the validation year (ROADMAP.md, phase 3): the mean of LightGBM (configs/lgbm_v3.toml) and the
LSTM (configs/lstm_v3.toml), 3 seeds each, refitted at the start of every month on all days before it, the LSTM for a
fixed number of epochs (python -m spotforecast.walk_forward). The LSTM needs a GPU, so this runs on Colab; an
interrupted run continues with the months that are missing. The results are written to reports/test_results_v3.md.
The test period is evaluated only once: when that file exists, the command only shows it.

After the test, --export trains model v3 once on all days of the data, the next monthly refit of live operation, and
saves it to models/final_v3/. Its forecasts of days it was trained on are not an evaluation: the honest forecasts of the
test period are those of the walk-forward in logs/walk_forward/test_month.csv.

    python -m spotforecast.final_v3 --dry-run    # the report of the same procedure on the validation year
    python -m spotforecast.final_v3              # once, at the very end
    python -m spotforecast.final_v3 --export     # the model for live forecasts, on Colab (about 10 minutes)
"""
import argparse
import contextlib
import copy
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import build_datasets, get_feature_dims
from spotforecast.evaluate import HORIZON_BLOCKS, MODEL_DIR, diebold_mariano
from spotforecast.final import final_epochs, markdown_table
from spotforecast.final_v2 import commit_note
from spotforecast.models.baselines import fit_lgbm, stack_windows
from spotforecast.train import train
from spotforecast.walk_forward import SEEDS, load_frame, run

RESULTS = PROJECT_ROOT / "reports" / "test_results_v3.md"
# not models/final/ and models/final_v2/, which hold the models of v1 and v2
FINAL_DIR = MODEL_DIR / "final_v3"
# the columns of walk_forward.collect and their names in the report
FORECASTS = {"naive": "day D-1", "ridge": "ridge", "lgbm": "lightgbm", "lstm": "lstm", "v3": "model v3"}


def by_day(table: pd.DataFrame, column: str) -> np.ndarray:
    """A column of the hourly table as an array of shape (days, hours)."""
    return table.pivot(index="day", columns="hour", values=column).to_numpy()


def error_table(table: pd.DataFrame) -> pd.DataFrame:
    """The errors of every forecast in EUR/MWh; bias is the mean error (positive: too high)."""
    actual, rows = by_day(table, "actual"), {}
    for column, name in FORECASTS.items():
        err = by_day(table, column) - actual
        rows[name] = {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                      **{f"MAE {block}": np.abs(err[:, hours]).mean() for block, hours in HORIZON_BLOCKS.items()},
                      "median AE": np.median(np.abs(err)), "within 10": (np.abs(err) < 10).mean(),
                      "within 20": (np.abs(err) < 20).mean(), "bias": err.mean()}
    return pd.DataFrame(rows).T


def quarterly(table: pd.DataFrame, columns: list[str], absolute: bool=True) -> pd.DataFrame:
    """The mean (absolute) error of the forecasts (rows) in every quarter (columns)."""
    quarters = pd.to_datetime(table["day"]).dt.to_period("Q").astype(str)
    err = table[columns].sub(table["actual"], axis=0)
    err = err.abs() if absolute else err
    result = err.groupby(quarters).mean()
    days = table.groupby(quarters)["day"].nunique()
    result.index = [f"{q} ({n} days)" for q, n in days.items()]
    return result.rename(columns=FORECASTS).T


def report(table: pd.DataFrame, split: str, epochs: int, dry_run: bool) -> str:
    days = sorted(table["day"].unique())
    actual = by_day(table, "actual")
    tests = "\n".join(f"- vs {FORECASTS[other]}: statistic {stat:.2f}, p-value {p:.4f}"
                      for other in ["ridge", "lgbm", "lstm", "naive"]
                      for stat, p in [diebold_mariano(by_day(table, "v3"), by_day(table, other), actual)])
    used_before = "" if dry_run else ("\nThe test period was used once before, for the evaluation of model v2 "
                                      "(phase 2 of ROADMAP.md).")
    return f"""# {"Dry run of the final procedure of model v3" if dry_run else "Test results of model v3"}

Day-ahead forecast of the German spot price on the {split}: {days[0]} to {days[-1]}, {len(days)} days, evaluated on
{datetime.now():%Y-%m-%d} with commit {commit_note()}.

Model v3 was fixed on the validation year in phase 3 of ROADMAP.md: the mean of LightGBM (`configs/lgbm_v3.toml`)
and the LSTM (`configs/lstm_v3.toml`), each the mean of the seeds 1, 2 and 3. As in live operation both are refitted
at the start of every month on all days before it, with the scaling of these days; the LSTM trains {epochs} epochs
without early stopping (the median best epoch of its validation runs). Ridge is refitted the same way with the
features of the LSTM; day D-1 repeats the prices of the day before.

Only what is known at 12:00 on D-1 is used: the prices of D-1, the day-ahead forecasts of load, wind and solar (the
EU transparency regulation allows the wind and solar forecasts to be published until 18:00 on D-1, so the versions on
SMARD may be slightly better than those known at 12:00), the gas price (TTF settlement of D-2) and the CO2 price (the
last EUA auction before D-1). Since 1 October 2025 the auction sets quarter-hourly prices; the hourly price is their
mean.{used_before}

Errors in EUR/MWh (median AE: the median absolute error; within 10/20: the share of hours with an error below 10 and
20 EUR/MWh; bias: the mean error, positive when the forecast is too high):

{markdown_table(error_table(table))}

MAE per quarter in EUR/MWh:

{markdown_table(quarterly(table, list(FORECASTS)))}

Bias per quarter in EUR/MWh:

{markdown_table(quarterly(table, ["ridge", "lgbm", "lstm", "v3"], absolute=False))}

Diebold-Mariano test on the daily MAE of model v3 (one-sided; a negative statistic and a small p-value mean that
model v3 is more accurate):

{tests}
"""


def export(config: dict, lgbm_config: dict, epochs: int) -> None:
    """Trains LightGBM and the LSTM with every seed on all days of the data and saves them to models/final_v3/: the
    LSTM weights without optimizer state, with their config and feature dimensions (spotforecast.evaluate.load_run
    rebuilds them), and the LightGBM trees. The scaling is not stored, it is refitted on the same days ("trained_until"
    in the JSON files), which gives the same numbers."""
    df, settings = load_frame({"lstm": config, "lgbm": lgbm_config})
    until = f"{settings['end']:%Y-%m-%d %H:%M:%S}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in [("lgbm", lgbm_config), ("lstm", config)]:
        # both cutoffs at the end of the data: every complete day is a training day
        train_ds, _, _ = build_datasets(df, cfg["features"], cfg["targets"], cfg["forecast_history"],
                                        cfg["forecast_horizon"], settings["datetime_col"], until, until,
                                        cfg.get("exclude_days", []))
        days = train_ds.forecast_starts()
        print(f"{name}: training on {len(train_ds)} days, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, device {device}",
              flush=True)
        for seed in SEEDS:
            path = FINAL_DIR / f"final_{cfg['model_name']}_seed{seed}"
            if name == "lgbm":
                X, Y = stack_windows(train_ds)
                fit_lgbm(X, Y, seed, **cfg.get("params", {})).booster_.save_model(str(path.with_suffix(".txt")))
                info = {"config": cfg, "seed": seed}
            else:
                seed_config = copy.deepcopy(cfg)
                seed_config.update(seed=seed, epochs=epochs, early_stopping_patience=0)
                with contextlib.redirect_stdout(io.StringIO()):
                    model, _ = train(seed_config, train_ds, None, device, path.with_suffix(".pth"))
                torch.save(model.state_dict(), path.with_suffix(".pth"))  # without the optimizer state
                info = {"config": seed_config, "feature_dims": list(get_feature_dims(train_ds))}
            with open(path.with_suffix(".json"), "w") as f:
                json.dump({**info, "trained_until": until, "last_day": f"{days[-1]:%Y-%m-%d}"}, f, indent=2)
            print(f"  seed {seed} saved to {path.relative_to(PROJECT_ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="The single evaluation of model v3 on the test period.")
    parser.add_argument("--dry-run", action="store_true",
                        help="the same procedure on the validation year, the test period is not touched and nothing "
                             "is written to reports/")
    parser.add_argument("--force", action="store_true", help="evaluate the test period again although it already was")
    parser.add_argument("--export", action="store_true", help="after the test: train model v3 on all days and save "
                                                              "it to models/final_v3/")
    args = parser.parse_args()

    if args.export:
        if not RESULTS.exists():
            parser.error("the model is exported after the test, the export trains on the test period")
        config = load_config("lstm_v3")
        export(config, load_config("lgbm_v3"), final_epochs(config))
        return
    if not args.dry_run and RESULTS.exists() and not args.force:
        print(f"The test period was already evaluated, these are the results of {RESULTS.relative_to(PROJECT_ROOT)}:\n")
        print(RESULTS.read_text(encoding="utf-8"))
        return

    config, lgbm_config = load_config("lstm_v3"), load_config("lgbm_v3")
    epochs = final_epochs(config)
    split = "validation year (dry run)" if args.dry_run else "test period"
    table = run(config, lgbm_config, "val" if args.dry_run else "test", "month", ["lgbm", "lstm", "ridge"], epochs)
    if "v3" not in table:
        raise RuntimeError("Not every month has the forecasts of LightGBM and the LSTM")
    text = report(table, split, epochs, args.dry_run)
    print("\n" + text)
    if not args.dry_run:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(text, encoding="utf-8")
        print(f"written to {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
