"""Step P5 of the roadmap: the forecasts of a split as in live operation, where the models are refitted regularly.

The split is cut into periods (the whole split, quarters or months). For every period LightGBM, the LSTM and Ridge are
fitted on all days before it, with the scaling of these days, and forecast the days of the period. With the interval
"none" they are fitted once, before the split, as in python -m spotforecast.evaluate. The LSTM trains a fixed number
of epochs without early stopping, because live operation has no validation split: the median best epoch of its
validation runs, as in python -m spotforecast.final. LightGBM and the LSTM are the means of the seeds 1 to 3; model v3
is the mean of both. The prices of day D-1 are kept as the naive reference.

The forecasts are saved in EUR/MWh per period, model and seed in logs/walk_forward/<split>_<interval>/, so an
interrupted run continues where it stopped, and the models can run on different machines (LightGBM locally, the LSTM
on Colab) into the same folder. At the end the MAE of everything that is complete is printed and written to
logs/walk_forward/<split>_<interval>.csv, one row per hour. The single evaluation on the test period is
python -m spotforecast.final_v3.

    python -m spotforecast.walk_forward --models lgbm ridge --interval month    # locally, a few minutes
    python -m spotforecast.walk_forward --models lstm --interval month          # on Colab, 36 LSTM runs
"""
import argparse
import contextlib
import copy
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import EnergyDataset, build_datasets, data_paths, load_preprocessed_data
from spotforecast.evaluate import predict
from spotforecast.final import final_epochs
from spotforecast.models.baselines import (day_before, fit_lgbm, fit_ridge, predict_lgbm, predict_ridge,
                                           stack_windows)
from spotforecast.train import train

SEEDS = [1, 2, 3]
OUTPUT_DIR = PROJECT_ROOT / "logs" / "walk_forward"
# pandas frequencies of the period starts; "none" fits once before the split
INTERVALS = {"none": None, "quarter": "QS", "month": "MS"}
# the forecasts that are saved: the models and the naive reference "naive", the prices of day D-1
MODELS = ["lgbm", "lstm", "ridge", "naive"]


def seeds_of(model: str) -> list[int]:
    """LightGBM and the LSTM are fitted with several seeds, Ridge and the naive reference have none."""
    return SEEDS if model in ("lgbm", "lstm") else [0]


def periods(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The periods [s, e) of the split from start to end; the models are refitted at the start of each."""
    freq = INTERVALS[interval]
    starts = [start] if freq is None else list(pd.date_range(start, end, freq=freq, inclusive="left"))
    bounds = starts + [end]
    return list(zip(bounds[:-1], bounds[1:]))


def to_eur(pred: np.ndarray, ds: EnergyDataset) -> np.ndarray:
    """Scaled spot prices in EUR/MWh; every period has its own scaling."""
    return pred * ds.target_scaler.scale_[0] + ds.target_scaler.mean_[0]


def cache_file(directory: Path, model: str, seed: int, start: pd.Timestamp) -> Path:
    return directory / f"{model}_seed{seed}_{start:%Y-%m-%d}.npz"


def collect(directory: Path, starts: list[pd.Timestamp]) -> pd.DataFrame:
    """One row per forecast hour with the actual price and the forecast of every model whose seeds are complete in
    every period (the mean of the seeds), plus model v3 when LightGBM and the LSTM are there."""
    complete = [m for m in MODELS if all(cache_file(directory, m, s, start).exists()
                                         for start in starts for s in seeds_of(m))]
    if not complete:
        return pd.DataFrame()
    frames = []
    for start in starts:
        columns = {}
        for model in complete:
            runs = [np.load(cache_file(directory, model, s, start)) for s in seeds_of(model)]
            columns[model] = np.mean([r["pred"] for r in runs], axis=0).reshape(-1)
            columns["actual"], days = runs[0]["actual"].reshape(-1), runs[0]["days"]
        horizon = len(columns["actual"]) // len(days)
        frames.append(pd.DataFrame({"day": np.repeat(days, horizon), "hour": np.tile(np.arange(horizon), len(days)),
                                    **columns}))
    table = pd.concat(frames, ignore_index=True)
    if {"lgbm", "lstm"} <= set(complete):
        table["v3"] = (table["lgbm"] + table["lstm"]) / 2
    return table


def run(config: dict, lgbm_config: dict, split: str, interval: str, models: list[str],
        epochs: Optional[int]=None) -> pd.DataFrame:
    """Fits and forecasts every period of the split that is not saved yet for the models (and the naive reference),
    then returns the table of collect, which is also written to logs/walk_forward/<split>_<interval>.csv."""
    data_config = load_data_config(config)
    kwargs = dict(data_config["preprocessing"])
    valid_cutoff, test_cutoff = kwargs.pop("valid_cutoff_datetime"), kwargs.pop("test_cutoff_datetime")
    datetime_col = kwargs.get("datetime_col", "Date")
    configs = {"lstm": config, "lgbm": lgbm_config}  # Ridge and the naive reference use the data of the LSTM
    print("Loading the data...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        df = load_preprocessed_data(data_paths(data_config),
                                    list(dict.fromkeys(c for cfg in configs.values()
                                                       for c in cfg["features"] + cfg["targets"])), kwargs)
    last_day = pd.Timestamp(df[datetime_col].max()).tz_localize(None).normalize() + pd.Timedelta(days=1)
    start, end = (pd.Timestamp(valid_cutoff), pd.Timestamp(test_cutoff)) if split == "val" else \
        (pd.Timestamp(test_cutoff), last_day)
    split_periods = periods(start, end, interval)
    directory = OUTPUT_DIR / f"{split}_{interval}"
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(split_periods)} periods from {start:%Y-%m-%d} to {end:%Y-%m-%d}, models {models}, device {device}",
          flush=True)

    for period_start, period_end in split_periods:
        windows = {}  # per feature set: the datasets and stacked windows of this period
        for model in models + ["naive"]:
            family = "lgbm" if model == "lgbm" else "lstm"
            for seed in seeds_of(model):
                path = cache_file(directory, model, seed, period_start)
                if path.exists():
                    continue
                if family not in windows:
                    cfg = configs[family]
                    cutoffs = [f"{period_start:%Y-%m-%d %H:%M:%S}", f"{period_end:%Y-%m-%d %H:%M:%S}"]
                    train_ds, period_ds, _ = build_datasets(df, cfg["features"], cfg["targets"],
                                                            cfg["forecast_history"], cfg["forecast_horizon"],
                                                            datetime_col, *cutoffs, cfg.get("exclude_days", []))
                    windows[family] = (train_ds, period_ds, stack_windows(train_ds), stack_windows(period_ds))
                train_ds, period_ds, (X_train, Y_train), (X, Y) = windows[family]
                if model == "naive":
                    pred = day_before(X, Y.shape[1])
                elif model == "lgbm":
                    pred = predict_lgbm(fit_lgbm(X_train, Y_train, seed, **lgbm_config.get("params", {})), X, Y)
                elif model == "ridge":
                    pred = predict_ridge(fit_ridge(train_ds), X, Y)
                else:
                    epochs = epochs or final_epochs(config)
                    seed_config = copy.deepcopy(config)
                    seed_config.update(seed=seed, epochs=epochs, early_stopping_patience=0)
                    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                        lstm, _ = train(seed_config, train_ds, None, device, Path(tmp) / "model.pth")
                    pred = predict(lstm, period_ds, device)[0]
                pred, actual = to_eur(pred, period_ds), to_eur(Y[:, :, 0], period_ds)
                days = np.array(period_ds.forecast_starts().strftime("%Y-%m-%d"), dtype="U10")
                np.savez(path, pred=pred, actual=actual, days=days)
                fitted = "no fit" if model == "naive" else f"trained on {len(train_ds)} days" + \
                    (f" for {epochs} epochs" if model == "lstm" else "")
                print(f"  {period_start:%Y-%m-%d} {model:5s} seed {seed}: {fitted}, "
                      f"MAE {np.abs(pred - actual).mean():.2f} on {len(X)} days", flush=True)

    table = collect(directory, [s for s, _ in split_periods])
    if not table.empty:
        table.to_csv(OUTPUT_DIR / f"{split}_{interval}.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description="Forecasts of a split with models that are refitted regularly, as "
                                                 "in live operation.")
    parser.add_argument("--config", default="lstm_v3", help="the LSTM config, also for Ridge and the data")
    parser.add_argument("--lgbm", default="lgbm_v3", help="the LightGBM config")
    parser.add_argument("--models", nargs="+", default=["lgbm", "ridge"], choices=["lgbm", "lstm", "ridge"])
    parser.add_argument("--interval", default="month", choices=list(INTERVALS))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="needed for --split test, the single final evaluation; "
                                                             "use python -m spotforecast.final_v3 for it")
    parser.add_argument("--epochs", type=int, help="the epochs of the LSTM, default: the median best epoch of its "
                                                   "validation runs in models/ and logs/experiments.csv")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is evaluated only once, at the very end: python -m spotforecast.final_v3")

    table = run(load_config(args.config), load_config(args.lgbm), args.split, args.interval, args.models, args.epochs)
    if table.empty:
        return
    models = [c for c in ["lgbm", "lstm", "ridge", "naive", "v3"] if c in table]
    errors = table[models].sub(table["actual"], axis=0).abs()
    quarter = pd.to_datetime(table["day"]).dt.to_period("Q").astype(str)
    print(f"\n{args.split} split, models refitted: {args.interval}; MAE in EUR/MWh (lgbm, lstm: mean of the seeds; "
          f"naive: the prices of day D-1; v3: mean of lgbm and lstm)")
    print(errors.mean().round(2).to_frame("MAE").T.to_string())
    print(errors.groupby(quarter).mean().round(2).to_string())


if __name__ == '__main__':
    main()
