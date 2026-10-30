"""Step P5 of the roadmap: the forecasts of a split as in live operation, where the models are refitted regularly.

The split is cut into periods (the whole split, quarters or months). For every period LightGBM, the LSTM and Ridge are
fitted on all days before it, with the scaling of these days, and forecast the days of the period. With the interval
"none" they are fitted once, before the split, as in python -m spotforecast.evaluate. The LSTM trains a fixed number
of epochs without early stopping, because live operation has no validation split: the median best epoch of its
validation runs, as in python -m spotforecast.final. LightGBM and the LSTM are the means of the seeds 1 to 3; model v3
is the mean of both.

The forecasts are saved in EUR/MWh per period, model and seed in logs/walk_forward/<split>_<interval>/, so an
interrupted run continues where it stopped, and the models can run on different machines (LightGBM locally, the LSTM
on Colab) into the same folder. At the end the MAE of everything that is complete is printed and written to
logs/walk_forward/<split>_<interval>.csv, one row per hour.

    python -m spotforecast.walk_forward --models lgbm ridge --interval month    # locally, a few minutes
    python -m spotforecast.walk_forward --models lstm --interval month          # on Colab, 36 LSTM runs
"""
import argparse
import contextlib
import copy
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import EnergyDataset, build_datasets, data_paths, load_preprocessed_data
from spotforecast.evaluate import predict
from spotforecast.final import final_epochs
from spotforecast.models.baselines import fit_lgbm, fit_ridge, predict_lgbm, predict_ridge, stack_windows
from spotforecast.train import train

SEEDS = [1, 2, 3]
OUTPUT_DIR = PROJECT_ROOT / "logs" / "walk_forward"
# pandas frequencies of the period starts; "none" fits once before the split
INTERVALS = {"none": None, "quarter": "QS", "month": "MS"}


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


def collect(directory: Path, models: list[str], starts: list[pd.Timestamp]) -> pd.DataFrame:
    """One row per forecast hour with the actual price and the forecast of every model whose seeds are complete in
    every period (the mean of the seeds), plus model v3 when LightGBM and the LSTM are there."""
    complete = [m for m in models if all(cache_file(directory, m, s, start).exists()
                                         for start in starts for s in (SEEDS if m != "ridge" else [0]))]
    if not complete:
        return pd.DataFrame()
    frames = []
    for start in starts:
        columns = {}
        for model in complete:
            runs = [np.load(cache_file(directory, model, s, start)) for s in (SEEDS if model != "ridge" else [0])]
            columns[model] = np.mean([r["pred"] for r in runs], axis=0).reshape(-1)
            columns["actual"], days = runs[0]["actual"].reshape(-1), runs[0]["days"]
        horizon = len(columns["actual"]) // len(days)
        frames.append(pd.DataFrame({"day": np.repeat(days, horizon), "hour": np.tile(np.arange(horizon), len(days)),
                                    **columns}))
    table = pd.concat(frames, ignore_index=True)
    if {"lgbm", "lstm"} <= set(complete):
        table["v3"] = (table["lgbm"] + table["lstm"]) / 2
    return table


def main():
    parser = argparse.ArgumentParser(description="Forecasts of a split with models that are refitted regularly, as "
                                                 "in live operation.")
    parser.add_argument("--config", default="lstm_v3", help="the LSTM config, also for Ridge and the data")
    parser.add_argument("--lgbm", default="lgbm_v3", help="the LightGBM config")
    parser.add_argument("--models", nargs="+", default=["lgbm", "ridge"], choices=["lgbm", "lstm", "ridge"])
    parser.add_argument("--interval", default="month", choices=list(INTERVALS))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="needed for --split test, the single final evaluation")
    parser.add_argument("--epochs", type=int, help="the epochs of the LSTM, default: the median best epoch of its "
                                                   "validation runs in models/ and logs/experiments.csv")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is evaluated only once, at the very end: add --final")

    config, lgbm_config = load_config(args.config), load_config(args.lgbm)
    data_config = load_data_config(config)
    kwargs = dict(data_config["preprocessing"])
    valid_cutoff, test_cutoff = kwargs.pop("valid_cutoff_datetime"), kwargs.pop("test_cutoff_datetime")
    datetime_col = kwargs.get("datetime_col", "Date")
    configs = {"lstm": config, "lgbm": lgbm_config}  # Ridge uses the features of the LSTM
    print("Loading the data...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        df = load_preprocessed_data(data_paths(data_config),
                                    list(dict.fromkeys(c for cfg in configs.values()
                                                       for c in cfg["features"] + cfg["targets"])), kwargs)
    last_day = pd.Timestamp(df[datetime_col].max()).tz_localize(None).normalize() + pd.Timedelta(days=1)
    start, end = (pd.Timestamp(valid_cutoff), pd.Timestamp(test_cutoff)) if args.split == "val" else \
        (pd.Timestamp(test_cutoff), last_day)
    split_periods = periods(start, end, args.interval)
    directory = OUTPUT_DIR / f"{args.split}_{args.interval}"
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = (args.epochs or final_epochs(config)) if "lstm" in args.models else None
    print(f"{len(split_periods)} periods from {start:%Y-%m-%d} to {end:%Y-%m-%d}, models {args.models}, "
          f"LSTM epochs {epochs}, device {device}", flush=True)

    for period_start, period_end in split_periods:
        windows = {}  # per feature set: the datasets and stacked windows of this period
        for model in args.models:
            family = "lgbm" if model == "lgbm" else "lstm"
            for seed in SEEDS if model != "ridge" else [0]:
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
                if model == "lgbm":
                    pred = predict_lgbm(fit_lgbm(X_train, Y_train, seed, **lgbm_config.get("params", {})), X, Y)
                elif model == "ridge":
                    pred = predict_ridge(fit_ridge(train_ds), X, Y)
                else:
                    seed_config = copy.deepcopy(config)
                    seed_config.update(seed=seed, epochs=epochs, early_stopping_patience=0)
                    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                        lstm, _ = train(seed_config, train_ds, None, device, Path(tmp) / "model.pth")
                    pred = predict(lstm, period_ds, device)[0]
                pred, actual = to_eur(pred, period_ds), to_eur(Y[:, :, 0], period_ds)
                days = np.array(period_ds.forecast_starts().strftime("%Y-%m-%d"), dtype="U10")
                np.savez(path, pred=pred, actual=actual, days=days)
                print(f"  {period_start:%Y-%m-%d} {model:5s} seed {seed}: trained on {len(train_ds)} days, "
                      f"MAE {np.abs(pred - actual).mean():.2f} on {len(X)} days", flush=True)

    table = collect(directory, ["lgbm", "lstm", "ridge"], [s for s, _ in split_periods])
    if table.empty:
        return
    table.to_csv(OUTPUT_DIR / f"{args.split}_{args.interval}.csv", index=False)
    models = [c for c in ["lgbm", "lstm", "ridge", "v3"] if c in table]
    errors = table[models].sub(table["actual"], axis=0).abs()
    quarter = pd.to_datetime(table["day"]).dt.to_period("Q").astype(str)
    print(f"\n{args.split} split, models refitted: {args.interval}; MAE in EUR/MWh (lgbm, lstm: mean of the seeds; "
          f"v3: mean of lgbm and lstm)")
    print(errors.mean().round(2).to_frame("MAE").T.to_string())
    print(errors.groupby(quarter).mean().round(2).to_string())


if __name__ == '__main__':
    main()
