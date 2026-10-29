"""Compares a trained run with the baselines, errors in EUR/MWh.

    python -m spotforecast.evaluate                 # newest run in models/ on the validation split
    python -m spotforecast.evaluate --run <name>    # a specific run
    python -m spotforecast.evaluate --split test    # only once, for the final result
    python -m spotforecast.evaluate --ensemble      # mean forecasts of the runs with the features of configs/lstm.toml
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import EnergyDataset, load_datasets, split_history_future_batch_transform
from spotforecast.models.baselines import fit_ridge, naive_forecasts, ridge_features, stack_windows
from spotforecast.models.lstm import EnergyLSTMNet, build_model
from spotforecast.utils import load_model

MODEL_DIR = PROJECT_ROOT / "models"
# the forecast starts at midnight, so the horizon steps are the hours of the forecast day
HORIZON_BLOCKS = {"00-06": slice(0, 6), "06-12": slice(6, 12), "12-18": slice(12, 18), "18-24": slice(18, 24)}


def compute_metrics(pred: np.ndarray, y: np.ndarray, ds: EnergyDataset) -> dict[str, float]:
    """Errors of scaled spot price forecasts of shape (n, horizon), in EUR/MWh and in the scaled space of the loss."""
    err_z = pred - y
    err = err_z * ds.target_scaler.scale_[0]  # the spot price is the first target
    metrics = {
        "mse_z": (err_z ** 2).mean(),
        "mae_eur": np.abs(err).mean(),
        "rmse_eur": np.sqrt((err ** 2).mean()),
    }
    for name, hours in HORIZON_BLOCKS.items():
        metrics[f"mae_eur_{name}"] = np.abs(err[:, hours]).mean()
    return {k: float(v) for k, v in metrics.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, ds: EnergyDataset, device: torch.device, batch_size: int=256) -> tuple[np.ndarray, np.ndarray]:
    """Scaled spot price forecasts and targets for all windows of a dataset, both of shape (n, horizon)."""
    model.eval()
    preds, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch_size):
        x, y = split_history_future_batch_transform(x, y, device, train=False)
        preds.append(model(x).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


def evaluate_model(model: torch.nn.Module, ds: EnergyDataset, device: torch.device) -> dict[str, float]:
    return compute_metrics(*predict(model, ds, device), ds)


def model_forecast(model: torch.nn.Module, device: torch.device) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The model as a function of stacked windows (X, Y) that returns the scaled spot price forecasts."""
    @torch.no_grad()
    def forecast(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        model.eval()
        x, _ = split_history_future_batch_transform(torch.from_numpy(X), torch.from_numpy(Y), device, train=False)
        return model(x).cpu().numpy()
    return forecast


def future_shuffle_gain(forecast: Callable[[np.ndarray, np.ndarray], np.ndarray], X: np.ndarray, Y: np.ndarray,
                        ds: EnergyDataset, n_repeats: int=10, seed: int=1) -> float:
    """How much the MAE in EUR/MWh grows when the inputs of the forecast day (the future covariates in Y) are shuffled
    between the days, i.e. how much the forecast relies on them."""
    if Y.shape[-1] == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    mae = compute_metrics(forecast(X, Y), Y[:, :, 0], ds)["mae_eur"]
    shuffled = []
    for _ in range(n_repeats):
        Y_shuffled = Y.copy()
        Y_shuffled[:, :, 1:] = Y[rng.permutation(len(Y)), :, 1:]
        shuffled.append(compute_metrics(forecast(X, Y_shuffled), Y[:, :, 0], ds)["mae_eur"])
    return float(np.mean(shuffled) - mae)


def latest_run() -> Optional[str]:
    """Name of the run in models/ whose weights were saved last."""
    runs = [p for p in MODEL_DIR.glob("*.pth") if p.with_suffix(".json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime).stem if runs else None


def load_run(run: str, device: torch.device, model_dir: Path=MODEL_DIR) -> tuple[EnergyLSTMNet, dict]:
    """Rebuilds a trained run with the model config it was trained with."""
    with open(model_dir / f"{run}.json") as f:
        info = json.load(f)
    model = build_model(info["config"], *info["feature_dims"]).to(device)
    load_model(model_path=model_dir / f"{run}.pth", model=model, optimizer=None, device=device, with_opt=False)
    model.eval()
    return model, info["config"]


def runs_with_features(config: dict) -> dict[str, dict]:
    """The runs in models/ that were trained on the features, targets and forecast horizon of the config, with their
    configs."""
    runs = {}
    for path in sorted(MODEL_DIR.glob("*.json")):
        with open(path) as f:
            run_config = json.load(f)["config"]
        if path.with_suffix(".pth").exists() and all(run_config.get(k) == config[k]
                                                      for k in ["features", "targets", "forecast_horizon"]):
            runs[path.stem] = run_config
    return runs


def seed_runs(config: dict) -> dict[str, dict]:
    """The runs in models/ with exactly the config apart from the seed."""
    without_seed = lambda c: {k: v for k, v in c.items() if k != "seed"}
    return {r: c for r, c in runs_with_features(config).items() if without_seed(c) == without_seed(config)}


def diebold_mariano(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test on the daily mean absolute errors of two forecasts of shape (days, hours), the variant for
    day-ahead prices of Lago et al. (2021). Returns the statistic and the one-sided p-value of "a is more accurate than
    b"; a negative statistic means a has the smaller errors."""
    loss_diff = np.abs(pred_a - y).mean(axis=1) - np.abs(pred_b - y).mean(axis=1)
    statistic = loss_diff.mean() / np.sqrt(loss_diff.var(ddof=1) / len(loss_diff))
    return float(statistic), float(norm.cdf(statistic))


def mean_forecast(runs: dict[str, dict], data_config: dict, split: str, device: torch.device
                  ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The mean of the scaled spot price forecasts of several runs with the same features, and the forecast days. The
    runs may differ in their history length, the days they forecast are the same."""
    datasets, preds, days = {}, [], None
    for run, run_config in runs.items():
        history = run_config["forecast_history"]
        if history not in datasets:
            with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
                _, valid_ds, test_ds = load_datasets(run_config, data_config)
            datasets[history] = valid_ds if split == "val" else test_ds
        ds = datasets[history]
        if days is not None and not days.equals(ds.forecast_starts()):
            raise ValueError(f"Run {run} forecasts other days than the runs before it")
        days = ds.forecast_starts()
        model, _ = load_run(run, device)
        preds.append(predict(model, ds, device)[0])
    return np.mean(preds, axis=0), days


def main():
    parser = argparse.ArgumentParser(description="Compares a trained run with the baselines, errors in EUR/MWh.")
    parser.add_argument("--run", help="name of a run in models/, default: the newest one")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="compare models and features on 'val', use 'test' only once at the very end")
    parser.add_argument("--ensemble", action="store_true",
                        help="instead of a single run: the mean forecasts of the runs in models/ with the features of "
                             "the model config, of the seeds of its exact config and of all these runs, each also "
                             "averaged with Ridge")
    parser.add_argument("--config", default="lstm",
                        help="the model config for --ensemble or without any run, e.g. lstm_v2 for the SMARD data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = None if args.ensemble else args.run or latest_run()
    if run is not None:
        model, config = load_run(run, device)
    else:
        if not args.ensemble:
            print("No trained run in models/, evaluating only the baselines.")
        config = load_config(args.config)
    data_config = load_data_config(config)

    train_ds, valid_ds, test_ds = load_datasets(config, data_config)
    ds = valid_ds if args.split == "val" else test_ds
    X, Y = stack_windows(ds)
    y = Y[:, :, 0]

    results = {name: compute_metrics(pred, y, ds) for name, pred in naive_forecasts(X, y.shape[1]).items()}
    ridge = fit_ridge(train_ds)
    forecasts = {"ridge": lambda X_, Y_: ridge.predict(ridge_features(X_, Y_))}
    if run is not None:
        forecasts[run] = model_forecast(model, device)
    for name, forecast in forecasts.items():
        results[name] = compute_metrics(forecast(X, Y), y, ds)
        results[name]["future_gain_eur"] = future_shuffle_gain(forecast, X, Y, ds)

    if args.ensemble:
        # averaging the forecasts of several runs smooths out the noise of the single runs
        runs = runs_with_features(config)
        seeds = seed_runs(config)
        ridge_pred = forecasts["ridge"](X, Y)
        for name, members in [(f"lstm, seeds of the config ({len(seeds)} runs)", seeds),
                              (f"lstm, all runs with these features ({len(runs)})", runs)]:
            if members:
                pred, days = mean_forecast(members, data_config, args.split, device)
                if not days.equals(ds.forecast_starts()):
                    raise ValueError("The runs forecast other days than the baselines")
                results[name] = compute_metrics(pred, y, ds)
                # LSTM and Ridge make different errors; a fixed 50/50 weight, so that nothing is fitted on the split
                results[f"{name} + ridge"] = compute_metrics((pred + ridge_pred) / 2, y, ds)

    table = pd.DataFrame(results).T
    print(f"\n{args.split} split, errors in EUR/MWh (mse_z: scaled space of the training loss, future_gain_eur: growth "
          f"of the MAE when the inputs of day D are shuffled)")
    print(table.round({c: 3 if c == "mse_z" else 1 for c in table.columns}).to_string())


if __name__ == '__main__':
    main()
