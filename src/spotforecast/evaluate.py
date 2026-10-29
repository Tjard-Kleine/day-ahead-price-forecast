"""Compares a trained run with the baselines, errors in EUR/MWh.

    python -m spotforecast.evaluate                 # newest run in models/ on the validation split
    python -m spotforecast.evaluate --run <name>    # a specific run
    python -m spotforecast.evaluate --split test    # only once, for the final result
"""
import argparse
import json
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spotforecast.config import PROJECT_ROOT, load_config
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


def load_run(run: str, device: torch.device) -> tuple[EnergyLSTMNet, dict]:
    """Rebuilds a trained run with the model config it was trained with."""
    with open(MODEL_DIR / f"{run}.json") as f:
        info = json.load(f)
    model = build_model(info["config"], *info["feature_dims"]).to(device)
    load_model(model_path=MODEL_DIR / f"{run}.pth", model=model, optimizer=None, device=device, with_opt=False)
    model.eval()
    return model, info["config"]


def main():
    parser = argparse.ArgumentParser(description="Compares a trained run with the baselines, errors in EUR/MWh.")
    parser.add_argument("--run", help="name of a run in models/, default: the newest one")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="compare models and features on 'val', use 'test' only once at the very end")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.run or latest_run()
    if run is not None:
        model, config = load_run(run, device)
    else:
        print("No trained run in models/, evaluating only the baselines.")
        config = load_config("lstm")

    train_ds, valid_ds, test_ds = load_datasets(config, load_config("data"))
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

    table = pd.DataFrame(results).T
    print(f"\n{args.split} split, errors in EUR/MWh (mse_z: scaled space of the training loss, future_gain_eur: growth "
          f"of the MAE when the inputs of day D are shuffled)")
    print(table.round({c: 3 if c == "mse_z" else 1 for c in table.columns}).to_string())


if __name__ == '__main__':
    main()
