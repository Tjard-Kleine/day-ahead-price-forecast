"""Compares a trained run with the baselines, errors in EUR/MWh.

    python -m spotforecast.evaluate                 # newest run in models/ on the validation split
    python -m spotforecast.evaluate --run <name>    # a specific run
    python -m spotforecast.evaluate --split test    # only once, for the final result
"""
import argparse
import json
from typing import Optional

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
HORIZON_BLOCKS = {"h1-6": slice(0, 6), "h7-24": slice(6, 24), "h25-48": slice(24, None)}


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
    results["ridge"] = compute_metrics(fit_ridge(train_ds).predict(ridge_features(X, Y)), y, ds)
    if run is not None:
        results[run] = evaluate_model(model, ds, device)

    table = pd.DataFrame(results).T
    print(f"\n{args.split} split, errors in EUR/MWh (mse_z: scaled space of the training loss)")
    print(table.round({c: 3 if c == "mse_z" else 1 for c in table.columns}).to_string())


if __name__ == '__main__':
    main()
