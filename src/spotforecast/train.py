import argparse
import json
import subprocess
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, ReduceLROnPlateau
from typing import Callable, Optional
from pathlib import Path

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import load_datasets, get_feature_dims, default_batch_transform, split_history_future_batch_transform
from spotforecast.evaluate import MODEL_DIR, evaluate_model
from spotforecast.models.lstm import build_model
from spotforecast.utils import seed_torch, load_model

EXPERIMENT_LOG = PROJECT_ROOT / "logs" / "experiments.csv"


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, writer, device, transform_batch_fn:Callable=default_batch_transform,
        save_model_path: Optional[Path]=None, scheduler: Optional[LRScheduler]=None) -> int:
    """Trains the model and returns the epoch with the best validation loss."""
    best_val_loss = torch.inf
    best_epoch = 0
    eps = 1e-4

    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        train_batch_num_acc = 0.0

        for x, y in train_loader:
            x, y = transform_batch_fn(x, y, device, train=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # x may be a tuple of model inputs, y always holds one row per sample
            train_loss_acc += loss.item() * len(y)
            train_batch_num_acc += len(y)

        avg_train_loss = train_loss_acc / train_batch_num_acc

        model.eval()
        val_loss_acc = 0.0
        val_batch_num_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = transform_batch_fn(x, y, device, train=False)

                pred = model(x)
                loss = criterion(pred, y)

                val_loss_acc += loss.item() * len(y)
                val_batch_num_acc += len(y)

        avg_val_loss = val_loss_acc / val_batch_num_acc
        if avg_val_loss + eps < best_val_loss:
            print(f"+++ Found better validation loss at epoch {epoch+1}: {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            if save_model_path is not None:
                state = {'model': model.state_dict(), 'opt': optimizer.state_dict()}
                torch.save(state, save_model_path)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        if writer is not None:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Hparams/lr", current_lr)

        # Adjust learning rate
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    return best_epoch


def log_experiment(run: str, config: dict, best_epoch: int, metrics: dict[str, dict[str, float]]):
    """Appends one row with the config and the results of a run to logs/experiments.csv."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    row = {"time": datetime.now().isoformat(timespec="seconds"), "run": run, "commit": commit, "best_epoch": best_epoch}
    for key, value in config.items():
        if isinstance(value, dict):
            row.update(value)
        elif isinstance(value, list):
            row[key] = "|".join(value)
        else:
            row[key] = value
    for split, split_metrics in metrics.items():
        row.update({f"{split}_{name}": value for name, value in split_metrics.items()})

    EXPERIMENT_LOG.parent.mkdir(exist_ok=True)
    log = pd.read_csv(EXPERIMENT_LOG) if EXPERIMENT_LOG.exists() else pd.DataFrame()
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(EXPERIMENT_LOG, index=False)


def main():
    parser = argparse.ArgumentParser(description="Trains the LSTM described by configs/lstm.toml.")
    parser.add_argument("--seed", type=int, help="overrides the seed of the config, e.g. to measure the run-to-run noise")
    parser.add_argument("--epochs", type=int, help="overrides the number of epochs of the config, e.g. for a quick test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on device: {device}")

    config = load_config("lstm")
    data_config = load_config("data")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["epochs"] = args.epochs

    # N.B.: using `42` is never a good idea!
    seed_torch(config["seed"], reproducible=True)
    _rng_cpu = torch.Generator()
    _rng_cpu.manual_seed(config["seed"])

    # every run gets its own weights, config and logs, so runs never overwrite each other
    run = f"{config['model_name']}_{datetime.now():%Y%m%d-%H%M%S}_seed{config['seed']}"
    print(f"run: {run}")
    writer = SummaryWriter(str(PROJECT_ROOT / "logs" / run))

    # the test split is not touched during development
    train_ds, valid_ds, _ = load_datasets(config, data_config)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=torch.cuda.is_available(), generator=_rng_cpu)
    val_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=torch.cuda.is_available(), generator=_rng_cpu)

    n_hist_features, n_future_features = get_feature_dims(train_ds)
    model = build_model(config, n_hist_features, n_future_features, rng=_rng_cpu).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    criterion = nn.MSELoss(reduction="mean")
    batch_tsfm_fn = split_history_future_batch_transform
    save_model_path = MODEL_DIR / f"{run}.pth"
    with open(save_model_path.with_suffix(".json"), "w") as f:
        json.dump({"config": config, "feature_dims": [n_hist_features, n_future_features]}, f, indent=2)

    best_epoch = fit(model, train_loader, val_loader, optimizer, criterion, config['epochs'], writer, device,
                     transform_batch_fn=batch_tsfm_fn, save_model_path=save_model_path, scheduler=scheduler)

    print(f"Loading best model...\n")
    load_model(model_path=save_model_path, model=model, optimizer=optimizer, device=device)

    writer.close()

    # without dropout, so that train and val errors are comparable
    metrics = {"train": evaluate_model(model, train_ds, device), "val": evaluate_model(model, valid_ds, device)}
    log_experiment(run, config, best_epoch, metrics)
    for split, m in metrics.items():
        print(f"{split:5}: MAE {m['mae_eur']:.1f} EUR/MWh, RMSE {m['rmse_eur']:.1f} EUR/MWh, MSE (scaled) {m['mse_z']:.3f}")
    print(f"best epoch {best_epoch}, results added to {EXPERIMENT_LOG.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
