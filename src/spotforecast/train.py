import argparse
import json
import subprocess
import tomllib
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from typing import Callable, Optional
from pathlib import Path

from spotforecast.config import PROJECT_ROOT, load_config, load_data_config
from spotforecast.data.dataset import (EnergyDataset, load_datasets, get_feature_dims, default_batch_transform,
                                       split_history_future_batch_transform)
from spotforecast.data.features import add_feature_groups
from spotforecast.evaluate import MODEL_DIR, evaluate_model, future_shuffle_gain, model_forecast
from spotforecast.models.baselines import stack_windows
from spotforecast.models.lstm import build_model
from spotforecast.utils import seed_torch, load_model

EXPERIMENT_LOG = PROJECT_ROOT / "logs" / "experiments.csv"


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, writer, device, transform_batch_fn:Callable=default_batch_transform,
        save_model_path: Optional[Path]=None, scheduler: Optional[LRScheduler]=None, patience: int=0,
        epoch_callback: Optional[Callable[[int, nn.Module], None]]=None) -> int:
    """Trains the model and returns the epoch with the best validation loss. With patience > 0 the training stops after
    that many epochs without a better validation loss. epoch_callback(epoch, model) is called after every epoch, e.g.
    to stop a hyperparameter search trial early by raising an exception. Without a val_loader (the final training on
    train + validation) all epochs are trained and the model of the last one is kept."""
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

        if val_loader is not None:
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

        val_info = f" | Val loss: {avg_val_loss:.4f}" if val_loader is not None else ""
        print(f"Epoch {epoch + 1}/{epochs} | Train loss: {avg_train_loss:.4f}{val_info} | LR: {current_lr:.2e}")

        if writer is not None:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            if val_loader is not None:
                writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Hparams/lr", current_lr)

        # Adjust learning rate, without a validation split by the training loss
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss if val_loader is not None else avg_train_loss)
            else:
                scheduler.step()

        if epoch_callback is not None:
            epoch_callback(epoch + 1, model)

        if patience > 0 and val_loader is not None and epoch + 1 - best_epoch >= patience:
            print(f"Early stopping: no better validation loss for {patience} epochs")
            break

    if val_loader is None:
        best_epoch = epochs
        if save_model_path is not None:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, save_model_path)

    return best_epoch


def build_criterion(config: dict) -> nn.Module:
    """The training loss of the config: "mse", or "huber", which grows only linearly for errors above huber_delta
    (in the scaled space) and lets the price spikes weigh less."""
    if config["loss"] == "huber":
        return nn.HuberLoss(delta=config["huber_delta"])
    return nn.MSELoss(reduction="mean")


def set_config_value(config: dict, key: str, value) -> None:
    """Sets an existing config value in place, nested keys are joined by dots, e.g. "net_params.dropout_p"."""
    *parents, name = key.strip().split(".")
    section = config
    for parent in parents:
        section = section[parent]
    if name not in section:
        raise KeyError(f"Unknown config key: {key}")
    section[name] = value


def apply_overrides(config: dict, overrides: list[str]) -> None:
    """Sets config values in place from "key=value" strings, e.g. "lr=1e-3", "loss=huber" or
    "net_params.dropout_p=0.2". The value is read as TOML, anything else is taken as a string."""
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            value = tomllib.loads(f"value = {value}")["value"]
        except tomllib.TOMLDecodeError:
            pass
        set_config_value(config, key, value)


def train(config: dict, train_ds: Dataset, valid_ds: Optional[EnergyDataset], device: torch.device, save_model_path: Path,
          writer: Optional[SummaryWriter]=None, epoch_callback: Optional[Callable[[int, nn.Module], None]]=None
          ) -> tuple[nn.Module, int]:
    """Builds the model of the config and trains it. Returns it with the weights of its best epoch on the validation
    split, which are saved to save_model_path, and that epoch. Without a valid_ds all epochs of the config are trained
    and the last one is kept."""
    # N.B.: using `42` is never a good idea!
    seed_torch(config["seed"], reproducible=True)
    _rng_cpu = torch.Generator()
    _rng_cpu.manual_seed(config["seed"])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=torch.cuda.is_available(), generator=_rng_cpu)
    val_loader = None
    if valid_ds is not None:
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

    best_epoch = fit(model, train_loader, val_loader, optimizer, build_criterion(config), config['epochs'], writer, device,
                     transform_batch_fn=split_history_future_batch_transform, save_model_path=save_model_path,
                     scheduler=scheduler, patience=config['early_stopping_patience'], epoch_callback=epoch_callback)

    print(f"Loading best model...\n")
    load_model(model_path=save_model_path, model=model, optimizer=optimizer, device=device)
    return model, best_epoch


def git_commit() -> str:
    """The short hash of the checked out commit, empty if git is not available."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def log_experiment(run: str, config: dict, best_epoch: int, metrics: dict[str, dict[str, float]]):
    """Appends one row with the config and the results of a run to logs/experiments.csv."""
    row = {"time": datetime.now().isoformat(timespec="seconds"), "run": run, "commit": git_commit(),
           "best_epoch": best_epoch}
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
    parser.add_argument("--config", default="lstm", help="the model config in configs/, e.g. lstm_v2 for the SMARD data")
    parser.add_argument("--seed", type=int, help="overrides the seed of the config, e.g. to measure the run-to-run noise")
    parser.add_argument("--epochs", type=int, help="overrides the number of epochs of the config, e.g. for a quick test")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="overrides any config value, e.g. --set lr=1e-3 --set net_params.zero_init_output=true")
    parser.add_argument("--groups", nargs="+", default=[], metavar="GROUP",
                        help="adds feature groups of the selection config, e.g. --groups calendar residual_load")
    parser.add_argument("--selection", default="feature_selection",
                        help="the config of the feature groups, e.g. feature_selection_v2 for model v2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on device: {device}")

    config = load_config(args.config)
    data_config = load_data_config(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    apply_overrides(config, args.set)
    config["features"], config["targets"] = add_feature_groups(config["features"], config["targets"],
                                                               load_config(args.selection)["groups"], args.groups)

    # every run gets its own weights, config and logs, so runs never overwrite each other
    run = f"{config['model_name']}_{datetime.now():%Y%m%d-%H%M%S}_seed{config['seed']}"
    print(f"run: {run}")
    writer = SummaryWriter(str(PROJECT_ROOT / "logs" / run))

    # the test split is not touched during development
    train_ds, valid_ds, _ = load_datasets(config, data_config)

    save_model_path = MODEL_DIR / f"{run}.pth"
    with open(save_model_path.with_suffix(".json"), "w") as f:
        json.dump({"config": config, "feature_dims": list(get_feature_dims(train_ds))}, f, indent=2)

    model, best_epoch = train(config, train_ds, valid_ds, device, save_model_path, writer)
    writer.close()

    # without dropout, so that train and val errors are comparable
    metrics = {"train": evaluate_model(model, train_ds, device), "val": evaluate_model(model, valid_ds, device)}
    metrics["val"]["future_gain_eur"] = future_shuffle_gain(model_forecast(model, device), *stack_windows(valid_ds), valid_ds)
    log_experiment(run, config, best_epoch, metrics)
    for split, m in metrics.items():
        print(f"{split:5}: MAE {m['mae_eur']:.1f} EUR/MWh, RMSE {m['rmse_eur']:.1f} EUR/MWh, MSE (scaled) {m['mse_z']:.3f}")
    print(f"the val MAE grows by {metrics['val']['future_gain_eur']:.1f} EUR/MWh when the inputs of day D are shuffled")
    print(f"best epoch {best_epoch}, results added to {EXPERIMENT_LOG.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
