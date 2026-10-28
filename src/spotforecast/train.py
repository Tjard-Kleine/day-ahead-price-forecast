import torch
import secrets
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, ReduceLROnPlateau
from typing import Callable, Optional
from pathlib import Path

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import load_datasets, get_feature_dims, default_batch_transform, split_history_future_batch_transform
from spotforecast.models.lstm import build_model
from spotforecast.utils import seed_torch, load_model


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, writer, device, transform_batch_fn:Callable=default_batch_transform,
        save_model_path: Optional[Path]=None, scheduler: Optional[LRScheduler]=None):
    best_val_loss = torch.inf
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on device: {device}")

    # Draw random seeds using Python's cryptographic library and cut them down to size
    _global_seed = secrets.randbits(128) % (2**64-1)
    _local_seed = secrets.randbits(128) % (2**64-1)

    # If you want to reproduce a previous experiment (and only then) set the seeds to that previously recorded value
    # and set `reproducible=True`. N.B.: using `42` is never a good idea!
    seed_torch(_global_seed, reproducible=False)
    _rng_cpu = torch.Generator()
    _rng_cpu.manual_seed(_local_seed)

    config = load_config("lstm")
    data_config = load_config("data")

    writer = SummaryWriter(str(PROJECT_ROOT / "logs" / config["model_name"]))
    writer.add_scalar("hparams/global_seed", _global_seed)
    writer.add_scalar("hparams/local_seed", _local_seed)

    train_ds, valid_ds = load_datasets(config, data_config)

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
    save_model_path = PROJECT_ROOT / "models" / f"{config['model_name']}.pth"

    fit(model, train_loader, val_loader, optimizer, criterion, config['epochs'], writer, device, transform_batch_fn=batch_tsfm_fn,
        save_model_path=save_model_path, scheduler=scheduler)

    print(f"Loading best model...\n")
    load_model(model_path=save_model_path, model=model, optimizer=optimizer, device=device)

    writer.close()


if __name__ == '__main__':
    main()
