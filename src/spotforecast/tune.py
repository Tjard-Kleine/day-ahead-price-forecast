"""Hyperparameter search for the LSTM with Optuna, step 6 of the roadmap.

Every trial trains the model of configs/lstm.toml with sampled hyperparameters and returns its validation MAE in
EUR/MWh. Trials whose MAE after some epochs is worse than the median of the earlier trials at that epoch are stopped
early. The trials are stored in logs/optuna_<study>.log, so a search that was interrupted (e.g. by a Colab disconnect)
continues when it is started again. Trials save no weights and are not added to logs/experiments.csv: one seed per
trial is noisy, so the best configuration is trained with 3 seeds afterwards like every other change.

    python -m spotforecast.tune --trials 50     # about 1 to 1.5 h on a Colab GPU
    python -m spotforecast.tune --summary       # the best trials and the command to train the best one
"""
import argparse
import contextlib
import copy
import functools
import io
import tempfile
from pathlib import Path

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import build_datasets, load_preprocessed_data
from spotforecast.evaluate import evaluate_model
from spotforecast.train import apply_overrides, set_config_value, train

# the search space: config key -> how optuna samples it
SEARCH_SPACE = {
    "lr": lambda trial: trial.suggest_float("lr", 1e-4, 3e-3, log=True),
    "weight_decay": lambda trial: trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
    "batch_size": lambda trial: trial.suggest_categorical("batch_size", [16, 32, 64]),
    "forecast_history": lambda trial: trial.suggest_categorical("forecast_history", [168, 192, 336]),
    "loss": lambda trial: trial.suggest_categorical("loss", ["mse", "huber"]),
    "net_params.hidden_dim": lambda trial: trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
    "net_params.num_layers": lambda trial: trial.suggest_int("num_layers", 1, 3),
    "net_params.dropout_p": lambda trial: trial.suggest_float("dropout_p", 0.0, 0.5),
}


def storage(study: str) -> JournalStorage:
    # a journal file instead of a database, it works on the Google Drive folder that Colab mounts. The lock is a plain
    # file, the default lock needs symbolic links, which Windows does not allow without admin rights
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    path = str(PROJECT_ROOT / "logs" / f"optuna_{study}.log")
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def config_value(config: dict, key: str):
    """A config value, nested keys are joined by dots."""
    for part in key.split("."):
        config = config[part]
    return config


def sampled_config(trial: optuna.Trial, config: dict) -> dict:
    """A copy of the config with the hyperparameters of the search space sampled for the trial."""
    config = copy.deepcopy(config)
    for key, suggest in SEARCH_SPACE.items():
        set_config_value(config, key, suggest(trial))
    return config


def train_command(params: dict) -> str:
    """The command that trains the config with the parameters of a trial."""
    keys = {key.split(".")[-1]: key for key in SEARCH_SPACE}
    return "python -m spotforecast.train " + " ".join(f"--set {keys[name]}={value}" for name, value in params.items())


def search(config: dict, data_config: dict, study_name: str, n_trials: int, device: torch.device) -> None:
    preprocess_kwargs = dict(data_config["preprocessing"])
    valid_cutoff = preprocess_kwargs.pop("valid_cutoff_datetime")
    test_cutoff = preprocess_kwargs.pop("test_cutoff_datetime")
    print("Loading and preprocessing the data...")
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        df = load_preprocessed_data(PROJECT_ROOT / data_config["path"],
                                    list(dict.fromkeys(config["features"] + config["targets"])), preprocess_kwargs)

    @functools.cache
    def datasets(forecast_history: int):
        # the history length is searched as well, the windows are cut once per length
        train_ds, valid_ds, _ = build_datasets(df, config["features"], config["targets"], forecast_history,
                                               config["forecast_horizon"], preprocess_kwargs.get("datetime_col", "Date"),
                                               valid_cutoff, test_cutoff)
        return train_ds, valid_ds

    def objective(trial: optuna.Trial) -> float:
        trial_config = sampled_config(trial, config)
        train_ds, valid_ds = datasets(trial_config["forecast_history"])

        def report(epoch: int, model: torch.nn.Module):
            # the MAE and not the loss, because MSE and Huber losses of different trials cannot be compared
            trial.report(evaluate_model(model, valid_ds, device)["mae_eur"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            model, best_epoch = train(trial_config, train_ds, valid_ds, device, Path(tmp) / "trial.pth",
                                      epoch_callback=report)
        val = evaluate_model(model, valid_ds, device)
        trial.set_user_attr("val_rmse_eur", val["rmse_eur"])
        trial.set_user_attr("train_mae_eur", evaluate_model(model, train_ds, device)["mae_eur"])
        trial.set_user_attr("best_epoch", best_epoch)
        return val["mae_eur"]

    study = optuna.create_study(study_name=study_name, storage=storage(study_name), load_if_exists=True,
                                direction="minimize",
                                # no stop before 8 complete trials and in the first 15 epochs of a trial
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15))
    # seeded by the number of earlier trials, a resumed search would otherwise repeat the random trials it started with
    study.sampler = optuna.samplers.TPESampler(seed=config["seed"] + len(study.trials))
    if len(study.trials) == 0:
        # the current config as the first trial, so the search starts from a known result
        study.enqueue_trial({key.split(".")[-1]: config_value(config, key) for key in SEARCH_SPACE})
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(remaining, 0))


def summary(study_name: str) -> None:
    study = optuna.load_study(study_name=study_name, storage=storage(study_name))
    trials = study.trials_dataframe()
    finished = trials[trials["state"] == "COMPLETE"].sort_values("value")
    columns = ["number", "value"] + [c for c in trials.columns if c.startswith(("params_", "user_attrs_"))]
    print(f"{len(finished)} complete trials, {(trials['state'] == 'PRUNED').sum()} stopped early; "
          f"value = validation MAE in EUR/MWh\n")
    print(finished[columns].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
          .round(4).to_string(index=False))
    if len(finished) > 1:
        print("\nImportance of the hyperparameters for the MAE:")
        for name, importance in optuna.importance.get_param_importances(study).items():
            print(f"  {name:17s} {importance:.2f}")
    print("\nTrain the best trial with 3 seeds:")
    print(f"  {train_command(study.best_trial.params)} --seed <1, 2, 3>")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for the LSTM of configs/lstm.toml with Optuna.")
    parser.add_argument("--trials", type=int, default=50,
                        help="number of finished trials of the study, including earlier and stopped ones")
    parser.add_argument("--study", default="step6", help="name of the study, its trials are in logs/optuna_<study>.log")
    parser.add_argument("--summary", action="store_true", help="only show the best trials of the study")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="overrides config values that are not searched, e.g. --set epochs=5 for a quick test")
    args = parser.parse_args()

    if args.summary:
        summary(args.study)
        return
    config = load_config("lstm")
    apply_overrides(config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"searching on device: {device}")
    search(config, load_config("data"), args.study, args.trials, device)
    summary(args.study)


if __name__ == '__main__':
    main()
