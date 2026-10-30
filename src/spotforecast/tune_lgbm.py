"""Hyperparameter search for LightGBM with Optuna, steps V6 (model v2) and P4d (phase 3) of the roadmap.

Every trial fits LightGBM of a config (configs/lgbm_v2.toml, configs/lgbm_v3.toml) with sampled hyperparameters on the
training split and returns its MAE on the validation year in EUR/MWh, the procedure of
python -m spotforecast.final_v2 --dry-run. The trials are stored in logs/optuna_<study>.log, so an interrupted search
continues when it is started again. One seed per trial, the seeds of LightGBM differ by only about 0.05 EUR/MWh;
afterwards the best trials are confirmed with all seeds of the config, as the mean forecast of the final model.

    python -m spotforecast.tune_lgbm --trials 60    # about 20 to 40 min on a laptop CPU
    python -m spotforecast.tune_lgbm --config lgbm_v3 --study lgbm_v3 --trials 60
    python -m spotforecast.tune_lgbm --summary      # the best trials
    python -m spotforecast.tune_lgbm --confirm      # the best trials and the current config with all seeds
"""
import argparse
import contextlib
import io
import time

import numpy as np
import optuna
import pandas as pd

from spotforecast.config import load_config, load_data_config
from spotforecast.data.dataset import load_datasets
from spotforecast.evaluate import compute_metrics
from spotforecast.models.baselines import LGBM_PARAMS, fit_lgbm, predict_lgbm, stack_windows
from spotforecast.tune import storage


def sample_params(trial: optuna.Trial) -> dict:
    """The search space, the names are those of LGBM_PARAMS."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        # the MAE is the target metric, the L1 and Huber losses weigh the price spikes less than the squared error
        "objective": trial.suggest_categorical("objective", ["regression", "huber", "l1"]),
    }


def current_params(config: dict) -> dict:
    """The searched hyperparameters of the config, a reg_lambda of 0 as the lower end of the search range."""
    params = {**LGBM_PARAMS, **config.get("params", {})}
    params["reg_lambda"] = max(params["reg_lambda"], 1e-3)
    return {k: v for k, v in params.items() if k != "subsample_freq"}


def load_windows(config: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
        train_ds, valid_ds, _ = load_datasets(config, load_data_config(config))
    (X_train, Y_train), (X_val, Y_val) = stack_windows(train_ds), stack_windows(valid_ds)
    return {"X_train": X_train, "Y_train": Y_train, "X_val": X_val, "Y_val": Y_val, "valid_ds": valid_ds}


def validate(params: dict, seed: int, w: dict) -> np.ndarray:
    """The scaled forecasts of the validation year of LightGBM with the params, fitted on the training split."""
    return predict_lgbm(fit_lgbm(w["X_train"], w["Y_train"], seed, **params), w["X_val"], w["Y_val"])


def search(config: dict, w: dict, study_name: str, n_trials: int) -> None:
    y = w["Y_val"][:, :, 0]

    def objective(trial: optuna.Trial) -> float:
        start = time.time()
        params = sample_params(trial)
        model = fit_lgbm(w["X_train"], w["Y_train"], config["seeds"][0], **params)
        val = compute_metrics(predict_lgbm(model, w["X_val"], w["Y_val"]), y, w["valid_ds"])
        train = compute_metrics(predict_lgbm(model, w["X_train"], w["Y_train"]), w["Y_train"][:, :, 0], w["valid_ds"])
        trial.set_user_attr("val_rmse_eur", val["rmse_eur"])
        trial.set_user_attr("train_mae_eur", train["mae_eur"])
        trial.set_user_attr("seconds", round(time.time() - start))
        return val["mae_eur"]

    study = optuna.create_study(study_name=study_name, storage=storage(study_name), load_if_exists=True,
                                direction="minimize")
    # seeded by the number of earlier trials, a resumed search would otherwise repeat the random trials it started with
    study.sampler = optuna.samplers.TPESampler(seed=config["seeds"][0] + len(study.trials))
    if len(study.trials) == 0:
        # the current config as the first trial, so the search starts from a known result
        study.enqueue_trial(current_params(config))
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(remaining, 0))


def best_trials(study_name: str, n_best: int) -> list[optuna.trial.FrozenTrial]:
    study = optuna.load_study(study_name=study_name, storage=storage(study_name))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return sorted(complete, key=lambda t: t.value)[:n_best]


def summary(study_name: str) -> None:
    study = optuna.load_study(study_name=study_name, storage=storage(study_name))
    trials = study.trials_dataframe()
    finished = trials[trials["state"] == "COMPLETE"].sort_values("value")
    columns = ["number", "value"] + [c for c in trials.columns if c.startswith(("params_", "user_attrs_"))]
    print(f"{len(finished)} complete trials; value = MAE on the validation year in EUR/MWh, trial 0 is the current "
          f"config\n")
    print(finished[columns].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
          .round(4).to_string(index=False))
    if len(finished) > 1:
        print("\nImportance of the hyperparameters for the MAE:")
        for name, importance in optuna.importance.get_param_importances(study).items():
            print(f"  {name:17s} {importance:.2f}")


def confirm(config: dict, w: dict, study_name: str, n_best: int=5) -> None:
    """The best trials and the current config with every seed of the config and as the mean forecast of the seeds, the
    final model. The decision rule: at least 0.5 EUR/MWh better than the current config, with every seed."""
    y = w["Y_val"][:, :, 0]
    candidates = {"current config": current_params(config)}
    candidates |= {f"trial {t.number}": t.params for t in best_trials(study_name, n_best)}
    rows = {}
    for name, params in candidates.items():
        preds = [validate(params, seed, w) for seed in config["seeds"]]
        rows[name] = {f"seed {s}": compute_metrics(p, y, w["valid_ds"])["mae_eur"] for s, p in zip(config["seeds"], preds)}
        mean = compute_metrics(np.mean(preds, axis=0), y, w["valid_ds"])
        rows[name] |= {"mean forecast": mean["mae_eur"], "mean forecast rmse": mean["rmse_eur"]}
        print(f"  {name} done", flush=True)
    table = pd.DataFrame(rows).T
    seed_columns = [f"seed {s}" for s in config["seeds"]]
    table["every seed 0.5 better"] = (table[seed_columns] <= table.loc["current config", seed_columns] - 0.5).all(axis=1)
    print(f"\nMAE on the validation year in EUR/MWh with the seeds {config['seeds']}:\n")
    print(table.round(2).to_string())
    for name, params in candidates.items():
        print(f"\n{name}: [params] for the LightGBM config")
        print("\n".join(f"{k} = {v!r}".replace("'", '"') for k, v in params.items()))


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for LightGBM of a config in configs/ with "
                                                 "Optuna.")
    parser.add_argument("--trials", type=int, default=60, help="number of finished trials of the study, including "
                                                               "earlier ones")
    parser.add_argument("--config", default="lgbm_v2", help="the LightGBM config in configs/")
    parser.add_argument("--study", default="lgbm_v2", help="name of the study, its trials are in logs/optuna_<study>.log")
    parser.add_argument("--summary", action="store_true", help="only show the best trials of the study")
    parser.add_argument("--confirm", action="store_true", help="only confirm the best trials with all seeds")
    args = parser.parse_args()

    if args.summary:
        summary(args.study)
        return
    config = load_config(args.config)
    print("Loading the data...", flush=True)
    w = load_windows(config)
    if not args.confirm:
        search(config, w, args.study, args.trials)
        summary(args.study)
    print("\nConfirming the best trials with all seeds...", flush=True)
    confirm(config, w, args.study)


if __name__ == '__main__':
    main()
