# Roadmap

Working plan from the first trained LSTM to a tested day-ahead price forecast. Status: 2026-10-29.

## Where things stand

Day-ahead task: at 12:00 on D-1 the 24 prices of D are forecast, with the measured values of day D (weather, wind, solar, residual load) as perfect forecasts. Validation split Q4 2024 (91 days), errors in €/MWh:

| Model | MAE | RMSE |
|---|---|---|
| Mean of the training prices | 41.4 | 73.4 |
| Persistence (price at 23:00 on D-1) | 34.6 | 66.3 |
| Day D-1 | 37.9 | 62.9 |
| Day D-7 | 47.1 | 86.6 |
| Ridge, base features | 24.4 | 51.2 |
| Ridge, selected features | 20.4 | 48.1 |
| LSTM reference (step 3), 3 seeds | 24.39 ± 0.15 | 53.9 |
| LSTM with the training of step 4, base features | 23.55 ± 0.22 | 52.5 |
| LSTM, current config (step 5), 3 seeds | 18.68 ± 0.43 | 48.6 |
| LSTM, mean forecast of 18 runs (step 6) | 18.0 | 48.0 |
| Mean of the 18 LSTM runs and Ridge, 50/50 (step 6) | 17.5 | 47.1 |
| **Final model: mean of the 3 seeds of the current config and Ridge, 50/50 (step 6)** | **17.6** | **47.3** |

- All numbers are an upper bound for real forecasts, because the values of day D are measurements.
- The validation quarter is volatile: mean price 102.6, std 70.6, max 936 €/MWh (Dunkelflaute on 11 and 12 December 2024). These two days make up more than half of the squared error.
- The LSTM now uses the inputs of day D: the MAE grows by 15.3 €/MWh when they are shuffled, before step 5 by 6.7. Ridge with the base features: 10.5.
- The gap between training and validation grew (train MAE 11.2, val 18.7), a target for step 6.

## Rules for every experiment

1. Decide on the validation split only. The test split (Q1 2025) is used once, in step 7.
2. Change one thing at a time and train it with 3 seeds, all on Colab: CPU and GPU runs differ by about 0.2 €/MWh.
3. A change counts if the mean MAE improves by at least 0.5 €/MWh and all 3 seeds are better than the mean it is compared with. The seed noise is 0.15 to 0.43 €/MWh per run, so the difference of two means of 3 seeds varies by about ±0.2. This protects against seed noise, not against effects that only hold in Q4 2024; the test split shows those.
4. Whenever the model gets new information, compare it with Ridge on the same information (`python -m spotforecast.evaluate --run <run>` does that).
5. Every run is logged in `logs/experiments.csv`; `python scripts/compare_runs.py` shows mean ± std per configuration.

## Steps

### 0. Decide the task ✔

- [x] Task definition: **the real day-ahead task**. At 12:00 on D-1, the day of the auction, the 24 hourly prices of D are forecast. It matches the auction and the literature. Cost: one sample per day, 628 training days.
- [x] Forecast data: no real forecasts. The measured values of day D are used as perfect day-ahead forecasts.

### 1. Adapt the pipeline ✔

- [x] One window per day: the targets are the hours 00 to 23 of day D, the history ends at midnight before D. The days with a clock change are skipped.
- [x] Only what is known at 12:00 on D-1 goes in: all prices of D-1 (auction on D-2), values with a day-ahead forecast (weather, wind, solar, load) as perfect forecasts, everything else shifted by 12 hours (`_lag12`), daily prices by 24 hours (`_lag24`).
- [x] Only Spot is forecast, the other `targets` are inputs of day D. Baselines "day D-1" and "day D-7", errors per block of hours of day D.

### 2. Sanity checks ✔

`notebooks/02_sanity_checks.ipynb`: no error in the data pipeline. Findings for step 4: the training was not finished (best epoch = last), the untrained model started at 3.4 times the persistence loss, the forecasts were too smooth and used the inputs of day D too little.

### 3. New reference ✔

- [x] Current config of that time (base features, lr 1e-4, 70 epochs, MSE) with seeds 1 to 3 on Colab: **24.39 ± 0.15** €/MWh, best epochs 53 to 68 of 70.

### 4. Training fixes ✔

3 seeds each on Colab, base features:

| Configuration | Val MAE | vs 4a | Decision |
|---|---|---|---|
| Reference (step 3) | 24.39 ± 0.15 | | |
| **4a** lr 1e-3, 300 epochs, early stopping after 20 | 23.77 ± 0.28 | | ✅ −0.62, best epochs now 19 to 30 |
| 4a + **4b** zero-initialized output layer | 23.63 ± 0.16 | −0.14 | noise |
| 4a + **4c** Huber loss | 23.35 ± 0.19 | −0.42 | noise (just below 0.5) |
| 4a + 4b + 4c | 23.55 ± 0.22 | −0.22 | noise |
| 4a + **4d** bidirectional | 23.78 ± 0.21 | +0.01 | ❌ no effect, twice the compute |

- [x] 4a solved the training problem. Huber and the zero initialization have no measurable effect (Huber alone −0.42, together with the zero initialization only −0.08), but they do not hurt. They stay in the config, because all feature results were measured with them; step 6 decides on the loss again.
- [x] Bidirectional is dropped.

### 5. Features ✔

- [x] New columns: calendar (Saturday, Sunday, share of the population with a public holiday; Christmas Eve and New Year's Eve count as holidays), residual load (load − wind − solar), shifted columns (`_lag12`, `_lag24`). Generation, load and trade are hourly means in MW.
- [x] Screening of all 64 combinations of six groups (`configs/feature_selection.toml`, `scripts/select_features.py`) with Ridge and LightGBM in four rolling folds (validation Q1, Q2, Q3, Q4 2024). A group is kept if it lowers the mean MAE by at least 0.1 €/MWh and is better in at least 3 folds.
- [x] Confirmed with the LSTM (training of step 4, 3 seeds on Colab):

| Features | LSTM val MAE | vs no groups | Ridge, same information | `future_gain` |
|---|---|---|---|---|
| no groups | 23.55 ± 0.22 | | 24.4 | 6.7 |
| calendar + residual_load | 20.64 ± 0.37 | −2.9 | 21.1 | 12.1 |
| **calendar + residual_load + renewables + fuel** | **18.68 ± 0.43** | **−4.9** | **20.4** | **15.3** |

- [x] Candidate 2 is the new config in `configs/lstm.toml`. The LSTM beats Ridge on the same information by 1.7 €/MWh.

Screening results, mean MAE over the four folds in €/MWh (fold 4 = the validation split):

| Group | What it adds | Ridge | LightGBM | Verdict |
|---|---|---|---|---|
| residual_load | load − wind − solar, history and day D | −0.74, better in 72 % of the folds | −0.57, 59 %; selected first | kept |
| calendar | Saturday, Sunday, holiday share, history and day D | −0.50, 92 % | −0.21, 58 % | kept |
| fuel | gas, coal, CO₂ of the day before | −0.83, 94 % | −0.10, 54 % | kept |
| renewables | wind and solar generation of day D | −0.45, 84 % | −0.37, 66 % | kept |
| load | load of day D | −0.31, 62 % | −0.45, 61 % | dropped: residual load + renewables already contain it |
| conventional | coal and gas generation, trade, shifted by 12 h | −0.30, 83 % | +0.83, 39 % | dropped |

The numbers are the average effect of adding the group to any combination without it.

**Needs data that is not in the project:**

- [ ] More history. 628 training days are few for an LSTM. SMARD.de (Bundesnetzagentur) offers the same prices, generation and load hourly from 2015 on. The weather columns would have to come from the same source as `wind_mean`/`solar_mean`, otherwise the wind and solar generation replaces them. 2021–2022 was the gas crisis, a different price regime that only helps together with the fuel prices.
- [ ] Neighbouring countries (prices and load of FR, NL, AT, CH), via the ENTSO-E Transparency Platform (free API token).
- [ ] Real day-ahead forecasts of wind, solar and load (ENTSO-E) instead of the measured values of day D. This turns the upper bound into a realistic result and shows what perfect forecasts are worth.
- [ ] Power plant outages (REMIT unavailability, ENTSO-E or EEX Transparency). Together with the neighbouring countries they are the most likely help for the spikes, which no model foresees so far.

### 6. Tuning and model comparison

- [x] Code: `python -m spotforecast.tune` searches the hyperparameters below with Optuna (TPE). Trials that are worse than the median of the earlier ones at the same epoch are stopped after epoch 15. The first trial is the current config. Trials are stored in `logs/optuna_step6.log`, so the search continues after an interruption; they save no weights and are not added to `experiments.csv`.
- [x] Search on Colab, 50 trials (45 complete, 5 stopped early), one seed per trial: the configurations differ by 1 to 3 €/MWh, the seed noise is about ±0.4, so one seed is enough to find the good region. Not extended to 100, because the top is flat.
- Result: the 10 best trials lie within 0.15 €/MWh (18.16 to 18.31); the current config (trial 0) is 6th with 18.23. The top shares batch size 16, hidden size 256, 2 layers, dropout 0.2 to 0.45, lr 4e-4 to 1.2e-3 and a small weight decay. By Optuna's importance, weight decay (0.55) and lr (0.22) explain most of the differences, mainly because large weight decays and extreme learning rates are bad. The loss has an importance of 0: the MSE trials were bad because of their other settings, so Huber is not proven better, it just does not hurt.
- All trials ran with seed 1, which is a lucky seed for the current config: 18.23 against its 3-seed mean of 18.68. The 3-seed means of the finalists will likely be about as high, so probably none of them passes rule 3.
- [x] The 5 best trials with 3 seeds each. Near the top the trials differ by less than the seed noise, and the best of many noisy trials looks better than it is, so only the 3-seed mean counts:

| Finalist | Val MAE, 3 seeds | vs current config (18.68) | Trial value (seed 1) |
|---|---|---|---|
| Trial 28: batch 16, history 168, dropout 0.20, 2 layers | 18.38 ± 0.54 | −0.30 | 18.22 |
| Trial 18: batch 16, dropout 0.39, 3 layers | 18.41 ± 0.60 | −0.27 | 18.17 |
| Trial 44: batch 16, dropout 0.38, 2 layers | 18.48 ± 0.43 | −0.20 | 18.17 |
| Trial 22 (best trial of the search) | 18.48 ± 0.35 | −0.20 | 18.16 |
| Trial 25: batch 16, dropout 0.26, 2 layers | 18.60 ± 0.33 | −0.08 | 18.23 |

- [x] Decision: **the current config stays.** No finalist is 0.5 €/MWh better (rule 3). With 3 seeds every trial is 0.2 to 0.4 worse than with seed 1 alone, and the best trial of the search falls to 4th place. Further tuning of the hyperparameters brings nothing measurable.
- [ ] LightGBM tuned as a second model; Ridge with the selected features is the bar (20.4).
- [x] Code: `python -m spotforecast.evaluate --ensemble` averages the forecasts of the runs in `models/` that have the features of `configs/lstm.toml`. It shows three variants: the 3 seeds of the current config, all 18 runs with these features (the current config and the 5 finalists), and the mean of all of them with Ridge, with a fixed 50/50 weight so that nothing is fitted on the validation split.
- [x] Run on Colab. The ensemble is a fixed combination of existing runs and has no seed noise; it is adopted if it is at least 0.5 better than the single runs (18.68):

| Ensemble | Val MAE | vs single runs | RMSE | MAE 00–06 | 06–12 | 12–18 | 18–24 |
|---|---|---|---|---|---|---|---|
| Ridge alone | 20.4 | | 48.1 | 12.1 | 20.4 | 26.7 | 22.2 |
| LSTM, the 3 seeds of the current config | 18.4 | −0.28 | 48.4 | 8.1 | 19.6 | 25.5 | 20.5 |
| LSTM, all 18 runs with these features | 18.0 | −0.68 | 48.0 | 8.2 | 18.9 | 25.0 | 20.0 |
| **All 18 runs + Ridge, 50/50** | **17.5** | **−1.18** | **47.1** | 8.2 | 17.8 | 24.5 | 19.4 |

- LSTM and Ridge make different errors: their mean keeps the LSTM's level at night and beats both during the day. MSE (scaled) 1.45, the best so far.
- [x] "3 seeds + Ridge": **17.6** (RMSE 47.3, MSE scaled 1.46), only 0.1 behind "18 runs + Ridge" (17.5), which contains the finalists chosen on the validation split and is slightly flattered.
- [x] Decision, **the final model is frozen: the mean of the forecasts of the LSTM of `configs/lstm.toml` with the seeds 1, 2 and 3 and of Ridge, 50/50.** It is the simpler of the two, nothing in it is chosen on the validation split, and step 7 retrains 3 runs instead of 18.

| Hyperparameter | Search range | Current config |
|---|---|---|
| `lr` | 1e-4 to 3e-3, log scale | 1e-3 |
| `weight_decay` | 1e-5 to 1e-1, log scale | 1e-4 |
| `batch_size` | 16, 32, 64 | 64 |
| `forecast_history` | 168, 192, 336 hours | 192 |
| `loss` | mse, huber | huber |
| `net_params.hidden_dim` | 64, 128, 256, 512 | 256 |
| `net_params.num_layers` | 1 to 3 | 3 |
| `net_params.dropout_p` | 0 to 0.5 | 0.5 |

One seed per trial is noisy, and choosing the best of 50 trials on the same validation split flatters it. The 3-seed confirmation and finally the test split correct for that.

Done when the best model is chosen. It does not have to be the LSTM.

### 7. Final evaluation and write-up

- [x] Freeze the final model and the procedure before the test split is touched (step 6).
- [x] Code: `python -m spotforecast.final` retrains the final model on train + validation (the validation quarter holds the most recent prices and the December spikes) and evaluates it once on the test split. Without a validation split there is no early stopping: every LSTM trains for the median best epoch of the validation runs of the config and keeps its last epoch; the learning rate schedule follows the training loss. Ridge is refitted on the same data. The report goes to `reports/test_results.md`: the baselines, Ridge, the single seeds, their mean and the final model, with the error distribution and Diebold-Mariano tests against Ridge, the LSTM mean and day D-1. The final models are saved in `models/final/`, apart from the validation runs. When the report exists, the command only shows it, a second evaluation needs `--force`.
- [ ] Dry run on Colab — **next step**: `--dry-run` runs the same procedure on train → validation and saves nothing. It checks that everything runs and that the fixed number of epochs keeps the validation result of the early-stopped runs (17.6). If it is clearly worse (by more than about 0.5), the procedure is changed before the test split is touched.
- [ ] Evaluate once on the test split (Q1 2025). Expectation: Q1 2025 was calmer than Q4 2024, so a MAE of about 14 to 15 €/MWh and a scaled MSE of about 0.5 to 0.6.
- [ ] Report: comparison with the baselines, error per hour of the day and per price level, the distribution of the errors (median, share of hours within 10 and 20 €/MWh), the spike days separately, a Diebold-Mariano test against Ridge. State clearly that the inputs of day D are perfect forecasts.
- [ ] Optional: probabilistic forecasts (e.g. the 10 %, 50 % and 90 % quantiles), which show when the model is unsure, e.g. before a Dunkelflaute.

How to read the MAE (Ridge with the selected features on the validation split, MAE 20.4; the LSTM is about 10 % better): the median error is 11.8 €/MWh, 44 % of the hours are within 10 €/MWh and 73 % within 20. Without the spike days 11 and 12 December the MAE is 16.9, on these two days 173. By price level: 11 to 14 €/MWh for prices between 50 and 150, 77 above 150, 21 below 50.
- [ ] README, LICENSE, tests that guard the split against leakage.

## Colab runs

Push the code first; models/ and logs/ stay on Drive as before. A training run takes about one minute on the Colab GPU. In a Colab cell, `!` commands take Python variables in `{}`; no comments after `%cd`.

Step 6, the search (about 20 to 30 minutes: with early stopping a trial takes 15 to 35 seconds on the Colab GPU). If Colab disconnects, run the same cell again, the search continues where it stopped. `--trials 100` would extend the same search to 100 trials:

```python
%cd /content/day-ahead-price-forcast
!git pull
!pip install -q -e .
!python -m spotforecast.tune --trials 50
```

Then, in a new cell, the best trials and the options of the 5 best:

```python
!git pull
!python -m spotforecast.tune --summary
```

Copy the printed `candidates = [...]` list into a cell and train every finalist with 3 seeds (15 runs, about 10 minutes):

```python
candidates = [...]   # replace [...] by the list from the summary
for best in candidates:
    for s in [1, 2, 3]:
        !python -m spotforecast.train --seed {s} {best}
!python scripts/compare_runs.py
```

Step 6, the ensemble of the trained runs (a few minutes, nothing is trained):

```python
%cd /content/day-ahead-price-forcast
!git pull
!python -m spotforecast.evaluate --ensemble
```

Step 7, first the dry run on train → validation (about 2 minutes, nothing is saved, the test split is not touched):

```python
%cd /content/day-ahead-price-forcast
!git pull
!python -m spotforecast.final --dry-run
```

Then, only if the dry run is fine and only once, the evaluation on the test split:

```python
!python -m spotforecast.final
```

## Expected accuracy

Validation MAE in €/MWh, with the inputs of day D as perfect forecasts:

| | MAE | vs day D-1 (37.9) |
|---|---|---|
| Measured: Ridge, base features | 24.4 | −36 % |
| Measured: Ridge, selected features | 20.4 | −46 % |
| Measured: LSTM, current config | 18.7 | −51 % |
| Measured: 18 LSTM runs + Ridge (step 6) | 17.5 | −54 % |
| Measured: final model, 3 seeds + Ridge (step 6) | 17.6 | −54 % |

- Tuning brought nothing measurable, the ensemble with Ridge brought 1.2 €/MWh. The estimate of 17–18.5 after step 6 held.
- The scaled validation MSE of the ensemble is 1.45 (RMSE 47.1). It will hardly fall below 1 unless a model foresees the December spikes.
- The test quarter Q1 2025 was calmer. Expect a test MAE about 15 to 20 % lower and a test MSE of roughly 0.5 to 0.6.
- With real forecasts instead of measured values, the error would be several €/MWh higher.
