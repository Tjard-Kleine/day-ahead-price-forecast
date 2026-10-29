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
| **LSTM, current config (step 5), 3 seeds** | **18.68 ± 0.43** | **48.6** |

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

### 6. Tuning and model comparison

- [x] Code: `python -m spotforecast.tune` searches the hyperparameters below with Optuna (TPE). Trials that are worse than the median of the earlier ones at the same epoch are stopped after epoch 15. The first trial is the current config. Trials are stored in `logs/optuna_step6.log`, so the search continues after an interruption; they save no weights and are not added to `experiments.csv`.
- [ ] Run the search with 50 trials on Colab — **next step**.
- [ ] Train the best trial with 3 seeds and compare it with 18.68 ± 0.43 (rule 3).
- [ ] LightGBM tuned as a second model; Ridge with the selected features is the bar (20.4).
- [ ] Optional: an ensemble, e.g. the mean of LSTM and LightGBM or Ridge.

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

- [ ] Retrain the best model on train + validation and evaluate it once on the test split.
- [ ] Report: comparison with the baselines, error per hour of the day, behaviour on price spikes. State clearly that the inputs of day D are perfect forecasts.
- [ ] README, LICENSE, tests that guard the split against leakage.

## Colab runs

Push the code first; models/ and logs/ stay on Drive as before. A training run takes about one minute on the Colab GPU. In a Colab cell, `!` commands take Python variables in `{}`; no comments after `%cd`.

Step 6, the search (about 1 to 1.5 hours). If Colab disconnects, run the same cell again, the search continues where it stopped:

```python
%cd /content/day-ahead-price-forcast
!git pull
!pip install -q -e .
!python -m spotforecast.tune --trials 50
```

Then, in a new cell, the best trials and the command for the best one:

```python
!python -m spotforecast.tune --summary
```

Copy the `--set ...` part of the printed command into `best` and train it with 3 seeds:

```python
best = "--set lr=... --set ..."
for s in [1, 2, 3]:
    !python -m spotforecast.train --seed {s} {best}
!python scripts/compare_runs.py
```

## Expected accuracy

Validation MAE in €/MWh, with the inputs of day D as perfect forecasts:

| | MAE | vs day D-1 (37.9) |
|---|---|---|
| Measured: Ridge, base features | 24.4 | −36 % |
| Measured: Ridge, selected features | 20.4 | −46 % |
| Measured: LSTM, current config | 18.7 | −51 % |
| Estimate after tuning (step 6), about ±1 | 17–18.5 | −51 to −55 % |

- The scaled validation MSE of the current config is about 1.5 (RMSE 48.6). It will hardly fall below 1 unless a model foresees the December spikes.
- The test quarter Q1 2025 was calmer. Expect a test MAE about 15 to 20 % lower and a test MSE of roughly 0.5 to 0.6.
- With real forecasts instead of measured values, the error would be several €/MWh higher.
