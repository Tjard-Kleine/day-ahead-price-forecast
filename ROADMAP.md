# Roadmap

Working plan from the first trained LSTM to a tested day-ahead price forecast. Status: 2026-10-29.

## Where things stand

Day-ahead task: at 12:00 on D-1 the 24 prices of D are forecast, with the measured weather of D as a perfect forecast. Validation split Q4 2024 (91 days), errors in €/MWh:

| Model | MAE | RMSE | MAE 00–06 | MAE 06–12 | MAE 12–18 | MAE 18–24 |
|---|---|---|---|---|---|---|
| Mean of the training prices | 41.4 | 73.4 | 26.8 | 43.6 | 51.0 | 44.3 |
| Persistence (price at 23:00 on D-1) | 34.6 | 66.3 | 11.1 | 37.2 | 47.0 | 43.2 |
| Day D-1 | 37.9 | 62.9 | 24.9 | 41.6 | 46.4 | 38.5 |
| Day D-7 | 47.1 | 86.6 | 35.5 | 49.6 | 57.8 | 45.5 |
| Ridge | 24.4 | 51.2 | 13.4 | 26.1 | 31.6 | 26.2 |
| LSTM, seed 1 (CPU) | 24.5 | 53.8 | 8.7 | 28.2 | 34.0 | 27.2 |
| Ridge with the selected features (step 5) | 20.4 | | | | | |

- `wind_mean` and `solar_mean` (and in step 5 the residual load and the generation of day D) are measurements used as perfect forecasts. All numbers are an upper bound for real forecasts.
- The validation quarter is volatile: mean price 102.6, std 70.6, max 936 €/MWh (Dunkelflaute on 11 and 12 December 2024).
- The LSTM uses the inputs of day D only half as much as Ridge: the MAE grows by 5.1 €/MWh when they are shuffled, for Ridge by 10.5 (`future_gain_eur`).

## Rules for every experiment

1. Decide on the validation split only. The test split (Q1 2025) is used once, in step 7.
2. Change one thing at a time and train it with 3 seeds.
3. A change counts only if the mean MAE improves by at least 1.5 €/MWh. Check this threshold against the seed noise measured in step 3.
4. Whenever the model gets new information, compare it with Ridge on the same information (`python -m spotforecast.evaluate --run <run>` does that).
5. Every run is logged in `logs/experiments.csv`; `python scripts/compare_runs.py` shows mean ± std per configuration.

## Steps

### 0. Decide the task ✔

- [x] Task definition: **the real day-ahead task**. At 12:00 on D-1, the day of the auction, the 24 hourly prices of D are forecast. It matches the auction and the literature. Cost: one sample per day, 628 training days.
- [x] Forecast data: no real forecasts. The measured values of day D are used as perfect day-ahead forecasts, so all results are an upper bound for what real forecasts would reach.

### 1. Adapt the pipeline ✔

- [x] One window per day: the targets are the hours 00 to 23 of day D, the history ends at midnight before D. The days with a clock change are skipped.
- [x] Only what is known at 12:00 on D-1 goes in: all prices of D-1 (auction on D-2), values with a day-ahead forecast (weather, wind, solar, load) as perfect forecasts, everything else shifted by 12 hours (`_lag12`), daily prices by 24 hours (`_lag24`).
- [x] Only Spot is forecast, the other `targets` are inputs of day D. Baselines "day D-1" and "day D-7", errors per block of hours of day D.

### 2. Sanity checks ✔

`notebooks/02_sanity_checks.ipynb`: no error in the data pipeline. Alignment, overfit test, single days and the bidirectional readout passed. Findings for step 4: the training is not finished (best epoch = last, only 10 steps per epoch), the untrained model starts at 3.4 times the persistence loss, the forecasts are too smooth and use the inputs of day D too little.

### 3. New reference — Colab

- [ ] Train the current config with seeds 1 to 3 and measure the seed noise. Seed 1 ran on the CPU (24.5), but GPU and CPU differ slightly, so all three run on Colab.

### 4. Training fixes — Colab

Fixes of the findings, not tuning; every option is a config value and can be set with `--set`. 3 seeds each, every step builds on the winner of the one before:

- [x] Code: `early_stopping_patience`, `loss = "huber"` with `huber_delta`, `net_params.zero_init_output`, a correct readout of both directions for `net_params.bidirectional`, `future_gain_eur` in every run.
- [ ] 4a. Training budget: `lr=1e-3`, `epochs=300`, `early_stopping_patience=20`. Goal: the best epoch lies clearly before the end.
- [ ] 4b. Start at persistence: `net_params.zero_init_output=true`.
- [ ] 4c. Huber loss: `loss=huber`. The metric is the MAE, and with MSE the few spike days dominate the gradients. Expect a better MAE and a worse RMSE.
- [ ] 4d. Bidirectional: `net_params.bidirectional=true`.

Learning rate details, batch size, dropout, hidden size, layers and history length are tuned in step 6.

### 5. Features

- [x] New columns: calendar (Saturday, Sunday, share of the population with a public holiday; Christmas Eve and New Year's Eve count as holidays), residual load (load − wind − solar), shifted columns (`_lag12`, `_lag24`). Generation, load and trade are now hourly means in MW, which leaves the model inputs unchanged.
- [x] Screening of all 64 combinations of six groups (`configs/feature_selection.toml`, `scripts/select_features.py`) with Ridge and LightGBM in four rolling folds (validation Q1, Q2, Q3, Q4 2024). A group is kept if it lowers the mean MAE by at least 0.1 €/MWh and is better in at least 3 folds.
- [ ] Confirm the two candidates with the LSTM (with the training of step 4) — Colab.
- [ ] More data, see below.

Screening results, mean MAE over the four folds in €/MWh (fold 4 = the validation split):

| Groups | Ridge mean | Ridge fold 4 | LightGBM mean | LightGBM fold 4 |
|---|---|---|---|---|
| none | 19.89 | 24.35 | 21.22 | 22.03 |
| calendar + residual_load **(candidate 1)** | 18.61 | 21.14 | 18.77 | 17.79 |
| calendar + residual_load + renewables + fuel **(candidate 2)** | 17.19 | 20.35 | 18.94 | 18.33 |

| Group | What it adds | Ridge | LightGBM | Verdict |
|---|---|---|---|---|
| residual_load | load − wind − solar, history and day D | −0.74, better in 72 % of the folds | −0.57, 59 %; selected first | keep |
| calendar | Saturday, Sunday, holiday share, history and day D | −0.50, 92 % | −0.21, 58 % | keep |
| fuel | gas, coal, CO₂ of the day before | −0.83, 94 % | −0.10, 54 % | candidate 2 |
| renewables | wind and solar generation of day D | −0.45, 84 % | −0.37, 66 % | candidate 2 |
| load | load of day D | −0.31, 62 % | −0.45, 61 % | drop: residual load + renewables already contain it |
| conventional | coal and gas generation, trade, shifted by 12 h | −0.30, 83 % | +0.83, 39 % | drop |

The numbers are the average effect of adding the group to any combination without it. Ridge selects calendar, fuel, residual_load and renewables, LightGBM only residual_load. LightGBM is untuned and weak in fold 3 (Q3 2024), so its votes count less.

**Needs data that is not in the project:**

- [ ] More history. 628 training days are few for an LSTM. SMARD.de (Bundesnetzagentur) offers the same prices, generation and load hourly from 2015 on. The weather columns would have to come from the same source as `wind_mean`/`solar_mean`, otherwise the wind and solar generation replaces them. 2021–2022 was the gas crisis, a different price regime that only helps together with the fuel prices.
- [ ] Neighbouring countries (prices and load of FR, NL, AT, CH), via the ENTSO-E Transparency Platform (free API token).

### 6. Tuning and model comparison

- [ ] Optuna on the validation split: learning rate, batch size, epochs, dropout, hidden size, layers, history length. One seed per trial is noisy, so confirm the winner with 3 seeds.
- [ ] LightGBM tuned as a second strong model, Ridge with the selected features as the bar (20.4 on the validation split).
- [ ] Optional: an ensemble, e.g. the mean of LSTM and LightGBM or Ridge.

Done when the best model is chosen. It does not have to be the LSTM.

### 7. Final evaluation and write-up

- [ ] Retrain the best model on train + validation and evaluate it once on the test split.
- [ ] Report: comparison with the baselines, error per hour of the day, behaviour on price spikes. State clearly that the inputs of day D are perfect forecasts.
- [ ] README, LICENSE, tests that guard the split against leakage.

## Colab runs

Push the code first; models/ and logs/ stay on Drive as before. Each run takes about 12 minutes on a laptop CPU, likely less on a GPU. In a Colab cell, `!` commands take Python variables in `{}`:

```python
# step 3: reference
for s in [1, 2, 3]:
    !python -m spotforecast.train --seed {s}

# step 4: each line builds on the winner of the one before, adjust `best` after each comparison
best = "--set lr=1e-3 --set epochs=300 --set early_stopping_patience=20"            # 4a
for extra in ["", "--set net_params.zero_init_output=true", "--set loss=huber", "--set net_params.bidirectional=true"]:
    for s in [1, 2, 3]:
        !python -m spotforecast.train --seed {s} {best} {extra}
!python scripts/compare_runs.py

# step 5: the two feature candidates, with the winning training settings of step 4 in `best`
for groups in ["calendar residual_load", "calendar residual_load renewables fuel"]:
    for s in [1, 2, 3]:
        !python -m spotforecast.train --seed {s} {best} --groups {groups}
!python scripts/compare_runs.py
!python -m spotforecast.evaluate     # the newest run against Ridge with the same information
```

The loop in step 4 trains 4a and then 4b, 4c and 4d each on top of 4a only. If 4b wins, add it to `best` before 4c and 4d are compared, or run them one after the other.

## Expected accuracy

Estimated validation MAE in €/MWh, about ±3, with the inputs of day D as perfect forecasts:

| | MAE | vs day D-1 (37.9) |
|---|---|---|
| Measured: Ridge, base features | 24.4 | −36 % |
| Measured: LSTM, seed 1, base features, not trained to the end | 24.5 | −35 % |
| Measured: Ridge with the selected features | 20.4 | −46 % |
| Estimate: well-trained LSTM or ensemble with the selected features | 18–21 | −45 to −53 % |

- The test quarter Q1 2025 was calmer. Expect a test MAE about 15 to 20 % lower.
- The RMSE stays at about twice the MAE because of price spikes.
- With real forecasts instead of measured values, the error would be several €/MWh higher.
