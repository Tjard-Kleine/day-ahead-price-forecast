# Roadmap

Working plan from the first trained LSTM to a tested day-ahead price forecast. Status: 2026-10-29.

## Where things stand

Day-ahead task: at 12:00 on D-1 the 24 prices of D are forecast, with the measured values of day D (weather, wind, solar, residual load) as perfect forecasts.

**Final result on the test split Q1 2025** (89 days, evaluated once, `reports/test_results.md`), errors in €/MWh:

| Forecast | MAE | RMSE | Median error | Hours within 10 / 20 €/MWh |
|---|---|---|---|---|
| Day D-1 | 29.0 | 45.3 | 17.7 | 33 % / 54 % |
| Ridge | 15.6 | 24.8 | 11.7 | 43 % / 74 % |
| LSTM, mean of 3 seeds | 15.5 | 24.1 | 10.7 | 48 % / 73 % |
| **Final model: LSTM mean + Ridge** | **13.9** | **22.9** | **9.5** | **52 % / 78 %** |

The final model halves the error of day D-1 (−52 %) and beats Ridge by 1.7 €/MWh (−11 %); both differences are significant (Diebold-Mariano, p ≤ 0.0001).

Validation split Q4 2024 (91 days), errors in €/MWh:

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
- [x] Dry run on Colab (`--dry-run`: the same procedure on train → validation, nothing saved). With a fixed 17 epochs the final model reaches **17.6** on the validation split, exactly the result of the early-stopped runs, so the procedure without early stopping works. It is also the more honest validation number, because no epoch was chosen on the validation split. Single seeds 18.7 / 18.9 / 19.2, their mean 18.5; median error of the final model 9.5 €/MWh, 52 % of the hours within 10 and 80 % within 20 €/MWh. Diebold-Mariano: the final model is more accurate than Ridge (p < 0.0001), than the LSTM mean alone (p = 0.011) and than day D-1 (p < 0.0001).
- [x] Evaluated once on the test split (Q1 2025), report in `reports/test_results.md` (committed, so the guard against a second evaluation works in every clone). The expectation written down before was a MAE of 14 to 15 €/MWh and a scaled MSE of 0.5 to 0.6:
  - **MAE 13.9 €/MWh**, slightly better than expected; RMSE 22.9, **scaled MSE 0.34**, much better than expected, because Q1 2025 had no spike like the December Dunkelflaute (RMSE only 1.6 times the MAE, on the validation split 2.7 times).
  - −52 % against day D-1 (29.0), −11 % against Ridge (15.6); Diebold-Mariano p ≤ 0.0001 for both.
  - **The LSTM alone is no better than Ridge on the test split** (15.5 against 15.6), while it was 2.0 better on the validation split. Only the combination wins clearly. The two models are complementary: the LSTM is much better at night (7.1 against 11.5), Ridge around midday (18.4 against 21.5).
  - The single seeds spread from 14.9 to 17.2 on the test split, much more than on the validation split. A single LSTM would have been a gamble; averaging the seeds and Ridge made the result robust.
  - The error distribution is the same as on the validation split (median 9.5, 52 % within 10 €/MWh); the lower MAE comes from fewer extreme hours.
- [x] Report: comparison with the baselines, error per block of hours, the distribution of the errors (median, share of hours within 10 and 20 €/MWh), Diebold-Mariano tests, the note that the inputs of day D are perfect forecasts (`reports/test_results.md`).
- [x] The model as a usable command: `python -m spotforecast.predict --date <day> [--plot]` forecasts the 24 prices of a day with the final model (the LSTMs in `models/final/`, Ridge refitted deterministically on train + validation; checked: it reproduces the Ridge of the test report exactly). It says whether the day was seen in training; only test days are honest forecasts.
- [x] Decisions: the final weights go into the repository (without optimizer state, 5.9 instead of 17.6 MB each, `scripts/export_final_models.py`, same forecasts); the dataset is not published, the README describes it; MIT license.
- [x] Final models copied from Drive to `models/final/`, optimizer state removed (5.9 MB each), checked: they reproduce the test report exactly (final model 13.9, LSTM mean 15.5, Ridge 15.6). Figures in `reports/figures/`: forecasts of four test days and the error per hour of the day.
- [x] Repository renamed to `Tjard-Kleine/day-ahead-price-forecast`.
- [x] `notebooks/03_results.ipynb` rebuilt as the detailed result report, in the style of the README figures: overview of all forecasts (with bias), the whole test quarter hour by hour, day by day, per hour of the day, per weekday and price level, the error distribution, why the ensemble works, six single days. Findings: the two parts have opposite biases (LSTM +5.4, Ridge −4.2, final model +0.6 €/MWh) and each is better on about half of the days; the 66 hours above 200 €/MWh are forecast 53 €/MWh too low on average.
- [ ] Optional: probabilistic forecasts (e.g. the 10 %, 50 % and 90 % quantiles), which show when the model is unsure, e.g. before a Dunkelflaute.

How to read the MAE (Ridge with the selected features on the validation split, MAE 20.4; the LSTM is about 10 % better): the median error is 11.8 €/MWh, 44 % of the hours are within 10 €/MWh and 73 % within 20. Without the spike days 11 and 12 December the MAE is 16.9, on these two days 173. By price level: 11 to 14 €/MWh for prices between 50 and 150, 77 above 150, 21 below 50.
- [x] README (task, data, model, development, results, limitations, usage), MIT LICENSE, tests against leakage (`python -m pytest tests`: windows end before their forecast day, splits do not overlap, the scaler sees only training rows, lags and calendar are correct; the tests that need the dataset are skipped without it).

## Phase 2 (proposal): price spikes and more data

The final model is good on normal days; its weak spots are price spikes. Phase 2 looks for their causes and the data that is missing.

**Rule:** the test split Q1 2025 is used up. Its spike days (15 and 20 January 2025) must not guide the new features, otherwise it can no longer measure a new model honestly. The spikes are analysed on training and validation data only, and a new model needs a new, untouched test period, i.e. data after March 2025. The result of phase 1 (13.9 €/MWh) stays the honest result of model v1.

First analysis (training and validation data, 2023 to 2024):
- Spikes are rare and come in episodes: about 57 hours above 300 €/MWh in two years, mainly September 2023, late-summer evenings 2024 and the Dunkelflaute in November and December 2024. Most of them are in the evening (16:00 to 20:00).
- The residual load does not explain them alone: in its top 10 % only 84 of 1755 hours are above 250 €/MWh.
- Compared with other hours of high residual load, the spike hours have less wind (3.1 against 7.4 GW), twice the imports (11.7 against 5.8 GW), more gas, oil and pumped storage, but **less lignite and hard coal** (11.7 against 13.5 and 6.4 against 7.4 GW). The gas price is the same (43 against 42 €/MWh). This points to unavailable power plants and to the situation in the neighbouring countries, neither of which is in the data.
- 26 June 2024 has prices up to 2326 €/MWh at night and in the evening of a summer day, which no fundamental explains.

Proposed steps:
- [x] Research 26 June 2024: on 25 June 2024 a technical failure at EPEX SPOT partially decoupled its markets from the European day-ahead market coupling (SDAC), so no cross-border capacity could be allocated for 26 June; in Germany the day-ahead price averaged about 492 €/MWh (the dataset gives exactly 492.0) and reached 2326 €/MWh in the morning and 1796 in the evening. The prices are no market outcome of supply and demand. To cite in the README: a primary source such as the EPEX SPOT communication or the SDAC incident report.
- [x] Consequence: `exclude_days = ["2024-06-26"]` in `configs/lstm.toml`. The day is no training target of new runs any more (627 instead of 628 training days); it stays in the history of the following days, where its prices were known values, and in the evaluation. The final model of phase 1 was trained with it and stays unchanged (`predict` still reproduces it exactly, because its saved config has no `exclude_days`).
- [ ] Research the causes of the Dunkelflaute spikes of November and December 2024 (Bundesnetzagentur, SMARD articles, press; was capacity withheld?).
- [ ] Decide the goal: a better MAE, or catching spikes (probability of a price above e.g. 250 €/MWh, quantile forecasts). The Huber loss and the MAE deliberately weigh spikes little.
- [ ] Data: power plant unavailability (ENTSO-E / REMIT), neighbouring countries' load, wind, solar, French nuclear availability and cross-border capacities (not their prices, which come from the same auction), real day-ahead forecasts; all from the ENTSO-E Transparency Platform (free API token). More history (from 2015) and newer data (from April 2025, the new test period) from SMARD.de.
- [ ] New splits with a new test period after March 2025, then steps 1 to 7 again for model v2.

## Colab runs

Push the code first; models/ and logs/ stay on Drive as before. A training run takes about one minute on the Colab GPU. In a Colab cell, `!` commands take Python variables in `{}`; no comments after `%cd`.

Step 6, the search (about 20 to 30 minutes: with early stopping a trial takes 15 to 35 seconds on the Colab GPU). If Colab disconnects, run the same cell again, the search continues where it stopped. `--trials 100` would extend the same search to 100 trials:

```python
%cd /content/day-ahead-price-forecast
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
%cd /content/day-ahead-price-forecast
!git pull
!python -m spotforecast.evaluate --ensemble
```

Step 7, first the dry run on train → validation (about 2 minutes, nothing is saved, the test split is not touched):

```python
%cd /content/day-ahead-price-forecast
!git pull
!python -m spotforecast.final --dry-run
```

Then, only if the dry run is fine and only once, the evaluation on the test split. The report is printed and written to `reports/test_results.md` of the Colab copy; copy it to Drive, then commit it to the repository:

```python
!python -m spotforecast.final
!cp reports/test_results.md /content/drive/MyDrive/spotforecast/
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
- Test split Q1 2025, measured once: **MAE 13.9** (expected 14 to 15), scaled MSE 0.34 (expected 0.5 to 0.6); 21 % below the validation MAE of 17.6, the calmer quarter as expected.
- With real forecasts instead of measured values, the error would be several €/MWh higher.
