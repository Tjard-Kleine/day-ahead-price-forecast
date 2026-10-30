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
- [x] Clean-up of unused code: `src/spotforecast/plotting.py` (plotly helpers of the old results notebook; plotly removed from the dependencies), `scripts/feature_importance.py` (replaced by `scripts/select_features.py`; its check was removed from `notebooks/02_sanity_checks.ipynb`, which now notes that its outputs show the state of step 2), `list_to_time` in `preprocessing.py`, an unused import in `train.py`, and the old local runs in `models/` and `logs/`.
- [ ] Optional: probabilistic forecasts (e.g. the 10 %, 50 % and 90 % quantiles), which show when the model is unsure, e.g. before a Dunkelflaute.

How to read the MAE (Ridge with the selected features on the validation split, MAE 20.4; the LSTM is about 10 % better): the median error is 11.8 €/MWh, 44 % of the hours are within 10 €/MWh and 73 % within 20. Without the spike days 11 and 12 December the MAE is 16.9, on these two days 173. By price level: 11 to 14 €/MWh for prices between 50 and 150, 77 above 150, 21 below 50.
- [x] README (task, data, model, development, results, limitations, usage), MIT LICENSE, tests against leakage (`python -m pytest tests`: windows end before their forecast day, splits do not overlap, the scaler sees only training rows, lags and calendar are correct; the tests that need the dataset are skipped without it).

## Phase 2: model v2 with public data and real forecasts

Model v1 is good on normal days; its weak spots are price spikes, and it used measured values of the forecast day instead of real forecasts. Phase 2 builds model v2 on public data with the real day-ahead forecasts.

**Rule:** the test split Q1 2025 is used up. Its spike days (15 and 20 January 2025) must not guide the new features, otherwise it can no longer measure a new model honestly. Model v2 is evaluated once on a new, untouched test period. The result of phase 1 (13.9 €/MWh) stays the honest result of model v1.

### Decisions

- **Goal: a realistic model.** Only what is known at 12:00 on D-1: the prices of D-1, the day-ahead forecasts of the transmission system operators for D-1 and D, measured values shifted by 12 hours, the calendar. The metric stays the MAE.
- **Data: SMARD.de** (market data under CC BY 4.0, attribution "Bundesnetzagentur | SMARD.de"). `data/raw/smard.parquet` is committed as a fixed snapshot, its download cache is not.
- **Fuel prices** (gas, coal, CO₂) are not on SMARD: v2 starts without them; they can be added later if a free source is found.
- **Splits** (`configs/data_smard.toml`): training October 2018 to March 2024 (1989 days), validation April 2024 to March 2025 (363 days, every season), test April 2025 to the end of the data (573 days), untouched until the final evaluation.
- **26 June 2024** is no training target (`exclude_days`): on 25 June 2024 a technical failure at EPEX SPOT partially decoupled its markets from the European day-ahead market coupling (SDAC); the German price averaged 492 €/MWh on 26 June and reached 2326 €/MWh, no market outcome of supply and demand. To cite in the README: a primary source such as the EPEX SPOT communication or the SDAC incident report.

### Spike analysis (training and validation data of phase 1, 2023 to 2024)

- Spikes are rare and come in episodes: about 57 hours above 300 €/MWh in two years, mainly September 2023, late-summer evenings 2024 and the Dunkelflaute in November and December 2024. Most of them are in the evening (16:00 to 20:00).
- The residual load does not explain them alone: in its top 10 % only 84 of 1755 hours are above 250 €/MWh.
- Compared with other hours of high residual load, the spike hours have less wind (3.1 against 7.4 GW), twice the imports (11.7 against 5.8 GW), more gas, oil and pumped storage, but **less lignite and hard coal** (11.7 against 13.5 and 6.4 against 7.4 GW). The gas price is the same (43 against 42 €/MWh). This points to unavailable power plants and to the situation in the neighbouring countries.
- [ ] Research the causes of the Dunkelflaute spikes of November and December 2024 (Bundesnetzagentur, SMARD articles, press; was capacity withheld?).

### Steps of model v2

- [x] **V1. Data.** `python -m spotforecast.data.smard` downloads hourly data from October 2018 to today into `data/raw/smard.parquet` (70,820 hours, 49 columns, about 20 minutes; SMARD's download center gives at most 14 days per file, so the module uses the JSON interface behind its charts, one week per request, cached in `data/raw/smard/cache/`; a later run only fetches new weeks). Price, load, all generation types, the day-ahead forecasts for Germany and Austria, the prices of the 11 neighbouring bidding zones, the commercial net export in total and per border, the installed wind and solar capacity. Every column, its meaning and what is known at 12:00 on D-1 is in `data/README.md`. Not on SMARD: weather, fuel prices, load and generation of the neighbouring countries (ENTSO-E).
  - Checked against the course data (November and December 2024): the price is identical, the load practically identical, generation correlates with 0.9999 to 1.0 with small level differences (gas about 1 GW, probably later revisions), so v2 uses SMARD only. SMARD's residual load is load − wind − solar, the definition of the pipeline; its forecast residual load is the same from the forecasts.
  - Gaps: the exchange with Belgium and Norway is 0 before their interconnectors (ALEGrO November 2020, NordLink December 2020); SMARD's total net export has gaps of up to three months in 2019 and 2020, so it is the sum of the borders (equal to SMARD's own total in all but 14 of 52,002 hours); the Polish price starts on 20 November 2019; a few single days of forecasts are missing and filled by the pipeline. Since October 2025 the auction sets quarter-hour prices; SMARD's hourly price is their mean (checked).
- [x] **V2. Pipeline.** `configs/data_smard.toml` and `configs/lstm_v2.toml`, a first model config with only what is known at 12:00 on D-1 (measured values shifted by 12 hours, the forecasts and the calendar as inputs of day D). `train`, `evaluate`, `tune`, `select_features` and `compare_runs` take `--config lstm_v2`; runs without a `data` key keep using the course data, so model v1 is still reproduced exactly. A test checks that the v2 config uses no measured value of day D.
  - First realistic numbers on the validation year: day D-1 31.1 €/MWh, **Ridge with the real forecasts 16.3**, Ridge with the measured values of day D (the perfect forecasts of phase 1) 16.7. **The real forecasts are not worse**: the auction sets the price from the forecasts, not from the later measurements. If model v2 confirms this, the README statement that v1 is an upper bound for real forecasts has to be revised.
- [x] **V3. Sanity checks** of the SMARD data (`notebooks/04_sanity_checks_v2.ipynb`, training and validation period only): all passed. Alignment of history and targets; every forecast correlates best with the measurement of the same hour (no time shift in SMARD); the 12-hour shift is exact and the information set holds; the forecast errors change only gradually over the years (the solar error grows with the capacity); the untrained model equals persistence and memorizes a batch.
  - Finding: **in the SMARD data 26 June 2024 has normal prices** (mean 103 €/MWh, almost the same as Austria); the decoupled prices up to 2326 €/MWh are only in the course data. Most likely there were two prices on that day, that of the decoupled EPEX SPOT auction (course data) and that of the European market coupling (SMARD); to be checked with the primary source. The day does not change the v2 validation numbers, so `configs/lstm_v2.toml` has no `exclude_days`; for v1 the exclusion stays right.
- [x] **V4. Feature selection** with Ridge and LightGBM, run locally (about 45 minutes; log `logs/select_features_v2.log`, results `logs/feature_selection_v2.csv`): `configs/feature_selection_v2.toml`, the base (price and forecast residual load) plus 7 groups (calendar, the single forecasts of load, wind and solar, the measured residual load shifted by 12 h, conventional generation shifted by 12 h, the prices of 7 neighbours of D-1, the net export shifted by 12 h, the Austrian forecasts), all 128 combinations, validated in the four quarters of the validation year; the script saves after every combination and continues after an interruption.

| | Ridge | LightGBM |
|---|---|---|
| Base only | 16.12 | 14.78 |
| Best combination | 15.78 (−0.34) | 13.81 (−0.97) |
| Forward selection (≥ 0.1 €/MWh, ≥ 3 of 4 folds) | no group | neighbour prices (−0.52), Austria (−0.36) → 13.90 |

| Group, average effect when added | Ridge | LightGBM |
|---|---|---|
| neighbour_prices | −0.19, better in 75 % of the folds | **−0.43, 91 %** |
| austria | +0.36, 33 % | −0.23, 69 % |
| forecast_details | −0.17, 70 % | −0.05, 50 % |
| measured_lag12, conventional_lag12 | −0.09 each | +0.03, −0.03 |
| calendar, net_export_lag12 | −0.04, −0.01 | +0.01, −0.02 |

  - **The groups matter much less than in phase 1**: the base already holds most of the information. Only the neighbours' prices of D-1 help robustly. The calendar, a strong group in phase 1, adds nothing any more; its effect is apparently already in the load forecast.
  - **LightGBM beats Ridge clearly** (14.8 against 16.1 with the base), unlike in phase 1: with almost 2000 training days the trees can learn the nonlinear merit order. LightGBM is a candidate for the v2 ensemble.
- [x] **V5. LSTM** on Colab: three candidates with the training settings of phase 1, 3 seeds each (the base; base + neighbour_prices + austria, the choice of LightGBM; base + calendar + forecast_details + measured_lag12, the starting config of V2). For a fair comparison Ridge and LightGBM were then trained on the same training split (until March 2024) and evaluated on the whole validation year like the LSTM; the numbers of V4 come from expanding folds and are not comparable with these.

| Features (MAE on the validation year, €/MWh) | LSTM, 3 seeds | Ridge | LightGBM, 1 seed |
|---|---|---|---|
| Base (price, forecast residual load) | **15.83 ± 0.21** | 16.56 | 15.16 |
| + calendar, forecast_details, measured_lag12 | 15.97 ± 0.02 | 16.25 | 15.15 |
| + neighbour_prices, austria | 17.06 ± 0.40 | 16.89 | **14.12** |

  - **The base wins among the LSTMs**: neither group set is better, so `configs/lstm_v2.toml` keeps the base. With the neighbours' prices the LSTM overfits, its best epoch is about 9.
  - **LightGBM is the best model of v2**: 0.7 €/MWh better than the LSTM with the same base, 1.7 better with the neighbours' prices and Austria (RMSE 29.2 against 30.6 of the LSTM base). Ridge stays 0.7 behind the LSTM.
  - LightGBM is now a model of the package (`fit_lgbm`, `predict_lgbm` in `src/spotforecast/models/baselines.py`); the feature selection uses the same functions and reproduces its numbers exactly.
- [ ] **V6. Ensemble and final evaluation**. LightGBM leads now; checked on the validation year was whether the mean with the LSTM seeds (and Ridge) beats LightGBM alone by at least 0.5 €/MWh, with fixed equal weights so that nothing is fitted on the validation year.
  - `configs/lgbm_v2.toml`: LightGBM with the base, the neighbours' prices and the Austrian forecasts, the mean of 3 seeds (single seeds 14.12, 14.14, 14.06). `python -m spotforecast.evaluate --ensemble --config lstm_v2 --lgbm lgbm_v2` adds LightGBM alone, + Ridge, + the LSTM seeds and + both. A test checks its information set like that of the LSTM.

| Validation year, €/MWh (Colab) | MAE | RMSE |
|---|---|---|
| Ridge | 16.6 | 30.8 |
| LSTM, mean of the 5 base runs | 15.8 | 30.4 |
| LSTM + Ridge | 14.6 | 29.0 |
| **LightGBM, mean of 3 seeds** | **13.9** | 29.2 |
| LightGBM + Ridge | 14.1 | 28.5 |
| LightGBM + LSTM | 13.6 | 28.8 |
| LightGBM + LSTM + Ridge | 13.6 | 28.3 |

  - [x] **Decision: LightGBM alone is model v2.** The ensembles are 0.3 €/MWh better, less than the 0.5 of the decision rule. LightGBM alone also needs no GPU and refits in a few minutes on a laptop CPU, which keeps the later live forecast and the regular retraining simple.
  - [x] `python -m spotforecast.final_v2` refits LightGBM on train + validation (a fixed number of trees, so nothing else changes), evaluates it once on the test period and writes `reports/test_results_v2.md`, the models go to `models/final_v2/`; with the report present it only shows it. The dry run (`--dry-run`, train → validation, nothing saved) gives 13.9 as above; Diebold-Mariano against Ridge −6.6 (p < 0.0001); per quarter 12.4, 12.6, 17.5 (the Dunkelflaute) and 13.2.
  - [x] **Tuning LightGBM** before the test (decision of the user), locally: `python -m spotforecast.tune_lgbm --trials 60` (25 minutes; log `logs/tune_lgbm_v2.log`, trials `logs/optuna_lgbm_v2.log`). Every trial fits on the training split and is validated on the whole validation year like the dry run, one seed per trial (the seeds differ by only about 0.05). Searched: number of trees, learning rate, leaves, minimum leaf size, row and column subsampling, L2 regularization and the loss (squared error, Huber, L1). Trial 0 is the current config. Afterwards the 5 best trials were confirmed with all 3 seeds (`--confirm`).

| Validation year, MAE in €/MWh | seed 1 | seed 2 | seed 3 | mean forecast | RMSE |
|---|---|---|---|---|---|
| settings of the feature selection | 14.15 | 14.15 | 14.03 | 13.93 | 29.1 |
| **trial 48** | 13.52 | 13.58 | 13.46 | **13.43** | 28.1 |
| trial 55 | 13.58 | 13.55 | 13.49 | 13.47 | 28.2 |
| trials 45, 42, 51 | | | | 13.50 to 13.53 | |

  - The top is flat: the 10 best trials lie between 13.52 and 13.61 with one seed, all with about 700 to 1500 slower trees (learning rate 0.013 to 0.022), many leaves (77 to 128), small leaves (5 to 9 samples), row subsampling of about 0.65 and some L2 regularization. The squared error wins; Huber and L1 do not help the MAE. Importance: column and row subsampling 0.20 each, number of trees 0.16, learning rate 0.13.
  - **Trial 48 passes the decision rule** (every seed at least 0.5 better, the mean forecast exactly 0.50) and is in `[params]` of `configs/lgbm_v2.toml`, unrounded: rounded to 3 digits the subsamples change and the MAE is about 13.5, so part of the gain is the luck of the best of 60 trials; on the test period about 0.4 can be expected. The dry run reproduces 13.4 (RMSE 28.0; per quarter 11.8, 11.8, 17.0, 13.2; Diebold-Mariano against Ridge −7.5). The 3 seeds now take about 2 minutes to fit instead of 30 seconds.
  - [x] **The single test evaluation** (2026-10-29, locally, `python -m spotforecast.final_v2`, report `reports/test_results_v2.md`, models `models/final_v2/`): 1 April 2025 to 28 October 2026, 573 days, trained on 2352 days until March 2025. The test period is used up.

| Test period, €/MWh | MAE | RMSE | median AE |
|---|---|---|---|
| day D-1 | 29.0 | 45.4 | 16.4 |
| **Ridge** (same features) | **16.6** | **26.5** | 11.5 |
| **LightGBM, final model v2** | **16.9** | 27.3 | **10.5** |

| MAE per quarter | 2025 Q2 | Q3 | Q4 | 2026 Q1 | Q2 | Q3 | Q4 (27 days) |
|---|---|---|---|---|---|---|---|
| LightGBM | 12.8 | 11.1 | 11.2 | 11.0 | 19.3 | 28.7 | 39.9 |
| Ridge | 15.2 | 15.1 | 13.9 | 13.1 | 21.5 | 19.9 | 20.2 |

  - **LightGBM is not better than Ridge over the whole test period** (Diebold-Mariano 0.56, p = 0.71). For 12 months, until April 2026, it is clearly better (11 to 13 against 13 to 15, as on the validation year); from May 2026 it breaks down.
  - **Diagnosis** (after the evaluation, the model stays as it is): no data error (no gaps, no frozen values). From May 2026 the price rises from 79 to 163 €/MWh (October), while the forecast residual load stays low (21 to 26 GW): the price is high at a low residual load, most likely because of higher fuel prices (gas, CO₂), which v2 does not have. LightGBM learned the relation between the residual load and the price of 2018 to 2025 and cannot extrapolate it: its bias grows from −10 €/MWh (May, June) to −28 (August) and −36 (October). Ridge, linear in the prices of D-1, carries the new level forward and stays almost unbiased (−2 to +6). The model was also not retrained for 18 months.
  - Consequences: fuel prices and a regular retraining are needed (they were already on the list for later); trees need a model of the price level (e.g. the target relative to the price of D-1) or a linear part.
- [ ] ~~**V7. Report**~~: replaced by phase 3 (decision of the user on 2026-10-30): v2 is not fundamentally better, so the work starts again with more data. The documentation is a description of the method, not a comparison of models.
- [ ] Research: when do the transmission system operators publish their day-ahead forecasts, before or after 12:00 on D-1? If after, the report has to say so.
- Later: a live forecast for tomorrow (for the portfolio web view; needs `predict` without an actual price and the forecasts of tomorrow) and a regular retraining, because the market changes; power plant outages and the neighbours' load from ENTSO-E; a probability of price spikes.

## Phase 3: a new start with fuel prices and weather (planned)

Decided by the user on 2026-10-30, after the test of v2:

- **Start again** instead of polishing v2, with **fuel prices** (gas, CO₂, possibly coal) and **weather data**; the user downloads what cannot be fetched automatically. A **quarter-hourly resolution** (the auction sets quarter-hourly prices since 1 October 2025) is an option to be weighed.
- **Documentation: the method, not a comparison.** The repository describes what was done and why, also mathematically (task and information set, scaling, models, losses, evaluation and test), and why one approach worked better than another, e.g. why trees cannot carry a new price level forward and a linear model can.

Lessons of phase 2 that the new start builds on:

- The price level changes (2026: +60 % at the same residual load). A model must either see its drivers (fuel prices) or carry the level of the last days forward (a target relative to the price of D-1, as the residual anchor of the LSTM, or a linear part), and it has to be retrained regularly.
### Decisions

- **Hourly resolution** for now: quarter-hourly prices exist only since October 2025, too little to train on; the hourly price is the mean of the quarter hours. A model of the profile within the hour can follow later.
- **Data**: fuel prices first (gas TTF and CO₂ from the EEX auctions, fetched by a script, kept out of the repository because their terms do not clearly allow it); weather from Open-Meteo (CC BY 4.0) in a second step, with the choice between real forecasts of D-1 (only from 2024), archived short-term forecasts (from 2021/2022) and reanalysis (ERA5, only shifted); the neighbours' forecasts and power plant outages from ENTSO-E later (needs a token of the user).
- **Splits as in v2** (decision of the user): training October 2018 to March 2024, validation April 2024 to March 2025, test from April 2025. A new model has not seen the test period, but we have: we know the price rise of 2026 and that v2 failed on it, and the plan of phase 3 follows from that. A test number of phase 3 is therefore slightly optimistic. Rules against it:
  1. Every decision is taken on the training and validation data with the decision rule, never on the test period; the test is evaluated once at the end, as before.
  2. The report says that the test period was already used for the evaluation of model v2. The fuel prices are not a consequence of that test: they were part of model v1, and phase 2 planned them from the start (its decision "v2 starts without them, they are added once a free source is found"); the report presents them as that plan.
  3. Optional, as a clean confirmation: the live forecasts of the portfolio view from a fixed date on are a test that nobody has seen.

### Steps of phase 3

- [x] **P1. Fuel prices.** `python -m spotforecast.data.fuel` downloads the TTF gas price (Yahoo Finance, from January 2018) and the EUA auction prices (EEX, from 2012) and writes them hourly to `data/raw/fuel.parquet` (not in the repository, see `data/README.md`). Every hour holds the price of an earlier day; models use them shifted by 12 hours (`gas price_lag12`, `CO2 price_lag12`), a test checks this for every v2/v3 config and the shift itself. No gaps from October 2018 to today. The quarterly means confirm the diagnosis of v2: the gas price doubled in 2026 (30 to 46 EUR/MWh in 2025, 85 in October 2026).
- [x] **P2. Pipeline.** `configs/data_v3.toml` extends `configs/data_smard.toml` (a config can name a `base` now) and joins `smard.parquet` and `fuel.parquet` (a data config can list several files). The splits are those of v2 (1989, 363 and 573 days).
- [ ] **P3. Weather** (Open-Meteo): the choice of the variant (real forecasts of D-1 only from 2024; archived short-term forecasts from 2021/2022; ERA5 only shifted) is open.
- [ ] **P4. Models on the validation year**, decided with the rules: feature groups with the rule of the feature selection, model changes with the decision rule; Ridge, LightGBM, the LSTM and their combinations, each with and without the fuel prices.
  - [x] **P4a. Screening** (`python scripts/screen_models.py`, 45 seconds, `logs/screen_models_v3.csv`): Ridge, LightGBM (settings of the feature selection, one seed) and their mean, trained until March 2024, MAE on the validation year in €/MWh:

| Features | Ridge | LightGBM | mean of both |
|---|---|---|---|
| base | 16.56 | 15.16 | 14.80 |
| base + fuel | 16.30 | 14.40 | 14.16 |
| base + neighbours (prices, Austria) | 16.89 | 14.12 | 14.12 |
| **base + neighbours + fuel** | 16.45 | **13.71** | 13.81 |

  - **The fuel prices help**: LightGBM −0.76 with the base, −0.41 with the neighbours, better in all four quarters; Ridge −0.26 and −0.44.
  - **The target relative to day D-1 loses everywhere** on the validation year (LightGBM +1.4 to +1.6, Ridge +0.7 to +0.8). A smoother anchor does not help either (LightGBM with base + neighbours + fuel: level 13.71, same hour of D-1 15.17, mean of the last 7 days 15.51, mean of D-1 16.34): the anchor brings the noise of single days into the target, and in a year without a large level shift the trees use the prices of D-1 better as features. **Decision: the price level stays the target**; with the fuel prices the drivers of a new level are now inputs. (`fit_ridge`, `fit_lgbm` keep the option `relative`.)
  - [x] **P4b. Feature selection** with the fuel group, locally (78 minutes; `python scripts/select_features.py --config lstm_v3 --selection feature_selection_v3`, log `logs/select_features_v3.log`, results `logs/feature_selection_v3.csv`): `configs/feature_selection_v3.toml`, the 7 groups of v2 and `fuel`, all 256 combinations, Ridge and LightGBM in the four quarterly folds of the validation year.

| Mean MAE of the 4 folds, €/MWh | Ridge | LightGBM |
|---|---|---|
| Base only | 16.12 | 14.78 |
| Forward selection, 1st group | **fuel** → 15.86 | **fuel** → 14.24 |
| Forward selection, 2nd group | forecast_details → **15.67** | austria → **13.72** |
| Best of the 256 combinations | 15.52 | 13.26 (calendar, forecast_details, neighbour_prices, austria, fuel) |

| Group, average effect when added | Ridge | LightGBM |
|---|---|---|
| **fuel** | **−0.23, better in 93 % of the folds** | **−0.34, 77 %** |
| neighbour_prices | −0.14, 72 % | −0.37, 89 % |
| austria | +0.32, 31 % | −0.26, 74 % |
| forecast_details | −0.17, 72 % | −0.08, 53 % |
| calendar, measured_lag12, conventional_lag12, net_export_lag12 | −0.01 to −0.06 | −0.01 to +0.03 |

  - **The fuel prices are the first choice of both models.** With them the neighbours' prices are no longer selected for LightGBM: on top of fuel and Austria they lower the mean by 0.13 but are better in only 2 of 4 folds. Both groups carry the price level of the coupled market, which is set by the same fuel costs, so they partly replace each other. Without fuel, the neighbours' prices were the strongest group (V4).
  - The best combination of 256 is better than the forward selection, but picking the best of many noisy candidates flatters it; the forward selection with its rule is the decision, as in V4.
  - [x] **P4c. LSTM** on Colab, `configs/lstm_v3.toml` (the LSTM of v2 on the v3 data), 3 seeds each, MAE on the validation year:

| LSTM features | MAE, €/MWh | RMSE | best epoch |
|---|---|---|---|
| base (v2, the same data without fuel; V5) | 15.83 ± 0.21 | 30.56 | |
| **base + fuel** | **15.13 ± 0.10** | 31.07 | 19 |
| base + fuel + forecast_details | 15.38 ± 0.21 | 31.31 | 17 |
| base + fuel + austria | 19.17 ± 0.78 | 34.43 | 19 |

  - **Fuel passes the decision rule for the LSTM** (0.70 better, the spread of the seeds is 0.1 to 0.2) and is in `configs/lstm_v3.toml`. More groups make it worse; with the Austrian forecasts it fails as it did with the neighbours' prices in V5.
  - LightGBM with the forward selection of P4b, `configs/lgbm_v3.toml` (base + austria + fuel, the settings of the feature selection), on the same validation year: seeds 13.94, 14.16, 13.94, their mean forecast **13.9** (RMSE 29.3); Ridge with the features of the LSTM 16.3. LightGBM stays ahead of the LSTM by 1.2. On this split it is as good as LightGBM of v2 before its tuning (13.9 as well): the gas price hardly moved in the validation year (31 to 47 €/MWh), so the fuel prices can show only a part of their value there.
  - [ ] **P4d. Tuning and combinations.** First the tuning of LightGBM v3, so that the combination is decided with the final LightGBM, then on Colab its combinations with the LSTM seeds and Ridge.
    - [x] **Tuning** of LightGBM v3 as in V6, locally (`python -m spotforecast.tune_lgbm --config lgbm_v3 --study lgbm_v3 --trials 60`, about 25 minutes, log `logs/tune_lgbm_v3.log`): the top is flat, the 5 best trials give 13.61 to 13.69 with 3 seeds against 13.86 with the settings of the feature selection; **no trial is 0.5 better with every seed, so `configs/lgbm_v3.toml` keeps them** (unlike v2, where the tuning gained 0.5). The best trials regularize more (leaves of 68 to 135 samples, L2 of 9 to 22, 280 to 500 trees), v2 went the other way; importance: leaves 0.29, column subsampling 0.19, L2 0.18.
    - [ ] **Combinations** — **on Colab, next step** — LightGBM v3 alone, with Ridge, with the 3 LSTM seeds of `configs/lstm_v3.toml` and with both, fixed equal weights (`python -m spotforecast.evaluate --ensemble --config lstm_v3 --lgbm lgbm_v3`). Decision rule: a combination has to be at least 0.5 better than LightGBM alone (13.9).
- [ ] **P5. Retraining** in the evaluation: a walk-forward over the validation year with monthly refits, as in the later live operation.
- [ ] **P6. Final evaluation** on the test period, once.
- [ ] **P7. Documentation of the method** (with the mathematics), figures, README.

## Colab runs of phase 3

P4d, the combinations of LightGBM v3 with the LSTM seeds of `configs/lstm_v3.toml` and Ridge on the validation year (about 2 minutes; the LSTM models are on Drive, LightGBM is fitted in the cell). Push the code first; `fuel.parquet` is still in the Colab session if it was not restarted, otherwise the download line fetches it again:

```python
%cd /content/spotforecast
!git pull
!pip install -q -e .
!python -m spotforecast.data.fuel
!python -m spotforecast.evaluate --ensemble --config lstm_v3 --lgbm lgbm_v3
```

P4c (done), the three LSTM candidates of phase 3 with 3 seeds each (9 runs, about 20 to 45 minutes on the GPU). Push the code first. The fuel prices are not in the repository, so the cell downloads them first (`data/raw/fuel.parquet`, about a minute; it needs `data/raw/smard.parquet`, which comes with `git pull`):

```python
%cd /content/spotforecast
!git pull
!pip install -q -e .
!python -m spotforecast.data.fuel
v3 = "--config lstm_v3 --selection feature_selection_v3"
for groups in ["--groups fuel", "--groups fuel austria", "--groups fuel forecast_details"]:
    for s in [1, 2, 3]:
        !python -m spotforecast.train {v3} --seed {s} {groups}
```

Then, in a new cell, the comparison to send back:

```python
!python scripts/compare_runs.py --config lstm_v3 --selection feature_selection_v3
```

## Colab runs of phase 2

V6, LightGBM together with the LSTM seeds of the base on the validation year (done; about 1 to 2 minutes, the LSTM models are on Drive, LightGBM is fitted in the cell). Push the code first:

```python
%cd /content/spotforecast
!git pull
!pip install -q -e .
!python -m spotforecast.evaluate --ensemble --config lstm_v2 --lgbm lgbm_v2
```

V5, the three LSTM candidates of model v2 with 3 seeds each (9 runs, about 20 to 45 minutes on the GPU). The data comes with `git pull`, because `data/raw/smard.parquet` is in the repository:

```python
%cd /content/spotforecast
!git pull
!pip install -q -e .
v2 = "--config lstm_v2 --selection feature_selection_v2"
for groups in ["", "--groups neighbour_prices austria", "--groups calendar forecast_details measured_lag12"]:
    for s in [1, 2, 3]:
        !python -m spotforecast.train {v2} --seed {s} {groups}
```

Then, in a new cell, the comparison to send back:

```python
!python scripts/compare_runs.py --config lstm_v2 --selection feature_selection_v2
```

V4, the feature selection, ran locally instead (Colab's 2 CPU cores are slower than a laptop):
`python scripts/select_features.py --config lstm_v2 --selection feature_selection_v2`.

`/content/spotforecast` is the folder of the clone on Colab (a new clone without a folder name would be called `day-ahead-price-forecast`).

## Colab runs of phase 1

Push the code first; models/ and logs/ stay on Drive as before. A training run takes about one minute on the Colab GPU. In a Colab cell, `!` commands take Python variables in `{}`; no comments after `%cd`.

Step 6, the search (about 20 to 30 minutes: with early stopping a trial takes 15 to 35 seconds on the Colab GPU). If Colab disconnects, run the same cell again, the search continues where it stopped. `--trials 100` would extend the same search to 100 trials:

```python
%cd /content/spotforecast
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
%cd /content/spotforecast
!git pull
!python -m spotforecast.evaluate --ensemble
```

Step 7, first the dry run on train → validation (about 2 minutes, nothing is saved, the test split is not touched):

```python
%cd /content/spotforecast
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
