# Test results of model v2

Day-ahead forecast of the German spot price with the SMARD data on the test period: 2025-04-01 to
2026-10-28, 573 days, evaluated on 2026-10-29 with commit 7a24985 with changes that were not committed yet.

The final model was fixed in step V6 of ROADMAP.md on the validation year: LightGBM of `configs/lgbm_v2.toml`, the
mean of the forecasts of the seeds 1, 2, 3, trained on 2352 days; Ridge on the
same days and features. The LightGBM settings: {"n_estimators": 1105, "learning_rate": 0.022340188465578614, "num_leaves": 122, "min_child_samples": 6, "subsample": 0.6867119942978767, "subsample_freq": 1, "colsample_bytree": 0.9231544244272338, "reg_lambda": 2.0240598520556343, "objective": "regression"}. Only what is known at 12:00 on D-1 is used: the prices of D-1 in Germany and in the coupled
neighbouring markets and the day-ahead forecasts of load, wind and solar for Germany and Austria (whether these are
published before 12:00 is still to be checked, see ROADMAP.md).
Since 1 October 2025 the auction sets quarter-hourly prices; the hourly price is their mean.

Errors in EUR/MWh (mse_z: in the scaled space; median_ae: the median absolute error; within_10/20: the share of hours
with an error below 10 and 20 EUR/MWh):

| forecast | mse_z | mae_eur | rmse_eur | mae_eur_00-06 | mae_eur_06-12 | mae_eur_12-18 | mae_eur_18-24 | median_ae_eur | within_10_eur | within_20_eur |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.679 | 44.9 | 62.9 | 28.2 | 44.1 | 54.1 | 53.3 | 31.7 | 21% | 37% |
| persistence | 0.595 | 38.7 | 58.9 | 13.6 | 39.3 | 67.7 | 34.0 | 21.8 | 28% | 47% |
| day D-1 | 0.354 | 29.0 | 45.4 | 19.4 | 33.2 | 33.9 | 29.3 | 16.4 | 35% | 56% |
| day D-7 | 0.508 | 35.2 | 54.4 | 25.6 | 35.5 | 43.0 | 36.7 | 20.9 | 30% | 49% |
| ridge | 0.121 | 16.6 | 26.5 | 9.9 | 17.1 | 20.9 | 18.6 | 11.5 | 44% | 72% |
| lightgbm, seed 1 | 0.128 | 16.9 | 27.3 | 12.8 | 16.1 | 21.1 | 17.8 | 10.6 | 48% | 73% |
| lightgbm, seed 2 | 0.128 | 17.0 | 27.3 | 12.6 | 16.1 | 21.4 | 17.8 | 10.4 | 48% | 73% |
| lightgbm, seed 3 | 0.129 | 17.0 | 27.4 | 12.7 | 16.0 | 21.8 | 17.5 | 10.6 | 48% | 73% |
| final model v2: lightgbm mean of the seeds | 0.128 | 16.9 | 27.3 | 12.6 | 16.0 | 21.3 | 17.6 | 10.5 | 48% | 73% |

MAE in EUR/MWh per quarter:

| forecast | 2025 Q2 (91 days) | 2025 Q3 (92 days) | 2025 Q4 (91 days) | 2026 Q1 (89 days) | 2026 Q2 (91 days) | 2026 Q3 (92 days) | 2026 Q4 (27 days) |
|---|---|---|---|---|---|---|---|
| final model v2: lightgbm mean of the seeds | 12.8 | 11.1 | 11.2 | 11.0 | 19.3 | 28.7 | 39.9 |
| ridge | 15.2 | 15.1 | 13.9 | 13.1 | 21.5 | 19.9 | 20.2 |
| day D-1 | 25.5 | 23.2 | 26.0 | 27.3 | 32.6 | 33.5 | 48.0 |

Diebold-Mariano test on the daily MAE of the final model (one-sided; a negative statistic and a small p-value mean that
the final model is more accurate):

- vs ridge: statistic 0.56, p-value 0.7119
- vs day D-1: statistic -15.39, p-value 0.0000
