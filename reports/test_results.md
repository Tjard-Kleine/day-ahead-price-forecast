# Test results

Day-ahead forecast of the German spot price on the test split: 2025-01-01 to 2025-03-31, 89 days,
evaluated on 2026-10-29 with commit f9f36c5.

The final model was fixed in step 6 of ROADMAP.md on the validation split: the mean of the forecasts of the LSTM of
`configs/lstm.toml` with the seeds 1, 2, 3 and of Ridge, 50/50. The LSTMs were trained on
719 days for 17 epochs each (the median best epoch of their validation runs), Ridge on the same days.
The inputs of day D (weather, wind, solar, residual load) are measured values used as perfect forecasts, so with real
forecasts the errors would be larger.

Errors in EUR/MWh (mse_z: in the scaled space of the training; median_ae: the median absolute error; within_10/20:
the share of hours with an error below 10 and 20 EUR/MWh):

| forecast | mse_z | mae_eur | rmse_eur | mae_eur_00-06 | mae_eur_06-12 | mae_eur_12-18 | mae_eur_18-24 | median_ae_eur | within_10_eur | within_20_eur |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 2.224 | 44.7 | 58.3 | 26.4 | 50.6 | 48.1 | 53.7 | 38.3 | 13% | 27% |
| persistence | 1.351 | 29.8 | 45.5 | 9.3 | 34.5 | 38.2 | 37.2 | 18.2 | 32% | 53% |
| day D-1 | 1.338 | 29.0 | 45.3 | 19.8 | 35.8 | 33.3 | 27.1 | 17.7 | 33% | 54% |
| day D-7 | 2.260 | 42.8 | 58.8 | 33.4 | 46.4 | 50.1 | 41.4 | 32.3 | 18% | 35% |
| ridge | 0.402 | 15.6 | 24.8 | 11.5 | 16.4 | 18.4 | 16.0 | 11.7 | 43% | 74% |
| lstm, seed 1 | 0.429 | 17.2 | 25.6 | 7.5 | 18.5 | 24.0 | 18.6 | 11.9 | 44% | 69% |
| lstm, seed 2 | 0.391 | 15.6 | 24.5 | 7.3 | 17.3 | 21.9 | 15.9 | 10.5 | 48% | 73% |
| lstm, seed 3 | 0.360 | 14.9 | 23.5 | 7.2 | 15.6 | 20.1 | 16.6 | 10.1 | 50% | 75% |
| lstm, mean of the seeds | 0.378 | 15.5 | 24.1 | 7.1 | 16.8 | 21.5 | 16.7 | 10.7 | 48% | 73% |
| final model: lstm mean + ridge | 0.343 | 13.9 | 22.9 | 7.9 | 14.5 | 18.2 | 15.0 | 9.5 | 52% | 78% |

Diebold-Mariano test on the daily MAE of the final model (one-sided; a negative statistic and a small p-value mean that
the final model is more accurate):

- vs ridge: statistic -3.88, p-value 0.0001
- vs lstm, mean of the seeds: statistic -4.34, p-value 0.0000
- vs day D-1: statistic -7.61, p-value 0.0000
