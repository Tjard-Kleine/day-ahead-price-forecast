# Test results of model v3

Day-ahead forecast of the German spot price on the test period: 2025-04-01 to 2026-10-28, 573 days, evaluated on
2026-11-02 with commit 1906e0c.

Model v3 was fixed on the validation year in phase 3 of ROADMAP.md: the mean of LightGBM (`configs/lgbm_v3.toml`)
and the LSTM (`configs/lstm_v3.toml`), each the mean of the seeds 1, 2 and 3. As in live operation both are refitted
at the start of every month on all days before it, with the scaling of these days; the LSTM trains 19 epochs
without early stopping (the median best epoch of its validation runs). Ridge is refitted the same way with the
features of the LSTM; day D-1 repeats the prices of the day before.

Only what is known at 12:00 on D-1 is used: the prices of D-1, the day-ahead forecasts of load, wind and solar (the
EU transparency regulation allows the wind and solar forecasts to be published until 18:00 on D-1, so the versions on
SMARD may be slightly better than those known at 12:00), the gas price (TTF settlement of D-2) and the CO2 price (the
last EUA auction before D-1). Since 1 October 2025 the auction sets quarter-hourly prices; the hourly price is their
mean.
The test period was used once before, for the evaluation of model v2 (phase 2 of ROADMAP.md).

Errors in EUR/MWh (median AE: the median absolute error; within 10/20: the share of hours with an error below 10 and
20 EUR/MWh; bias: the mean error, positive when the forecast is too high):

| forecast | MAE | RMSE | MAE 00-06 | MAE 06-12 | MAE 12-18 | MAE 18-24 | median AE | within 10 | within 20 | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| day D-1 | 29.0 | 45.4 | 19.4 | 33.2 | 33.9 | 29.3 | 16.4 | 35% | 56% | -0.3 |
| ridge | 16.1 | 26.1 | 9.5 | 16.7 | 20.7 | 17.5 | 11.0 | 46% | 73% | -0.8 |
| lightgbm | 14.7 | 24.4 | 9.8 | 14.1 | 18.8 | 16.2 | 9.6 | 52% | 77% | -4.2 |
| lstm | 14.9 | 25.6 | 8.6 | 15.4 | 18.9 | 16.9 | 9.9 | 50% | 77% | -0.7 |
| model v3 | 13.5 | 23.3 | 8.3 | 13.4 | 17.2 | 15.1 | 8.9 | 55% | 80% | -2.4 |

MAE per quarter in EUR/MWh:

| forecast | 2025Q2 (91 days) | 2025Q3 (92 days) | 2025Q4 (91 days) | 2026Q1 (89 days) | 2026Q2 (91 days) | 2026Q3 (92 days) | 2026Q4 (27 days) |
|---|---|---|---|---|---|---|---|
| day D-1 | 25.5 | 23.2 | 26.0 | 27.3 | 32.6 | 33.5 | 48.0 |
| ridge | 14.6 | 14.3 | 13.8 | 12.9 | 20.3 | 19.3 | 20.5 |
| lightgbm | 13.1 | 12.2 | 11.4 | 11.4 | 18.2 | 19.1 | 24.5 |
| lstm | 14.4 | 14.0 | 13.1 | 12.3 | 18.3 | 16.4 | 18.2 |
| model v3 | 12.8 | 12.2 | 10.6 | 10.8 | 16.6 | 16.2 | 18.9 |

Bias per quarter in EUR/MWh:

| forecast | 2025Q2 (91 days) | 2025Q3 (92 days) | 2025Q4 (91 days) | 2026Q1 (89 days) | 2026Q2 (91 days) | 2026Q3 (92 days) | 2026Q4 (27 days) |
|---|---|---|---|---|---|---|---|
| ridge | 0.6 | -3.6 | -0.6 | 1.6 | 3.2 | -3.7 | -7.7 |
| lightgbm | 0.2 | -4.5 | 0.7 | -2.3 | -5.0 | -10.1 | -17.1 |
| lstm | 2.9 | -3.3 | -6.1 | 0.2 | 4.0 | -0.6 | -5.0 |
| model v3 | 1.6 | -3.9 | -2.7 | -1.1 | -0.5 | -5.3 | -11.1 |

Diebold-Mariano test on the daily MAE of model v3 (one-sided; a negative statistic and a small p-value mean that
model v3 is more accurate):

- vs ridge: statistic -9.99, p-value 0.0000
- vs lightgbm: statistic -7.23, p-value 0.0000
- vs lstm: statistic -8.84, p-value 0.0000
- vs day D-1: statistic -20.70, p-value 0.0000
