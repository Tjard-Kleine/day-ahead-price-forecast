# Data

## SMARD data (phase 2): `raw/smard.parquet`

Hourly data of the German electricity market from [SMARD.de](https://www.smard.de), the market data platform of the
Bundesnetzagentur. The market data is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): it may be shared, combined and changed with the attribution
**Bundesnetzagentur | SMARD.de** ([Datennutzung](https://www.smard.de/home/datennutzung)).

```bash
python -m spotforecast.data.smard      # October 2018 to today; a later run only fetches the new weeks
```

- **Rows:** one per hour, column `Date` in German local time (`Europe/Berlin`), the start of the hour, from 1 October
  2018 (the start of the bidding zone Germany/Luxembourg) to the last hour in which every column has a value.
- **Units:** generation, load, forecasts and exchanges in MWh per hour, i.e. the mean power in MW; prices in EUR/MWh;
  capacities in MW.
- **Pipeline:** `configs/data_smard.toml`; a model config uses it with `data = "data_smard"` (see `configs/lstm_v2.toml`).

### Columns and what is known at the forecast time

The forecast for day D is made at 12:00 on D-1. "History" means the value may be used up to the end of D-1, "shift
12 h" that only values until about 12:00 on D-1 are known (in the pipeline: `<column>_lag12`), "day D" that it may
also be an input of the forecast day.

| Column | Meaning | SMARD id, region | Known at 12:00 on D-1 |
|---|---|---|---|
| `Spot` | day-ahead price Germany/Luxembourg, **the target** | 4169, DE-LU | history (set on D-2) |
| `conso` | realised load (Netzlast) | 410, DE | shift 12 h |
| `onshore wind`, `offshore wind`, `solar` | realised generation | 4067, 1225, 4068, DE | shift 12 h |
| `biomass`, `hydro`, `other renewable` | realised generation | 4066, 1226, 1228, DE | shift 12 h |
| `nuclear`, `brown coal`, `black coal`, `natural gas`, `other conventional` | realised generation | 1224, 1223, 4069, 4071, 1227, DE | shift 12 h |
| `pumped storage`, `pumped storage consumption` | generation and consumption of pumped storage | 4070, 4387, DE | shift 12 h |
| `forecast load`, `forecast residual load` | day-ahead forecasts of the transmission system operators | 411, 4362, DE | history and day D |
| `forecast onshore wind`, `forecast offshore wind`, `forecast solar` | day-ahead forecasts | 123, 3791, 125, DE | history and day D |
| `AT forecast load`, `AT forecast onshore wind`, `AT forecast solar` | day-ahead forecasts for Austria | 411, 123, 125, AT | history and day D |
| `price AT`, `price BE`, `price CH`, `price CZ`, `price DK1`, `price DK2`, `price FR`, `price NL`, `price NO2`, `price PL`, `price SE4` | day-ahead prices of the neighbouring bidding zones | 4170, 4996, 259, 261, 252, 253, 254, 256, 4997, 257, 258, DE-LU | history (same auction as `Spot`, so **never day D**) |
| `net export` | commercial net export of Germany/Luxembourg (negative: import) | 4629, DE-LU | shift 12 h |
| `net export AT` ... `net export SE4` | commercial net export per border: export + import | export and import series per border, DE-LU | shift 12 h |
| `onshore wind capacity`, `offshore wind capacity`, `solar capacity` | installed capacity, one value per year | 186, 4076, 188, DE | history and day D |

Computed in the pipeline (`src/spotforecast/data/features.py`): `residual load` = `conso` − wind − `solar` (the SMARD
definition, identical to its own series), the calendar (`saturday`, `sunday`, `holiday`) and shifted columns
(`<column>_lag<hours>`).

### Notes

- **Checked against the course data** of phase 1 (November and December 2024): the price is identical, the load
  practically identical; generation correlates with 0.9999 to 1.0 but has small level differences (gas about 1 GW on
  average, probably later revisions). Use one source for a model, do not mix them.
- **Forecasts** are real forecasts with real errors, e.g. a mean of 3.5 GW for the residual load and 1.3 GW for onshore
  wind (November and December 2024). Their exact publication time on D-1 should be checked; if it is after 12:00, the
  report has to say so.
- **Quarter hours:** since 1 October 2025 the day-ahead auction sets quarter-hour prices. `Spot` stays hourly: SMARD's
  hourly price is the mean of the four quarter hours (checked for a week in November 2025).
- **26 June 2024:** a technical failure at EPEX SPOT partially decoupled the German market from the European market
  coupling. SMARD has normal prices on that day (mean 103 EUR/MWh, almost the same as Austria); the course data of
  phase 1 has the decoupled prices up to 2326 EUR/MWh. Most likely there were two prices, that of the decoupled EPEX
  SPOT auction and that of the market coupling; SMARD has the latter. No `exclude_days` is needed for the SMARD data.
- **Nuclear:** the last plants went offline on 15 April 2023; SMARD has no values after that, the column is 0.
- **Capacities** are yearly values, carried forward until the next year.
- **Not on SMARD:** weather, fuel prices (gas, coal, CO₂), load and generation of the neighbouring countries
  (ENTSO-E Transparency Platform).

## Course data (phase 1): `raw/energy_ger.parquet`

The dataset of the university course that model v1 was trained and tested on (15-minute values, January 2023 to March
2025, `configs/data.toml`). It is not part of the repository; its electricity columns come from SMARD, the weather
columns `wind_mean` and `solar_mean` from an unknown source.
