"""Downloads hourly data of the German electricity market from SMARD.de (Bundesnetzagentur), the public data source of
phase 2 (ROADMAP.md). The columns are described in data/README.md.

The download center of SMARD gives at most 14 days of hourly data per file. This module uses the JSON interface behind
its charts instead, which delivers one week of a series per request. Every week is cached in data/raw/smard/cache/, so
an interrupted download continues where it stopped and a later run only fetches the new weeks. Source to cite:
Bundesnetzagentur | SMARD.de; the market data is licensed under CC BY 4.0.

    python -m spotforecast.data.smard                              # October 2018 to today -> data/raw/smard.parquet
    python -m spotforecast.data.smard --start 2024-11-01 --end 2024-12-31 --output <file>    # a short test
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

from spotforecast.config import PROJECT_ROOT

BASE_URL = "https://www.smard.de/app/chart_data"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "smard" / "cache"
OUTPUT = PROJECT_ROOT / "data" / "raw" / "smard.parquet"
WEEK_MS = 7 * 24 * 3600 * 1000
# bidding zones that border Germany/Luxembourg
NEIGHBOURS = ["AT", "BE", "CH", "CZ", "DK1", "DK2", "FR", "NL", "NO2", "PL", "SE4"]

# column -> (SMARD filter id, region). Hourly values are MWh per hour, i.e. the mean power in MW; prices in EUR/MWh.
# The names of the columns that also were in the course data of phase 1 are kept, so the pipeline works with both
SERIES = {
    # the day-ahead price of the bidding zone Germany/Luxembourg, which exists since October 2018
    "Spot": (4169, "DE-LU"),
    # realised load and generation in Germany
    "conso": (410, "DE"),
    "pumped storage consumption": (4387, "DE"),
    "onshore wind": (4067, "DE"),
    "offshore wind": (1225, "DE"),
    "solar": (4068, "DE"),
    "biomass": (4066, "DE"),
    "hydro": (1226, "DE"),
    "other renewable": (1228, "DE"),
    "nuclear": (1224, "DE"),
    "brown coal": (1223, "DE"),
    "black coal": (4069, "DE"),
    "natural gas": (4071, "DE"),
    "pumped storage": (4070, "DE"),
    "other conventional": (1227, "DE"),
    # the day-ahead forecasts of the transmission system operators, the real inputs of the forecast day
    "forecast load": (411, "DE"),
    "forecast residual load": (4362, "DE"),
    "forecast onshore wind": (123, "DE"),
    "forecast offshore wind": (3791, "DE"),
    "forecast solar": (125, "DE"),
    # Austria, the neighbour with the closest coupling
    "AT forecast load": (411, "AT"),
    "AT forecast onshore wind": (123, "AT"),
    "AT forecast solar": (125, "AT"),
    # the day-ahead prices of the neighbouring bidding zones; they come from the same auction as the German price
    "price AT": (4170, "DE-LU"),
    "price BE": (4996, "DE-LU"),
    "price CH": (259, "DE-LU"),
    "price CZ": (261, "DE-LU"),
    "price DK1": (252, "DE-LU"),
    "price DK2": (253, "DE-LU"),
    "price FR": (254, "DE-LU"),
    "price NL": (256, "DE-LU"),
    "price NO2": (4997, "DE-LU"),
    "price PL": (257, "DE-LU"),
    "price SE4": (258, "DE-LU"),
    # commercial exchanges of Germany/Luxembourg: the total net export and per border export (>= 0) and import (<= 0)
    "net export": (4629, "DE-LU"),
    "export AT": (4494, "DE-LU"), "import AT": (4512, "DE-LU"),
    "export BE": (4706, "DE-LU"), "import BE": (4708, "DE-LU"),
    "export CH": (4492, "DE-LU"), "import CH": (4510, "DE-LU"),
    "export CZ": (4493, "DE-LU"), "import CZ": (4511, "DE-LU"),
    "export DK1": (4486, "DE-LU"), "import DK1": (4504, "DE-LU"),
    "export DK2": (4487, "DE-LU"), "import DK2": (4505, "DE-LU"),
    "export FR": (4488, "DE-LU"), "import FR": (4506, "DE-LU"),
    "export NL": (4489, "DE-LU"), "import NL": (4507, "DE-LU"),
    "export NO2": (4718, "DE-LU"), "import NO2": (4720, "DE-LU"),
    "export PL": (4490, "DE-LU"), "import PL": (4508, "DE-LU"),
    "export SE4": (4491, "DE-LU"), "import SE4": (4509, "DE-LU"),
}
# realised values: at 12:00 on D-1 they are only known until about 12:00, so a model may only use them shifted by 12
# hours ("<column>_lag12"), and never as an input of day D
MEASURED = ["conso", "pumped storage consumption", "onshore wind", "offshore wind", "solar", "biomass", "hydro",
            "other renewable", "nuclear", "brown coal", "black coal", "natural gas", "pumped storage", "other conventional",
            "net export"] + [f"net export {country}" for country in NEIGHBOURS]
# the prices of the neighbours come from the same auction as the German price: known for D-1, never for day D
NEIGHBOUR_PRICES = [f"price {country}" for country in NEIGHBOURS]
# installed capacity in MW, one value per year
CAPACITIES = {
    "onshore wind capacity": (186, "DE"),
    "offshore wind capacity": (4076, "DE"),
    "solar capacity": (188, "DE"),
}


def fetch_json(url: str, retries: int=4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_week(filter_id: int, region: str, resolution: str, week: int, newest: bool, pause: float) -> list:
    cache = CACHE_DIR / f"{filter_id}_{region}_{resolution}_{week}.json"
    # the newest chunk is still being filled, so it is always fetched again
    if cache.exists() and not newest:
        return json.loads(cache.read_text())["series"]
    chunk = fetch_json(f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{week}.json")
    cache.write_text(json.dumps(chunk))
    time.sleep(pause)  # politely, it is a public service
    return chunk["series"]


def download_series(filter_id: int, region: str, start: pd.Timestamp, end: pd.Timestamp, resolution: str="hour",
                    workers: int=4, pause: float=0.1) -> pd.Series:
    """The values of one series between start and end, with timestamps in German time."""
    chunks = fetch_json(f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json")["timestamps"]
    start_ms, end_ms = start.value // 10**6, end.value // 10**6
    # the hourly chunks are weeks; chunks of coarser resolutions are kept whole (a year of capacities)
    wanted = [c for c in chunks if resolution != "hour" or (c + WEEK_MS > start_ms and c <= end_ms)]
    with ThreadPoolExecutor(workers) as pool:
        parts = pool.map(lambda c: fetch_week(filter_id, region, resolution, c, c == chunks[-1], pause), wanted)
        values = [value for part in parts for value in part]
    series = pd.Series(dict(values), dtype=float).sort_index()
    series.index = pd.to_datetime(series.index, unit="ms", utc=True).tz_convert("Europe/Berlin")
    return series


def download(start: str, end: Optional[str]=None) -> pd.DataFrame:
    """All series between start and end (local dates, end default: today) as one hourly table."""
    start = pd.Timestamp(start, tz="Europe/Berlin")
    end = pd.Timestamp(end, tz="Europe/Berlin") + pd.Timedelta(hours=23) if end else pd.Timestamp.now(tz="Europe/Berlin")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    columns = {}
    for name, (filter_id, region) in SERIES.items():
        started = time.time()
        series = download_series(filter_id, region, start, end)
        columns[name] = series[(series.index >= start) & (series.index <= end)]
        print(f"{name:28s} {columns[name].notna().sum():6d} hours ({time.time() - started:.0f} s)", flush=True)
    data = pd.DataFrame(columns)

    # the last nuclear power plants went offline on 15 April 2023, after that SMARD has no values
    data["nuclear"] = data["nuclear"].fillna(0.0)
    # the net export per border; imports are negative
    for country in NEIGHBOURS:
        data[f"net export {country}"] = data.pop(f"export {country}") + data.pop(f"import {country}")
    # before the interconnectors to Belgium (ALEGrO, November 2020) and Norway (NordLink, December 2020) the exchange
    # was zero, not unknown
    for country in ["BE", "NO2"]:
        first = data[f"net export {country}"].first_valid_index()
        data.loc[data.index < first, f"net export {country}"] = 0.0
    # SMARD's total net export has gaps of up to three months in 2019 and 2020, the sum of the borders has none
    data["net export"] = data[[f"net export {country}" for country in NEIGHBOURS]].sum(axis=1, min_count=len(NEIGHBOURS))
    # the yearly capacities hold until the next value
    for name, (filter_id, region) in CAPACITIES.items():
        yearly = download_series(filter_id, region, start - pd.DateOffset(years=1), end, resolution="year")
        data[name] = yearly.reindex(data.index.union(yearly.index)).ffill().reindex(data.index)
        print(f"{name:28s} {len(yearly):6d} yearly values", flush=True)
    # the newest hours of the measurements are still missing while the price and the forecasts reach further, so the
    # table ends with the last hour in which every column has a value
    complete = data.notna().all(axis=1)
    data = data.loc[:complete[complete].index.max()]
    data.index.name = "Date"
    return data


def main():
    parser = argparse.ArgumentParser(description="Downloads hourly German electricity market data from SMARD.de.")
    parser.add_argument("--start", default="2018-10-01", help="first day, default: the start of the DE-LU price")
    parser.add_argument("--end", help="last day, default: today")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    data = download(args.start, args.end)
    data.reset_index().to_parquet(args.output)
    print(f"\n{len(data)} hours and {data.shape[1]} columns from {data.index[0]:%Y-%m-%d} to "
          f"{data.index[-1]:%Y-%m-%d %H:%M}, saved to {args.output}")
    missing = data.isna().sum()
    print("missing hours per column:")
    print(missing[missing > 0].to_string() if missing.any() else "none")


if __name__ == '__main__':
    main()
