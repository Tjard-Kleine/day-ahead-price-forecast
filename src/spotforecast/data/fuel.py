"""Downloads daily fuel prices for phase 3 (ROADMAP.md) and turns them into hourly columns next to the SMARD data.

- "gas price": the settlement of the Dutch TTF natural gas front-month future in EUR/MWh, from Yahoo Finance (ticker
  TTF=F, from January 2018). For personal use only, so the data is not in the repository.
- "CO2 price": the clearing price of the EU emission allowance (EUA) primary market auctions at EEX in EUR/tCO2, the mean
  of the auctions of a day (from 2012; held on most working days, the results are published at about 11:00). EEX
  offers the yearly reports as free downloads; their terms do not clearly allow to redistribute them.

A price is only known after its trading day (the settlement in the evening), so every hour holds the last price of an
earlier day, forward-filled over weekends and holidays. A model uses them shifted by 12 hours ("gas price_lag12"), like
the measured values: at 12:00 on D-1 it then sees the gas settlement of D-2 and the last auction before D-1.

    python -m spotforecast.data.fuel    # -> data/raw/fuel.parquet, hourly like data/raw/smard.parquet
"""
import argparse
import io
import json
import urllib.request
import zipfile

import pandas as pd

from spotforecast.config import PROJECT_ROOT

# the columns; each hour holds the price of an earlier day, so they are known only shifted, like measured values
FUEL_PRICES = ["gas price", "CO2 price"]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fuel"
OUTPUT = PROJECT_ROOT / "data" / "raw" / "fuel.parquet"
SMARD = PROJECT_ROOT / "data" / "raw" / "smard.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TTF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/TTF=F?period1={start}&period2={end}&interval=1d"
EUA_HISTORY_URL = ("https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
                   "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
                   "emission-spot-primary-market-auction-report-2012-2025-data.zip")
# the report of the running year, updated after every auction
EUA_CURRENT_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
                   "emission-spot-primary-market-auction-report-{year}-data.xlsx")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return response.read()


def download_ttf(start: str="2018-01-01") -> pd.Series:
    """The daily settlement of the TTF front-month future by trading day, in EUR/MWh."""
    url = TTF_URL.format(start=int(pd.Timestamp(start, tz="UTC").timestamp()),
                         end=int(pd.Timestamp.now(tz="UTC").timestamp()))
    raw = fetch(url)
    (RAW_DIR / "ttf.json").write_bytes(raw)
    result = json.loads(raw)["chart"]["result"][0]
    # the timestamps are the starts of the trading days in the time zone of the exchange
    days = (pd.to_datetime(result["timestamp"], unit="s", utc=True)
            .tz_convert(result["meta"]["exchangeTimezoneName"]).tz_localize(None).normalize())
    prices = pd.Series(result["indicators"]["quote"][0]["close"], index=days, dtype=float)
    return prices.dropna().groupby(level=0).last().rename("gas price")


def read_eua_report(content: bytes) -> pd.DataFrame:
    """The date and auction price of every auction in one yearly EEX report; the header row is found by its "Date"
    cell, because the reports have a few title rows and some years an extra status column."""
    table = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    header_row = table.index[table.eq("Date").any(axis=1)][0]
    table.columns = table.loc[header_row].astype(str)
    table = table.loc[header_row + 1:]
    price = next(c for c in table.columns if c.startswith("Auction Price"))
    return pd.DataFrame({"date": pd.to_datetime(table["Date"], errors="coerce"),
                         "price": pd.to_numeric(table[price], errors="coerce")}).dropna()


def download_eua() -> pd.Series:
    """The mean clearing price of the EUA primary auctions of every auction day, in EUR/tCO2."""
    history = fetch(EUA_HISTORY_URL)
    (RAW_DIR / "eua_history.zip").write_bytes(history)
    reports = []
    with zipfile.ZipFile(io.BytesIO(history)) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith((".xls", ".xlsx")):
                reports.append(read_eua_report(archive.read(name)))
    last_year = max(r["date"].max().year for r in reports)
    for year in range(last_year + 1, pd.Timestamp.now().year + 1):
        content = fetch(EUA_CURRENT_URL.format(year=year))
        (RAW_DIR / f"eua_{year}.xlsx").write_bytes(content)
        reports.append(read_eua_report(content))
    auctions = pd.concat(reports)
    return auctions.groupby(auctions["date"].dt.normalize())["price"].mean().rename("CO2 price")


def hourly(daily: pd.Series, hours: pd.Series) -> pd.Series:
    """For every hour the last daily value of an earlier calendar day (Europe/Berlin)."""
    days = pd.date_range(daily.index.min(), hours.max().tz_localize(None).normalize(), freq="D")
    known_from_next_day = daily.reindex(days).ffill().shift(1)
    return pd.Series(known_from_next_day.reindex(hours.dt.tz_localize(None).dt.normalize()).to_numpy(),
                     index=hours.index, name=daily.name)


def main():
    parser = argparse.ArgumentParser(description="Downloads the daily gas and CO2 prices and writes them hourly next "
                                                 "to the SMARD data.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hours = pd.read_parquet(SMARD, columns=["Date"])["Date"]
    print("Downloading the TTF gas price from Yahoo Finance...")
    gas = download_ttf()
    print(f"  {len(gas)} trading days, {gas.index.min():%Y-%m-%d} to {gas.index.max():%Y-%m-%d}")
    print("Downloading the EUA auction results from EEX...")
    co2 = download_eua()
    print(f"  {len(co2)} auction days, {co2.index.min():%Y-%m-%d} to {co2.index.max():%Y-%m-%d}")
    data = pd.concat([hours] + [hourly(daily, hours) for daily in [gas, co2]], axis=1)[["Date"] + FUEL_PRICES]
    data.to_parquet(args.output)
    print(f"{len(data)} hours written to {args.output}; missing: {data.isna().sum().drop('Date').to_dict()}")


if __name__ == '__main__':
    main()
