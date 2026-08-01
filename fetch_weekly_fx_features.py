"""Build weekly foreign-exchange features from public BIS bulk data.

The output is suitable as an input data set for a forecasting model.  It does
not predict exchange rates by itself.

Requirements:
    pip install pandas requests
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from io import BytesIO
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


BIS_BULK_URLS = {
    "policy_rate": "https://data.bis.org/static/bulk/WS_CBPOL_csv_col.zip",
    "cpi": "https://data.bis.org/static/bulk/WS_LONG_CPI_csv_col.zip",
    "usd_fx": "https://data.bis.org/static/bulk/WS_XRU_csv_col.zip",
}

COUNTRIES = {"US": "us", "KR": "kr", "JP": "jp", "XM": "ea"}
CURRENCIES = {"KR": "KRW", "JP": "JPY", "XM": "EUR"}
DAILY_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
MONTHLY_DATE = re.compile(r"\d{4}-\d{2}")


def read_bis_rows(url: str, selections: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Download a BIS bulk ZIP and stream only the requested wide-format rows."""
    response = requests.get(
        url,
        # BIS bulk files are served through a CDN. A unique query string avoids
        # receiving an older cached ZIP after a new statistical release.
        params={"_": time.time_ns()},
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        timeout=90,
    )
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {url}, found {csv_names}")
        with archive.open(csv_names[0]) as csv_file:
            reader = csv.DictReader(TextIOWrapper(csv_file, encoding="utf-8"))
            found: dict[str, dict[str, str]] = {}
            for row in reader:
                for name, filters in selections.items():
                    if name in found:
                        continue
                    if all(str(row.get(column, "")) == str(value) for column, value in filters.items()):
                        found[name] = row

            missing = set(selections) - set(found)
            if missing:
                raise ValueError(f"BIS series selection failed: {sorted(missing)}")
            return found


def wide_row_to_series(row: dict[str, str], date_pattern: re.Pattern[str]) -> pd.Series:
    """Convert a BIS wide-format row into a numeric, date-indexed series."""
    date_columns = [column for column in row if date_pattern.fullmatch(column)]
    values = pd.to_numeric(pd.Series({column: row[column] for column in date_columns}), errors="coerce").dropna()
    values.index = pd.to_datetime(values.index)
    return values.sort_index()


def collect_policy_rates() -> pd.DataFrame:
    rows = read_bis_rows(
        BIS_BULK_URLS["policy_rate"],
        {name: {"FREQ": "D", "REF_AREA": bis_area} for bis_area, name in COUNTRIES.items()},
    )
    series: dict[str, pd.Series] = {}
    for bis_area, name in COUNTRIES.items():
        series[f"{name}_policy_rate"] = wide_row_to_series(rows[name], DAILY_DATE)
    return pd.DataFrame(series)


def collect_usd_exchange_rates() -> pd.DataFrame:
    rows = read_bis_rows(
        BIS_BULK_URLS["usd_fx"],
        {
            currency: {"FREQ": "D", "REF_AREA": bis_area, "CURRENCY": currency}
            for bis_area, currency in CURRENCIES.items()
        },
    )
    series: dict[str, pd.Series] = {}
    for bis_area, currency in CURRENCIES.items():
        series[f"usd_{currency.lower()}"] = wide_row_to_series(rows[currency], DAILY_DATE)

    rates = pd.DataFrame(series)
    # BIS quotes the selected currency against one USD.  Derive the two KRW crosses.
    rates["jpy_krw"] = rates["usd_krw"] / rates["usd_jpy"]
    rates["eur_krw"] = rates["usd_krw"] / rates["usd_eur"]
    return rates


def collect_cpi_yoy() -> pd.DataFrame:
    rows = read_bis_rows(
        BIS_BULK_URLS["cpi"],
        {
            name: {"FREQ": "M", "REF_AREA": bis_area, "UNIT_MEASURE": "771"}
            for bis_area, name in COUNTRIES.items()
        },
    )
    series: dict[str, pd.Series] = {}
    for bis_area, name in COUNTRIES.items():
        series[f"{name}_cpi_yoy"] = wide_row_to_series(rows[name], MONTHLY_DATE)
    return pd.DataFrame(series)


def build_weekly_features() -> pd.DataFrame:
    fx = collect_usd_exchange_rates()
    policy_rates = collect_policy_rates()
    cpi = collect_cpi_yoy()

    # Friday is the weekly observation date. Last available weekday is used on holidays.
    weekly = fx.join(policy_rates, how="outer").resample("W-FRI").last().ffill()

    # CPI is a monthly reference-period statistic; delaying it prevents obvious look-ahead bias.
    # Countries publish at different times. Fill each country's prior CPI value before
    # aligning dates so a partially published month cannot truncate all later weeks.
    cpi = cpi.sort_index().ffill()
    cpi.index = cpi.index + pd.Timedelta(days=21)
    weekly_cpi = cpi.reindex(weekly.index, method="ffill")
    weekly = weekly.join(weekly_cpi)

    weekly["us_kr_policy_spread"] = weekly["us_policy_rate"] - weekly["kr_policy_rate"]
    weekly["us_jp_policy_spread"] = weekly["us_policy_rate"] - weekly["jp_policy_rate"]
    weekly["us_ea_policy_spread"] = weekly["us_policy_rate"] - weekly["ea_policy_rate"]
    weekly["us_kr_cpi_spread"] = weekly["us_cpi_yoy"] - weekly["kr_cpi_yoy"]
    weekly["us_jp_cpi_spread"] = weekly["us_cpi_yoy"] - weekly["jp_cpi_yoy"]
    weekly["us_ea_cpi_spread"] = weekly["us_cpi_yoy"] - weekly["ea_cpi_yoy"]

    weekly.index.name = "week_ending"
    return weekly.dropna().sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weekly FX features from BIS data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weekly_fx_features.csv"),
        help="CSV output path (default: weekly_fx_features.csv)",
    )
    args = parser.parse_args()

    features = build_weekly_features()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, float_format="%.6f")
    print(f"{len(features)} weekly rows saved to {args.output.resolve()}")
    print(f"Period: {features.index.min().date()} to {features.index.max().date()}")


if __name__ == "__main__":
    main()
