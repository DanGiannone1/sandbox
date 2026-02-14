#!/usr/bin/env python3
"""Fetch historical daily humidity data and write it to a CSV spreadsheet."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List

import requests


DEFAULT_LATITUDE = 40.7128
DEFAULT_LONGITUDE = -74.006
DEFAULT_LOCATION_NAME = "New York City, USA"
DEFAULT_DAYS = 30
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "daily_humidity.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical daily humidity data from Open-Meteo and write it to a CSV."
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE, help="Latitude for the query.")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE, help="Longitude for the query.")
    parser.add_argument(
        "--location-name",
        type=str,
        default=DEFAULT_LOCATION_NAME,
        help="Friendly name for the location saved alongside the data.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of days of history to fetch (ignored if start/end dates are provided).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Inclusive start date in YYYY-MM-DD format. Requires --end-date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Inclusive end date in YYYY-MM-DD format. Requires --start-date.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output CSV spreadsheet (default: data/daily_humidity.csv).",
    )
    return parser.parse_args()


def resolve_date_range(days: int, start_date_str: str | None, end_date_str: str | None) -> tuple[dt.date, dt.date]:
    if start_date_str and not end_date_str or end_date_str and not start_date_str:
        raise ValueError("Both --start-date and --end-date must be provided together.")

    if start_date_str and end_date_str:
        start_date = dt.date.fromisoformat(start_date_str)
        end_date = dt.date.fromisoformat(end_date_str)
    else:
        if days < 1:
            raise ValueError("--days must be at least 1.")
        today = dt.date.today()
        end_date = today - dt.timedelta(days=1)
        start_date = end_date - dt.timedelta(days=days - 1)

    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")

    return start_date, end_date


def fetch_humidity_data(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
) -> Dict[str, List[float]]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "relative_humidity_2m_mean",
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily") or {}
    times = daily.get("time")
    humidity_values = daily.get("relative_humidity_2m_mean")

    if not times or not humidity_values:
        raise RuntimeError("API response did not include expected humidity data.")

    if len(times) != len(humidity_values):
        raise RuntimeError("Mismatched data lengths returned by API.")

    return {
        "time": times,
        "humidity": humidity_values,
        "unit": data.get("daily_units", {}).get("relative_humidity_2m_mean", "%"),
        "source": data.get("timezone", "UTC"),
    }


def write_csv(
    records: Dict[str, List[float]],
    location_name: str,
    latitude: float,
    longitude: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "date",
        "relative_humidity_mean",
        "humidity_unit",
        "location_name",
        "latitude",
        "longitude",
        "data_source",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for date_str, humidity in zip(records["time"], records["humidity"], strict=True):
            writer.writerow(
                [
                    date_str,
                    humidity,
                    records["unit"],
                    location_name,
                    latitude,
                    longitude,
                    f"Open-Meteo ({OPEN_METEO_ARCHIVE_URL})",
                ]
            )


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args.days, args.start_date, args.end_date)
    api_records = fetch_humidity_data(args.lat, args.lon, start_date, end_date)
    write_csv(api_records, args.location_name, args.lat, args.lon, args.output)
    print(
        f"Wrote {len(api_records['time'])} rows of daily humidity data for "
        f"{args.location_name} covering {start_date.isoformat()} to {end_date.isoformat()} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
