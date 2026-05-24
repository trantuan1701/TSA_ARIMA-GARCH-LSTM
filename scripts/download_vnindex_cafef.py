#!/usr/bin/env python3
"""Download and clean daily VNINDEX historical prices from CafeF."""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import numpy as np
import pandas as pd
import requests


SYMBOL = "VNINDEX"
START_DATE = "01/01/2010"
END_DATE = "31/12/2025"
PAGE_INDEX = 1
PAGE_SIZE = 100000

CAFEF_ENDPOINTS = (
    "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx",
    "https://cafef.vn/du-lieu/ajax/pagenew/datahistory/pricehistory.ashx",
)
DATE_FORMAT_DAYFIRST = "dayfirst"
DATE_FORMAT_US = "us"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://s.cafef.vn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
}

FIELD_MAP = {
    "date": ("Ngay", "Date", "date", "trading_date"),
    "open": ("GiaMoCua", "Open", "open", "GiaMoCuaDieuChinh"),
    "high": ("GiaCaoNhat", "High", "high"),
    "low": ("GiaThapNhat", "Low", "low"),
    "close": ("GiaDongCua", "Close", "close", "GiaDieuChinh"),
    "volume": ("KhoiLuongKhopLenh", "Volume", "volume", "KhoiLuong"),
    "trading_value": (
        "GiaTriKhopLenh",
        "GtKhopLenh",
        "GiaTriGiaoDich",
        "GTGD",
        "trading_value",
    ),
}

CLEAN_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "log_return_pct",
    "squared_return",
    "abs_return",
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
]


@dataclass(frozen=True)
class DownloadResult:
    payload: dict[str, Any]
    records: list[dict[str, Any]]
    url: str
    params: dict[str, Any]
    attempts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily VNINDEX historical data from CafeF and save raw/clean files."
    )
    parser.add_argument("--symbol", default=SYMBOL, help="CafeF symbol, default: VNINDEX.")
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="Start date passed to CafeF, default: 01/01/2010.",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="End date passed to CafeF, default: 31/12/2025.",
    )
    parser.add_argument("--page-index", type=int, default=PAGE_INDEX, help="CafeF page index.")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="CafeF page size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent month downloads. Use 1 for sequential requests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project/output root. Defaults to the repository root inferred from this script.",
    )
    return parser.parse_args()


def parse_cli_date(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=False, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date argument: {value!r}")
    return pd.Timestamp(parsed)


def format_cafef_date_pair(
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_style: str,
) -> tuple[str, str]:
    if date_style == DATE_FORMAT_DAYFIRST:
        return start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y")
    if date_style == DATE_FORMAT_US:
        return start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y")
    raise ValueError(f"Unsupported CafeF date style: {date_style}")


def format_cafef_date_variants(
    start: pd.Timestamp,
    end: pd.Timestamp,
    preferred_date_style: str | None = None,
) -> list[tuple[str, str]]:
    """Return the detected CafeF date style first, with the other style as fallback."""
    styles = [DATE_FORMAT_DAYFIRST, DATE_FORMAT_US]
    if preferred_date_style in styles:
        styles.remove(preferred_date_style)
        styles.insert(0, preferred_date_style)

    variants = [format_cafef_date_pair(start, end, style) for style in styles]
    return list(dict.fromkeys(variants))


def iter_month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.replace(day=1)

    while cursor <= end:
        month_start = max(start, cursor)
        month_end = min(end, cursor + pd.offsets.MonthEnd(0))
        ranges.append((month_start.normalize(), month_end.normalize()))
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()

    return ranges


def request_cafef(
    base_url: str,
    params: dict[str, Any],
    timeout: int = 30,
    session: requests.Session | None = None,
) -> requests.Response:
    client = session or requests
    response = client.get(
        base_url,
        params=params,
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=False,
    )

    if response.is_redirect and "Location" in response.headers:
        location = urljoin(response.url, response.headers["Location"])
        if not urlparse(location).query:
            location = f"{location}?{urlencode(params)}"
        response = client.get(location, headers=HEADERS, timeout=timeout)

    response.raise_for_status()
    return response


def response_to_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        preview = response.text[:500].replace("\n", " ")
        raise ValueError(f"CafeF did not return JSON. Response preview: {preview}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from CafeF, got {type(payload).__name__}.")
    return payload


def find_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value

    if isinstance(value, dict):
        data = value.get("Data")
        if isinstance(data, dict) and isinstance(data.get("Data"), list):
            records = data["Data"]
            if all(isinstance(item, dict) for item in records):
                return records

        for child in value.values():
            records = find_records(child)
            if records:
                return records

    return []


def payload_total_count(payload: dict[str, Any]) -> int | None:
    data = payload.get("Data")
    if not isinstance(data, dict):
        return None
    total_count = data.get("TotalCount")
    try:
        return int(total_count)
    except (TypeError, ValueError):
        return None


def record_date(record: dict[str, Any]) -> pd.Timestamp:
    for candidate in FIELD_MAP["date"]:
        if candidate in record:
            return parse_cafef_date(record[candidate])
    return pd.NaT


def filter_records_to_range(
    records: list[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    valid_records = []
    for record in records:
        parsed_date = record_date(record)
        if pd.notna(parsed_date) and start <= parsed_date <= end:
            valid_records.append(record)
    return valid_records


def fetch_page(
    endpoint: str,
    params: dict[str, Any],
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], requests.Response]:
    response = request_cafef(endpoint, params, session=session)
    payload = response_to_payload(response)
    return payload, find_records(payload), response


def detect_cafef_date_style(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    page_size: int,
    attempts: list[str],
    session: requests.Session,
) -> str | None:
    for date_style in (DATE_FORMAT_DAYFIRST, DATE_FORMAT_US):
        cafe_start_date, cafe_end_date = format_cafef_date_pair(start, end, date_style)
        params = {
            "Symbol": symbol,
            "StartDate": cafe_start_date,
            "EndDate": cafe_end_date,
            "PageIndex": 1,
            "PageSize": page_size,
        }

        for endpoint in CAFEF_ENDPOINTS:
            try:
                payload, records, response = fetch_page(endpoint, params, session=session)
            except (requests.RequestException, ValueError) as exc:
                attempts.append(f"probe {date_style}: {response_url(endpoint, params)} -> {exc}")
                continue

            valid_records = filter_records_to_range(records, start, end)
            attempts.append(
                f"probe {date_style}: {response.url} -> valid={len(valid_records)} raw={len(records)}"
            )
            if valid_records:
                return date_style

    return None


def download_month_records(
    symbol: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    page_size: int,
    attempts: list[str],
    session: requests.Session,
    preferred_date_style: str | None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
    month_label = month_start.strftime("%Y-%m")
    rejected_attempts: list[str] = []

    for cafe_start_date, cafe_end_date in format_cafef_date_variants(
        month_start,
        month_end,
        preferred_date_style=preferred_date_style,
    ):
        for endpoint in CAFEF_ENDPOINTS:
            params = {
                "Symbol": symbol,
                "StartDate": cafe_start_date,
                "EndDate": cafe_end_date,
                "PageIndex": 1,
                "PageSize": page_size,
            }

            try:
                payload, records, response = fetch_page(endpoint, params, session=session)
            except (requests.RequestException, ValueError) as exc:
                rejected_attempts.append(f"{response_url(endpoint, params)} -> {exc}")
                continue

            month_records = filter_records_to_range(records, month_start, month_end)
            total_count = payload_total_count(payload) or len(records)
            attempts.append(
                f"{month_label}: {response.url} -> valid={len(month_records)} raw={len(records)} total={total_count}"
            )

            if records and not month_records:
                first_date = record_date(records[0])
                last_date = record_date(records[-1])
                rejected_attempts.append(
                    f"{response.url} returned dates outside {month_label}: {first_date} to {last_date}"
                )
                continue

            if not records and total_count == 0:
                return [], response.url, params, payload

            if not month_records:
                continue

            page_index = 2
            while len(month_records) < total_count:
                page_params = dict(params, PageIndex=page_index)
                try:
                    next_payload, next_records, next_response = fetch_page(
                        endpoint,
                        page_params,
                        session=session,
                    )
                except (requests.RequestException, ValueError) as exc:
                    raise RuntimeError(
                        f"Failed to fetch {month_label} page {page_index}: {exc}"
                    ) from exc

                next_month_records = filter_records_to_range(next_records, month_start, month_end)
                if not next_records or not next_month_records:
                    break

                month_records.extend(next_month_records)
                total_count = payload_total_count(next_payload) or total_count
                page_index += 1

            return month_records, response.url, params, payload

    rejected_summary = "\n".join(f"- {attempt}" for attempt in rejected_attempts)
    raise RuntimeError(f"No valid CafeF records for {month_label}.\n{rejected_summary}")


def response_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{endpoint}?{urlencode(params)}"


def download_month_task(
    symbol: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    page_size: int,
    preferred_date_style: str | None,
) -> tuple[pd.Timestamp, list[dict[str, Any]], str, dict[str, Any], dict[str, Any], list[str]]:
    attempts: list[str] = []
    with requests.Session() as session:
        records, url, params, payload = download_month_records(
            symbol=symbol,
            month_start=month_start,
            month_end=month_end,
            page_size=page_size,
            attempts=attempts,
            session=session,
            preferred_date_style=preferred_date_style,
        )
    return month_start, records, url, params, payload, attempts


def download_cafef_history(
    symbol: str,
    start_date: str,
    end_date: str,
    page_index: int,
    page_size: int,
    workers: int,
) -> DownloadResult:
    attempts: list[str] = []
    start = parse_cli_date(start_date).normalize()
    end = parse_cli_date(end_date).normalize()

    if page_index != 1:
        raise ValueError("Month-by-month CafeF downloads require --page-index 1.")

    all_records: list[dict[str, Any]] = []
    first_url = ""
    first_params: dict[str, Any] = {}
    last_payload: dict[str, Any] = {}

    with requests.Session() as session:
        preferred_date_style = detect_cafef_date_style(
            symbol=symbol,
            start=start,
            end=end,
            page_size=page_size,
            attempts=attempts,
            session=session,
        )

    month_ranges = iter_month_ranges(start, end)
    completed_months: list[
        tuple[pd.Timestamp, list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]
    ] = []

    if workers <= 1:
        with requests.Session() as session:
            for month_start, month_end in month_ranges:
                month_records, url, params, payload = download_month_records(
                    symbol=symbol,
                    month_start=month_start,
                    month_end=month_end,
                    page_size=page_size,
                    attempts=attempts,
                    session=session,
                    preferred_date_style=preferred_date_style,
                )
                completed_months.append((month_start, month_records, url, params, payload))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    download_month_task,
                    symbol,
                    month_start,
                    month_end,
                    page_size,
                    preferred_date_style,
                )
                for month_start, month_end in month_ranges
            ]

            for future in as_completed(futures):
                month_start, month_records, url, params, payload, month_attempts = future.result()
                attempts.extend(month_attempts)
                completed_months.append((month_start, month_records, url, params, payload))

    for month_start, month_records, url, params, payload in sorted(
        completed_months,
        key=lambda item: item[0],
    ):
        if not first_url:
            first_url = url
            first_params = params
        last_payload = payload
        all_records.extend(month_records)

    if not all_records:
        attempt_summary = "\n".join(f"- {attempt}" for attempt in attempts)
        raise RuntimeError(f"No CafeF records were downloaded. Attempts:\n{attempt_summary}")

    return DownloadResult(
        payload=last_payload,
        records=all_records,
        url=first_url,
        params=first_params,
        attempts=attempts,
    )


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_source_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns_by_normalized = {normalize_column_name(str(col)): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        normalized = normalize_column_name(candidate)
        if normalized in columns_by_normalized:
            return columns_by_normalized[normalized]
    return None


def parse_cafef_number(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return math.nan

    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", "").replace(" ", "").replace("%", "")
    text = re.sub(r"[^\d,.\-+]", "", text)
    if not text or text in {"-", "+", ".", ","}:
        return math.nan

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        integer_part, fractional_part = text.rsplit(",", 1)
        if text.count(",") > 1 or (len(fractional_part) == 3 and len(integer_part) <= 3):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    elif "." in text:
        integer_part, fractional_part = text.rsplit(".", 1)
        if text.count(".") > 1 or (len(fractional_part) == 3 and len(integer_part) <= 3):
            text = text.replace(".", "")

    parsed = pd.to_numeric(text, errors="coerce")
    return float(parsed) if pd.notna(parsed) else math.nan


def parse_cafef_date(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    text = str(value).strip()
    if not text:
        return pd.NaT

    dotnet_match = re.search(r"/Date\((-?\d+)", text)
    if dotnet_match:
        timestamp_ms = int(dotnet_match.group(1))
        return pd.to_datetime(timestamp_ms, unit="ms", utc=True).tz_convert(None).normalize()

    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def clean_cafef_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The raw CafeF dataset is empty.")

    clean = pd.DataFrame(index=raw_df.index)

    for clean_column, candidates in FIELD_MAP.items():
        source_column = find_source_column(raw_df, candidates)
        if source_column is None:
            clean[clean_column] = np.nan
            continue

        if clean_column == "date":
            clean["date"] = raw_df[source_column].map(parse_cafef_date)
        else:
            clean[clean_column] = raw_df[source_column].map(parse_cafef_number)

    clean = clean.dropna(subset=["date"])
    clean = clean.drop_duplicates(subset=["date"], keep="last")
    clean = clean.sort_values("date").reset_index(drop=True)
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")

    if "close" not in clean.columns or clean["close"].isna().all():
        raise ValueError("The cleaned dataset has no usable close column.")

    close = clean["close"].astype(float)
    clean["log_return_pct"] = 100.0 * np.log(close / close.shift(1))
    clean["log_return_pct"] = clean["log_return_pct"].replace([np.inf, -np.inf], np.nan)
    clean["squared_return"] = clean["log_return_pct"] ** 2
    clean["abs_return"] = clean["log_return_pct"].abs()

    for window in (5, 10, 20):
        clean[f"rolling_vol_{window}"] = clean["log_return_pct"].rolling(window=window).std()

    return clean[CLEAN_COLUMNS]


def output_paths(output_root: Path, symbol: str, start_date: str, end_date: str) -> dict[str, Path]:
    start = parse_cli_date(start_date)
    end = parse_cli_date(end_date)
    slug = f"{symbol.lower()}_cafef_{start.year}_{end.year}"
    return {
        "raw_csv": output_root / "data" / "raw" / f"{slug}_raw.csv",
        "clean_csv": output_root / "data" / "processed" / f"{slug}_clean.csv",
        "clean_xlsx": output_root / "data" / "processed" / f"{slug}_clean.xlsx",
    }


def save_outputs(raw_df: pd.DataFrame, clean_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(paths["raw_csv"], index=False, encoding="utf-8-sig")
    clean_df.to_csv(paths["clean_csv"], index=False, encoding="utf-8-sig")
    try:
        clean_df.to_excel(paths["clean_xlsx"], index=False, sheet_name="vnindex")
    except ImportError as exc:
        raise ImportError("Writing XLSX requires openpyxl. Install it with: pip install openpyxl") from exc


def validate_and_print(clean_df: pd.DataFrame) -> None:
    if clean_df.empty:
        raise ValueError("Validation failed: cleaned dataset is empty.")
    if "close" not in clean_df.columns:
        raise ValueError("Validation failed: cleaned dataset has no close column.")
    if len(clean_df) < 3000:
        raise ValueError(f"Validation failed: expected at least 3000 rows, got {len(clean_df)}.")

    print("Validation summary")
    print(f"Row count: {len(clean_df)}")
    print(f"Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    print(f"Columns: {list(clean_df.columns)}")
    print("Missing values:")
    print(clean_df.isna().sum().to_string())
    print("\nHead:")
    print(clean_df.head().to_string(index=False))
    print("\nTail:")
    print(clean_df.tail().to_string(index=False))


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_root, args.symbol, args.start_date, args.end_date)

    result = download_cafef_history(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        page_index=args.page_index,
        page_size=args.page_size,
        workers=args.workers,
    )

    raw_df = pd.DataFrame(result.records)
    clean_df = clean_cafef_history(raw_df)
    validate_and_print(clean_df)
    save_outputs(raw_df, clean_df, paths)

    print("\nDownload source:")
    print(result.url)
    print("\nSaved files:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
