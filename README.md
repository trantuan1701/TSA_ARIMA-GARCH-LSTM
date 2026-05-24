# VNINDEX CafeF Historical Data

This project downloads daily VNINDEX historical price data from CafeF for
`01/01/2010` through `31/12/2025`, cleans it with pandas, and saves both raw
and processed files.

## Data Source

- Source: CafeF historical price data for `VNINDEX`.
- Historical page: `https://s.cafef.vn/lich-su-giao-dich-vnindex-1.chn`
- Ajax endpoint used first:
  `https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol=VNINDEX&StartDate=01/01/2010&EndDate=31/12/2025&PageIndex=1&PageSize=100000`
- CafeF's Ajax response is effectively capped/paginated by month for this
  symbol, even when a large `PageSize` is requested. The script therefore probes
  the requested endpoint, detects CafeF's accepted date format, then downloads
  each month in the period through the same endpoint.
- The script also handles CafeF's redirected lowercase endpoint.

## Outputs

The script writes:

- `data/raw/vnindex_cafef_2010_2025_raw.csv`
- `data/processed/vnindex_cafef_2010_2025_clean.csv`
- `data/processed/vnindex_cafef_2010_2025_clean.xlsx`

## Clean Variables

- `date`: trading date in `YYYY-MM-DD` format.
- `open`: opening index level.
- `high`: highest index level.
- `low`: lowest index level.
- `close`: closing index level.
- `volume`: matched trading volume from CafeF when available.
- `trading_value`: matched trading value from CafeF when available.
- `log_return_pct`: `100 * ln(close_t / close_t-1)`.
- `squared_return`: squared `log_return_pct`.
- `abs_return`: absolute `log_return_pct`.
- `rolling_vol_5`: 5-session rolling standard deviation of `log_return_pct`.
- `rolling_vol_10`: 10-session rolling standard deviation of `log_return_pct`.
- `rolling_vol_20`: 20-session rolling standard deviation of `log_return_pct`.

## How To Run

Install dependencies:

```bash
python -m pip install pandas requests openpyxl
```

Run from the project root:

```bash
python scripts/download_vnindex_cafef.py
```

Optional arguments:

```bash
python scripts/download_vnindex_cafef.py \
  --symbol VNINDEX \
  --start-date 01/01/2010 \
  --end-date 31/12/2025 \
  --page-size 100000 \
  --workers 6
```

## Validation

The script prints:

- row count
- minimum and maximum date
- columns
- missing-value counts
- head and tail

It raises an error if the dataset is empty, if `close` cannot be created, or if
the cleaned dataset has fewer than 3000 rows.
