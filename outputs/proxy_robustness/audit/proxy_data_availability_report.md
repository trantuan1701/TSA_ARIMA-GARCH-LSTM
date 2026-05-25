# Proxy Data Availability Report

## Data Files Inspected

- `data/raw/vnindex_cafef_2010_2025_raw.csv`: exists=True, has_ohlc=True; date=Ngay, open=GiaMoCua, high=GiaCaoNhat, low=GiaThapNhat, close=GiaDongCua.
- `data/processed/vnindex_cafef_2010_2025_clean.csv`: exists=True, has_ohlc=True; date=date, open=open, high=high, low=low, close=close.
- `data/processed/vnindex_model_ready.csv`: exists=True, has_ohlc=True; date=date, open=open, high=high, low=low, close=close.
- `data/processed/train.csv`: exists=True, has_ohlc=True; date=date, open=open, high=high, low=low, close=close.
- `data/processed/validation.csv`: exists=True, has_ohlc=True; date=date, open=open, high=high, low=low, close=close.
- `data/processed/test.csv`: exists=True, has_ohlc=True; date=date, open=open, high=high, low=low, close=close.

## Status

- OHLC columns are available. Proxy construction source: `processed_clean`.
- Files with OHLC columns: 6.
