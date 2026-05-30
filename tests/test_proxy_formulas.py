from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.stage1_extensions import construct_ohlc_proxies


def test_parkinson_proxy_uses_percent_squared_scale() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "open": [100.0] * 6,
            "high": [110.0] * 6,
            "low": [100.0] * 6,
            "close": [105.0] * 6,
        }
    )

    proxies = construct_ohlc_proxies(df)

    expected = (100.0**2) * (1.0 / (4.0 * math.log(2.0))) * (math.log(110.0 / 100.0) ** 2)
    assert proxies["parkinson"].iloc[0] == expected


def test_yang_zhang_proxy_is_trailing_and_requires_full_window() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        }
    )

    proxies = construct_ohlc_proxies(df)

    assert proxies["yang_zhang_5"].iloc[:5].isna().all()
    assert np.isfinite(proxies["yang_zhang_5"].iloc[5])
    modified = df.copy()
    modified.loc[7, "close"] = 120.0
    modified_proxies = construct_ohlc_proxies(modified)
    assert proxies["yang_zhang_5"].iloc[5] == modified_proxies["yang_zhang_5"].iloc[5]
