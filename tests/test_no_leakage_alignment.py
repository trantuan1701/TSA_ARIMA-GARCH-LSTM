from __future__ import annotations

import numpy as np
import pandas as pd

from src.stage1_extensions import compute_ewma_path, load_splits


def test_ewma_forecast_at_origin_uses_current_return_not_future_return() -> None:
    toy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "target_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "squared_return": [1.0, 100.0],
        }
    )

    forecasts = compute_ewma_path(toy, lambda_value=0.90, initial_variance=4.0)

    assert forecasts["pred_var"].iloc[0] == 0.90 * 4.0 + 0.10 * 1.0
    assert forecasts["pred_var"].iloc[0] != 0.90 * 4.0 + 0.10 * 100.0


def test_split_targets_are_strictly_after_forecast_origins() -> None:
    splits = load_splits()
    for split in splits.values():
        assert (split["target_date"] > split["date"]).all()
        assert np.isfinite(split["target_var_next"].to_numpy(dtype=float)).all()
