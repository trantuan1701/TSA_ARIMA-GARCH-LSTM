from __future__ import annotations

import numpy as np
import pandas as pd

from src.train_lstm_hybrid import create_sequences, create_sequences_with_context


def toy_split(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "target_date": dates + pd.Timedelta(days=1),
            "target_var_next": np.arange(periods, dtype=float) + 1.0,
            "_target_var_next_raw": [str(value) for value in np.arange(periods, dtype=float) + 1.0],
        }
    )


def test_context_sequences_recover_early_test_origins_without_future_rows() -> None:
    train = toy_split("2024-01-01", 5)
    test = toy_split("2024-01-06", 3)
    context = pd.concat([train, test], ignore_index=True)
    features = np.arange(len(context) * 2, dtype=np.float32).reshape(len(context), 2)

    split_local_x, _split_local_y, split_local_meta = create_sequences(
        test,
        features[len(train) :],
        seq_len=3,
        split_name="test",
    )
    context_x, context_y, context_meta = create_sequences_with_context(
        context,
        features,
        test,
        seq_len=3,
        split_name="test",
    )

    assert len(split_local_x) == 1
    assert len(split_local_meta) == 1
    assert len(context_x) == 3
    assert len(context_meta) == 3
    assert context_meta["date"].iloc[0] == test["date"].iloc[0]
    assert context_y[0] == test["target_var_next"].iloc[0]
    assert np.array_equal(context_x[0], features[3:6])
