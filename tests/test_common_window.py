from __future__ import annotations

import pandas as pd

from src.audit_experiment import (
    EXPECTED_COMMON_COUNTS,
    common_window_summary,
    validate_common_window,
)


def test_common_window_contract() -> None:
    assert validate_common_window() == []


def test_common_window_counts_fail_loudly_if_changed() -> None:
    summary = common_window_summary()
    for split, expected_count in EXPECTED_COMMON_COUNTS.items():
        assert summary[split]["common_count"] == expected_count
        assert len(summary[split]["wide"]) == expected_count
        assert set(pd.to_numeric(summary[split]["metrics"]["n_obs"], errors="raise")) == {expected_count}


def test_common_window_wide_tables_have_identical_keys_for_all_models() -> None:
    summary = common_window_summary()
    for split, item in summary.items():
        wide = item["wide"]
        pred_columns = [column for column in wide.columns if column.startswith("pred_")]
        assert pred_columns, split
        assert not wide.duplicated(["date", "target_date"]).any()
        assert wide[pred_columns].notna().all().all()
        assert len(pred_columns) == len(item["models"])
