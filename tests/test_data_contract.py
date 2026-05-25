from __future__ import annotations

from src.audit_experiment import (
    EXPECTED_COMMON_COUNTS,
    load_expected_split_counts,
    load_splits,
    validate_split_integrity,
    validate_target_construction,
)


def test_split_integrity_contract() -> None:
    assert validate_split_integrity() == []


def test_target_construction_contract() -> None:
    assert validate_target_construction() == []


def test_split_row_counts_match_existing_data_summary() -> None:
    expected_counts = load_expected_split_counts()
    splits = load_splits()
    assert expected_counts, "outputs/metrics/data_summary.csv should provide split row counts"
    for split_name, expected_count in expected_counts.items():
        assert len(splits[split_name]) == expected_count


def test_current_common_window_count_constants_are_documented() -> None:
    assert EXPECTED_COMMON_COUNTS == {"validation": 731, "test": 726}
