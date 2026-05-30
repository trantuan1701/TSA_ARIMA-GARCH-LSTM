from __future__ import annotations

import math

from src.stage1_extensions import (
    basel_zone,
    christoffersen_independence,
    kupiec_test,
    violation_cluster_lengths,
)


def test_violation_cluster_lengths() -> None:
    assert violation_cluster_lengths([False, True, True, False, True, False]) == [2, 1]
    assert violation_cluster_lengths([False, False]) == []
    assert violation_cluster_lengths([True, True]) == [2]


def test_kupiec_test_is_finite_for_zero_violations() -> None:
    stat, p_value = kupiec_test([False] * 100, alpha=0.01)

    assert math.isfinite(stat)
    assert 0.0 <= p_value <= 1.0


def test_christoffersen_independence_transition_counts() -> None:
    stat, p_value, counts = christoffersen_independence([False, True, True, False])

    assert counts == {"n00": 0, "n01": 1, "n10": 1, "n11": 1}
    assert math.isfinite(stat)
    assert 0.0 <= p_value <= 1.0


def test_basel_zone_only_applies_to_one_percent_var() -> None:
    assert basel_zone(violations=0, n_obs=250, alpha=0.05) == "not_applicable"
    assert basel_zone(violations=0, n_obs=250, alpha=0.01) == "green"
