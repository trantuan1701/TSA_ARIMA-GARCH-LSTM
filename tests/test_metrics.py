from __future__ import annotations

import math

import numpy as np

from src.audit_experiment import LOWER_IS_BETTER_METRICS, metric_is_improvement
from src.metrics import EPSILON, evaluate_volatility_predictions


def test_volatility_metrics_on_known_arrays() -> None:
    actual = np.array([1.0, 4.0, 9.0])
    pred = np.array([1.0, 2.0, 3.0])

    metrics = evaluate_volatility_predictions(actual, pred)

    expected_mse = ((0.0**2) + (2.0**2) + (6.0**2)) / 3.0
    expected_rmse = math.sqrt(expected_mse)
    expected_mae = (0.0 + 2.0 + 6.0) / 3.0
    expected_qlike = float(np.mean(np.log(pred) + actual / pred))

    assert metrics["mse"] == expected_mse
    assert metrics["rmse"] == expected_rmse
    assert metrics["mae"] == expected_mae
    assert metrics["qlike"] == expected_qlike
    assert metrics["n_obs"] == 3


def test_qlike_clips_predicted_variance_with_epsilon() -> None:
    actual = np.array([2.0])
    pred = np.array([0.0])

    metrics = evaluate_volatility_predictions(actual, pred, epsilon=EPSILON)

    assert metrics["mean_pred_var"] == EPSILON
    assert metrics["qlike"] == math.log(EPSILON) + 2.0 / EPSILON


def test_lower_metric_values_are_better() -> None:
    assert LOWER_IS_BETTER_METRICS == {"mse", "rmse", "mae", "qlike"}
    for metric in LOWER_IS_BETTER_METRICS:
        assert metric_is_improvement(metric, candidate=0.9, baseline=1.0)
        assert not metric_is_improvement(metric, candidate=1.1, baseline=1.0)
