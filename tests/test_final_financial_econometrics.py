from __future__ import annotations

import pandas as pd

from src.final_financial_econometrics import (
    classify_garch_failure,
    common_window_for_models,
    garch11_persistence_metrics,
)


def test_advanced_garch_failure_classifier_distinguishes_known_categories() -> None:
    nonfinite = pd.Series(
        {
            "error_message": "non-finite forecast",
            "error_type": "FloatingPointError",
        }
    )
    backcast = pd.Series(
        {
            "error_message": (
                "Due to backcasting and/or data availability start cannot be less than the index "
                "of the largest value in the right-hand-side variables used to fit the first observation."
            ),
            "error_type": "ValueError",
        }
    )

    assert classify_garch_failure(nonfinite)[0] == "nonfinite_forecast"
    assert classify_garch_failure(backcast)[0] == "invalid_forecast_start_for_ar_mean_lag"


def test_primary_common_window_aligns_by_date_not_row_order() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-04",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-02",
                    "2024-01-05",
                ]
            ),
            "actual_var": [1.0, 2.0, 3.0, 3.0, 1.0, 4.0],
            "pred_var": [1.1, 2.1, 3.1, 3.2, 1.2, 2.2],
            "model": ["A", "A", "A", "B", "B", "B"],
            "split": ["test"] * 6,
        }
    )

    common = common_window_for_models(predictions, split="test")

    assert set(common["model"]) == {"A", "B"}
    assert common.groupby("model").size().to_dict() == {"A": 2, "B": 2}
    assert common["target_date"].min() == pd.Timestamp("2024-01-02")
    assert common["target_date"].max() == pd.Timestamp("2024-01-04")


def test_garch11_persistence_metrics_are_only_reported_for_stationary_case() -> None:
    stationary = garch11_persistence_metrics(
        pd.Series({"omega": 0.02, "alpha[1]": 0.10, "beta[1]": 0.85})
    )
    assert stationary["validity_flag"] == "stationary_garch"
    assert abs(stationary["persistence_value"] - 0.95) < 1e-12
    assert stationary["unconditional_variance"] > 0
    assert stationary["shock_half_life_days"] > 0

    nonstationary = garch11_persistence_metrics(
        pd.Series({"omega": 0.02, "alpha[1]": 0.20, "beta[1]": 0.85})
    )
    assert nonstationary["validity_flag"] == "persistence_not_in_unit_interval"
    assert pd.isna(nonstationary["unconditional_variance"])
    assert pd.isna(nonstationary["shock_half_life_days"])
