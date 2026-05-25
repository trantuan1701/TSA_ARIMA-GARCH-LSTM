from __future__ import annotations

from src.audit_experiment import (
    PREDICTION_COLUMNS,
    load_prediction_file,
    prediction_files,
    validate_prediction_schema,
    validate_prediction_target_alignment,
)


def test_prediction_files_exist() -> None:
    paths = prediction_files()
    assert paths, "Expected generated prediction files under outputs/predictions"


def test_prediction_files_have_exact_schema() -> None:
    for path in prediction_files():
        df = load_prediction_file(path)
        assert list(df.columns) == PREDICTION_COLUMNS, str(path)


def test_prediction_schema_values_are_valid() -> None:
    assert validate_prediction_schema() == []


def test_prediction_actuals_align_to_model_ready_targets() -> None:
    assert validate_prediction_target_alignment() == []
