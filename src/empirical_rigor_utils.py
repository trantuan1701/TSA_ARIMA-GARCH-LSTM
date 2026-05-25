#!/usr/bin/env python3
"""Shared read-only helpers for empirical rigor scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover - fallback for package-style imports.
    from .metrics import EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
VALID_SPLITS = {"validation", "test"}
BENCHMARK_MODEL = "GARCH(1,1)"

CANONICAL_MODELS = [
    "HistoricalMean",
    "RollingVol-5",
    "RollingVol-10",
    "RollingVol-20",
    "GARCH(1,1)",
    "ARIMA-GARCH",
    "LSTM",
    "ARIMA-GARCH-LSTM",
    "LSTM-LogTarget",
    "Hybrid-LogTarget",
    "LSTM-QLIKE",
    "Hybrid-QLIKE",
    "LSTM-LogTarget-Small",
    "Hybrid-LogTarget-Small",
]


@dataclass(frozen=True)
class CommonWindow:
    """Long and wide representations of the common test window."""

    long: pd.DataFrame
    wide: pd.DataFrame
    models: list[str]
    common_keys: pd.DataFrame


def rel_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def ensure_output_dirs() -> None:
    for path in (TABLES_DIR, FIGURES_DIR, AUDIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def ordered_models(models: list[str] | pd.Series | np.ndarray) -> list[str]:
    order = {model: index for index, model in enumerate(CANONICAL_MODELS)}
    unique = sorted({str(model) for model in models})
    return sorted(unique, key=lambda model: (order.get(model, len(order)), model))


def prediction_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    pred_dir = project_root / "outputs" / "predictions"
    return sorted(path for path in pred_dir.rglob("*.csv") if path.is_file())


def _validate_prediction_frame(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if list(df.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"{rel_path(source_path)} has columns {list(df.columns)}, "
            f"expected exactly {PREDICTION_COLUMNS}."
        )

    parsed = df[PREDICTION_COLUMNS].copy()
    for column in ("date", "target_date"):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        bad_count = int(parsed[column].isna().sum())
        if bad_count:
            raise ValueError(f"{rel_path(source_path)} has {bad_count} bad {column} values.")

    for column in ("actual_var", "pred_var"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        values = parsed[column].to_numpy(dtype=float)
        bad_count = int((~np.isfinite(values)).sum())
        if bad_count:
            raise ValueError(f"{rel_path(source_path)} has {bad_count} non-finite {column} values.")

    if (parsed["actual_var"] < 0).any():
        raise ValueError(f"{rel_path(source_path)} contains negative actual_var values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{rel_path(source_path)} contains non-positive pred_var values.")

    parsed["model"] = parsed["model"].astype(str).str.strip()
    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    if parsed["model"].eq("").any():
        raise ValueError(f"{rel_path(source_path)} contains blank model values.")

    invalid_splits = sorted(set(parsed.loc[~parsed["split"].isin(VALID_SPLITS), "split"]))
    if invalid_splits:
        raise ValueError(
            f"{rel_path(source_path)} has invalid split values {invalid_splits}; "
            f"expected {sorted(VALID_SPLITS)}."
        )

    duplicate_mask = parsed.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicate_mask.any():
        examples = parsed.loc[
            duplicate_mask, ["model", "split", "date", "target_date"]
        ].head(10)
        raise ValueError(
            f"{rel_path(source_path)} has duplicate model/split/date/target_date rows:\n"
            f"{examples.to_string(index=False)}"
        )

    parsed["_source_file"] = rel_path(source_path)
    return parsed.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def load_all_predictions(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, list[Path]]:
    paths = prediction_files(project_root)
    if not paths:
        raise FileNotFoundError(f"No prediction CSV files found under {rel_path(PREDICTIONS_DIR)}.")

    frames = []
    for path in paths:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(_validate_prediction_frame(raw, path))

    predictions = pd.concat(frames, ignore_index=True)
    duplicate_mask = predictions.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicate_mask.any():
        examples = predictions.loc[
            duplicate_mask, ["_source_file", "model", "split", "date", "target_date"]
        ].head(10)
        raise ValueError(
            "Combined predictions contain duplicate model/split/date/target_date rows:\n"
            f"{examples.to_string(index=False)}"
        )

    predictions = predictions.sort_values(["split", "date", "target_date", "model"]).reset_index(
        drop=True
    )
    return predictions, paths


def build_common_test_window(predictions: pd.DataFrame) -> CommonWindow:
    test_predictions = predictions[predictions["split"] == "test"].copy()
    if test_predictions.empty:
        raise ValueError("No test-split predictions are available.")

    models = ordered_models(test_predictions["model"].unique())
    key_sets: dict[str, set[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for model in models:
        model_keys = set(
            test_predictions.loc[
                test_predictions["model"] == model, ["date", "target_date"]
            ].itertuples(index=False, name=None)
        )
        if not model_keys:
            raise ValueError(f"Model {model} has no test-split prediction keys.")
        key_sets[model] = model_keys

    common_key_set = set.intersection(*key_sets.values())
    if not common_key_set:
        raise ValueError("No exact common test window exists across prediction models.")

    common_keys = pd.DataFrame(sorted(common_key_set), columns=["date", "target_date"])
    key_filter = common_keys.copy()
    key_filter["split"] = "test"
    common_long = test_predictions.merge(
        key_filter, on=["split", "date", "target_date"], how="inner"
    )
    common_long = common_long.sort_values(["date", "target_date", "model"]).reset_index(drop=True)

    expected_count = len(common_keys)
    counts = common_long.groupby("model").size()
    bad_counts = counts[counts != expected_count]
    if not bad_counts.empty:
        raise ValueError(
            "Common-window alignment produced unequal row counts by model:\n"
            f"{bad_counts.to_string()}"
        )

    actual_by_key = common_long.groupby(["date", "target_date"])["actual_var"].agg(["min", "max"])
    if not np.isclose(
        actual_by_key["min"].to_numpy(dtype=float),
        actual_by_key["max"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ).all():
        raise ValueError("Common-window predictions have inconsistent actual_var values.")

    actual = (
        common_long.groupby(["date", "target_date"], as_index=False)["actual_var"]
        .first()
        .sort_values(["date", "target_date"])
    )
    pred_wide = common_long.pivot(
        index=["date", "target_date"], columns="model", values="pred_var"
    ).reset_index()
    pred_wide.columns.name = None
    wide = actual.merge(pred_wide, on=["date", "target_date"], how="left")
    wide = wide[["date", "target_date", "actual_var", *models]]
    return CommonWindow(long=common_long, wide=wide, models=models, common_keys=common_keys)


def loss_values(
    actual_var: pd.Series | np.ndarray,
    pred_var: pd.Series | np.ndarray,
    loss_type: str,
) -> np.ndarray:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    pred = np.clip(pred, EPSILON, None)

    if actual.shape[0] != pred.shape[0]:
        raise ValueError("actual_var and pred_var must have equal length.")

    if loss_type == "qlike":
        return np.log(pred) + actual / pred
    if loss_type == "squared_error":
        return (actual - pred) ** 2
    if loss_type == "absolute_error":
        return np.abs(actual - pred)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_metrics(df: pd.DataFrame, *, pred_column: str = "pred_var") -> dict[str, float | int]:
    if df.empty:
        raise ValueError("Cannot compute metrics on an empty frame.")
    squared_error = loss_values(df["actual_var"], df[pred_column], "squared_error")
    absolute_error = loss_values(df["actual_var"], df[pred_column], "absolute_error")
    qlike = loss_values(df["actual_var"], df[pred_column], "qlike")
    pred = np.clip(pd.to_numeric(df[pred_column], errors="coerce").to_numpy(dtype=float), EPSILON, None)
    actual = pd.to_numeric(df["actual_var"], errors="coerce").to_numpy(dtype=float)
    return {
        "n_obs": int(len(df)),
        "RMSE": float(np.sqrt(np.mean(squared_error))),
        "MAE": float(np.mean(absolute_error)),
        "QLIKE": float(np.mean(qlike)),
        "mean_actual_var": float(np.mean(actual)),
        "mean_pred_var": float(np.mean(pred)),
    }


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_float(value: float | int | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if not np.isfinite(float(value)):
        return "not_available"
    return f"{float(value):.{digits}f}"
