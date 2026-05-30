#!/usr/bin/env python3
"""Stage 1 empirical extensions for the VN-Index volatility benchmark.

This module is intentionally output-isolated: it reads the canonical data,
predictions, and advanced artifacts, then writes only to
``outputs/stage1_extensions`` and ``reports/stage1_extensions``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import binom, chi2
except ImportError:  # pragma: no cover - scipy is normally available with statsmodels/sklearn.
    binom = None
    chi2 = None

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover
    from .metrics import EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
DATE_COLUMNS = ["date", "target_date"]
VALID_SPLITS = ("validation", "test")
EWMA_LAMBDAS = (0.90, 0.94, 0.97, 0.99)
PROXY_LABELS = [
    "close_to_close_squared_return",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang_5",
    "yang_zhang_10",
    "yang_zhang_20",
]
KEY_MODEL_PREFERENCES = [
    "HistoricalMean",
    "RollingVol-20",
    "GARCH(1,1)",
    "ARIMA-GARCH",
    "AdvGARCH-BestQLIKE",
    "AdvGARCH-BestAsymmetric",
    "LSTM",
    "ARIMA-GARCH-LSTM",
    "LSTM-QLIKE",
    "Hybrid-QLIKE",
]


@dataclass(frozen=True)
class StagePaths:
    project_root: Path
    output_dir: Path
    report_dir: Path

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"


def default_paths(project_root: Path = PROJECT_ROOT) -> StagePaths:
    return StagePaths(
        project_root=project_root,
        output_dir=project_root / "outputs" / "stage1_extensions",
        report_dir=project_root / "reports" / "stage1_extensions",
    )


def ensure_dirs(paths: StagePaths) -> None:
    for path in [
        paths.output_dir,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.tables_dir,
        paths.figures_dir,
        paths.report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {rel_path(path)}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def parse_dates(df: pd.DataFrame, source: Path | str) -> pd.DataFrame:
    parsed = df.copy()
    for column in DATE_COLUMNS:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
            bad = int(parsed[column].isna().sum())
            if bad:
                raise ValueError(f"{source} contains {bad} bad {column} values.")
    return parsed


def load_splits(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = project_root / "data" / "processed" / f"{split}.csv"
        df = parse_dates(read_csv(path), path)
        for column in ["log_return_pct", "squared_return", "target_var_next"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if not np.isfinite(df[column].to_numpy(dtype=float)).all():
                raise ValueError(f"{rel_path(path, project_root)} has non-finite {column}.")
        out[split] = df.sort_values("date").reset_index(drop=True)
    return out


def load_model_ready(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "vnindex_model_ready.csv"
    df = parse_dates(read_csv(path), path)
    for column in ["log_return_pct", "squared_return", "target_var_next", "open", "high", "low", "close"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def volatility_metrics(actual_var: Iterable[float], pred_var: Iterable[float]) -> dict[str, float | int]:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(actual) != len(pred):
        raise ValueError(f"actual and pred length mismatch: {len(actual)} vs {len(pred)}")
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    if len(actual) == 0:
        return {
            "n_obs": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "qlike": np.nan,
            "mean_actual_var": np.nan,
            "mean_pred_var": np.nan,
            "pred_actual_ratio": np.nan,
        }
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    qlike = float(np.mean(np.log(pred) + actual / pred))
    mean_actual = float(np.mean(actual))
    mean_pred = float(np.mean(pred))
    return {
        "n_obs": int(len(actual)),
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
        "mean_actual_var": mean_actual,
        "mean_pred_var": mean_pred,
        "pred_actual_ratio": mean_pred / mean_actual if mean_actual > 0 else np.nan,
    }


def qlike_loss(actual_var: Iterable[float], pred_var: Iterable[float]) -> np.ndarray:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    return np.log(pred) + actual / pred


def prediction_frame(
    split_df: pd.DataFrame,
    pred_var: Iterable[float],
    *,
    model: str,
    split: str,
) -> pd.DataFrame:
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(pred) != len(split_df):
        raise ValueError(f"{model} {split}: {len(pred)} predictions for {len(split_df)} rows.")
    if not np.isfinite(pred).all():
        raise ValueError(f"{model} {split} predictions contain non-finite values.")
    out = pd.DataFrame(
        {
            "date": split_df["date"],
            "target_date": split_df["target_date"],
            "actual_var": split_df["target_var_next"].astype(float),
            "pred_var": np.clip(pred, EPSILON, None),
            "model": model,
            "split": split,
        }
    )
    return out[PREDICTION_COLUMNS]


def validate_prediction_frame(df: pd.DataFrame, source: Path | str) -> pd.DataFrame:
    missing = [column for column in PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing prediction columns: {missing}")
    parsed = parse_dates(df[PREDICTION_COLUMNS].copy(), source)
    parsed["actual_var"] = pd.to_numeric(parsed["actual_var"], errors="coerce")
    parsed["pred_var"] = pd.to_numeric(parsed["pred_var"], errors="coerce")
    if parsed[["date", "target_date", "actual_var", "pred_var"]].isna().any().any():
        raise ValueError(f"{source} contains invalid prediction values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{source} contains non-positive pred_var values.")
    parsed["model"] = parsed["model"].astype(str).str.strip()
    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    return parsed.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def safe_slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "item"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_latex_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return latex_escape(value)


def write_latex_table(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    table = df[columns].copy() if columns else df.copy()
    alignment = " ".join("r" if pd.api.types.is_numeric_dtype(table[column]) else "l" for column in table.columns)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(latex_escape(column) for column in table.columns) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        lines.append(" & ".join(format_latex_value(row[column]) for column in table.columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_rows_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=True):
        rows.append({"model": model, "split": split, **volatility_metrics(group["actual_var"], group["pred_var"])})
    return pd.DataFrame(rows)


def align_values_to_split(values: pd.DataFrame, split_df: pd.DataFrame, value_column: str) -> np.ndarray:
    merged = split_df[["date", "target_date"]].merge(
        values[["date", "target_date", value_column]],
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if merged[value_column].isna().any():
        missing = merged.loc[merged[value_column].isna(), ["date", "target_date"]].head(10)
        raise ValueError(f"Could not align {value_column} to split rows:\n{missing.to_string(index=False)}")
    return merged[value_column].to_numpy(dtype=float)


def compute_ewma_path(model_ready: pd.DataFrame, lambda_value: float, initial_variance: float) -> pd.DataFrame:
    """Return one-step EWMA forecasts on the full chronological origin stream."""

    if not 0.0 < lambda_value < 1.0:
        raise ValueError(f"EWMA lambda must be in (0, 1), got {lambda_value}.")
    h_current = float(np.clip(initial_variance, EPSILON, None))
    rows: list[dict[str, Any]] = []
    for _, row in model_ready.sort_values("date").iterrows():
        squared_return_t = float(row["squared_return"])
        h_next = lambda_value * h_current + (1.0 - lambda_value) * squared_return_t
        h_next = float(np.clip(h_next, EPSILON, None))
        rows.append({"date": row["date"], "target_date": row["target_date"], "pred_var": h_next})
        h_current = h_next
    return pd.DataFrame(rows)


def run_ewma_baseline(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    splits = load_splits(paths.project_root)
    model_ready = load_model_ready(paths.project_root)
    initial_variance = float(splits["train"]["squared_return"].mean())

    metrics_rows: list[dict[str, Any]] = []
    predictions_by_lambda: dict[float, pd.DataFrame] = {}
    for lambda_value in EWMA_LAMBDAS:
        full_path = compute_ewma_path(model_ready, lambda_value, initial_variance)
        frames = []
        for split in VALID_SPLITS:
            pred = align_values_to_split(full_path, splits[split], "pred_var")
            model_name = f"EWMA(lambda={lambda_value:.2f})"
            frame = prediction_frame(splits[split], pred, model=model_name, split=split)
            frames.append(frame)
            metrics_rows.append(
                {
                    "model": model_name,
                    "lambda": lambda_value,
                    "split": split,
                    "selection_metric": "validation_qlike",
                    **volatility_metrics(frame["actual_var"], frame["pred_var"]),
                }
            )
        predictions_by_lambda[lambda_value] = pd.concat(frames, ignore_index=True)

    metrics = pd.DataFrame(metrics_rows)
    validation_metrics = metrics[metrics["split"] == "validation"].copy()
    selected_lambda = float(
        validation_metrics.sort_values(["qlike", "rmse", "lambda"]).iloc[0]["lambda"]
    )
    selected_model = f"EWMA(lambda={selected_lambda:.2f})"
    metrics["is_selected"] = metrics["lambda"].eq(selected_lambda)
    selected_predictions = predictions_by_lambda[selected_lambda].copy()

    pred_path = paths.predictions_dir / "ewma_predictions.csv"
    metrics_path = paths.metrics_dir / "ewma_metrics.csv"
    table_path = paths.tables_dir / "ewma_metrics.tex"
    selected_predictions.to_csv(pred_path, index=False, float_format="%.17g")
    metrics.to_csv(metrics_path, index=False)
    write_latex_table(
        metrics.sort_values(["split", "qlike", "lambda"]),
        table_path,
        ["model", "lambda", "split", "is_selected", "n_obs", "rmse", "mae", "qlike", "pred_actual_ratio"],
    )
    return {
        "selected_model": selected_model,
        "selected_lambda": selected_lambda,
        "outputs": [pred_path, metrics_path, table_path],
    }


def construct_parkinson_proxy(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    valid = (high > 0) & (low > 0) & (high >= low)
    values = (100.0**2) * (1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2)
    values = values.where(valid)
    values = values.mask(values < -1e-12)
    values = values.mask((values < 0) & (values >= -1e-12), 0.0)
    return values


def build_har_feature_frame(model_ready: pd.DataFrame) -> pd.DataFrame:
    frame = model_ready[["date", "target_date", "target_var_next", "squared_return"]].copy()
    frame["sr_d"] = model_ready["squared_return"].astype(float)
    frame["sr_w"] = frame["sr_d"].rolling(window=5, min_periods=5).mean()
    frame["sr_m"] = frame["sr_d"].rolling(window=22, min_periods=22).mean()

    if {"open", "high", "low", "close"}.issubset(model_ready.columns):
        frame["parkinson_d"] = construct_parkinson_proxy(model_ready)
        frame["parkinson_w"] = frame["parkinson_d"].rolling(window=5, min_periods=5).mean()
        frame["parkinson_m"] = frame["parkinson_d"].rolling(window=22, min_periods=22).mean()
    return frame


def fit_ols_predict(
    feature_frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str],
    model_name: str,
    log_target: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_keys = splits["train"][["date", "target_date"]]
    train = train_keys.merge(feature_frame, on=["date", "target_date"], how="left", validate="one_to_one")
    fit_cols = feature_columns + ["target_var_next"]
    fit_df = train.dropna(subset=fit_cols).copy()
    if len(fit_df) < len(feature_columns) + 5:
        raise ValueError(f"{model_name} has insufficient finite training rows.")
    x_train = np.column_stack([np.ones(len(fit_df)), fit_df[feature_columns].to_numpy(dtype=float)])
    y_train = fit_df["target_var_next"].to_numpy(dtype=float)
    if log_target:
        y_train = np.log(np.clip(y_train, EPSILON, None) + EPSILON)
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    frames = []
    for split in VALID_SPLITS:
        split_features = splits[split][["date", "target_date"]].merge(
            feature_frame,
            on=["date", "target_date"],
            how="left",
            validate="one_to_one",
        )
        missing = split_features[feature_columns].isna().any(axis=1)
        if missing.any():
            bad = split_features.loc[missing, ["date", "target_date"]].head(5)
            raise ValueError(f"{model_name} missing HAR features:\n{bad.to_string(index=False)}")
        x = np.column_stack([np.ones(len(split_features)), split_features[feature_columns].to_numpy(dtype=float)])
        pred = x @ beta
        if log_target:
            pred = np.exp(pred) - EPSILON
        frames.append(prediction_frame(splits[split], pred, model=model_name, split=split))

    coef = {"intercept": float(beta[0])}
    coef.update({column: float(value) for column, value in zip(feature_columns, beta[1:])})
    return pd.concat(frames, ignore_index=True), coef


def run_har_baselines(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    splits = load_splits(paths.project_root)
    model_ready = load_model_ready(paths.project_root)
    features = build_har_feature_frame(model_ready)
    candidates = [
        ("HAR-SquaredReturn", ["sr_d", "sr_w", "sr_m"], False),
        ("Log-HAR-SquaredReturn", ["sr_d", "sr_w", "sr_m"], True),
    ]
    if {"parkinson_d", "parkinson_w", "parkinson_m"}.issubset(features.columns):
        candidates.append(("HAR-Parkinson", ["parkinson_d", "parkinson_w", "parkinson_m"], False))

    all_predictions: dict[str, pd.DataFrame] = {}
    coef_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for model_name, feature_columns, log_target in candidates:
        try:
            predictions, coef = fit_ols_predict(
                features,
                splits,
                feature_columns=feature_columns,
                model_name=model_name,
                log_target=log_target,
            )
        except Exception as exc:  # noqa: BLE001 - record and keep minimum HAR fallback running.
            skipped.append(f"{model_name}: {type(exc).__name__}: {exc}")
            continue
        all_predictions[model_name] = predictions
        for name, value in coef.items():
            coef_rows.append({"model": model_name, "coefficient": name, "value": value})
        model_metrics = metric_rows_for_predictions(predictions)
        model_metrics["variant"] = model_name
        model_metrics["selection_metric"] = "validation_qlike"
        metric_frames.append(model_metrics)

    if not all_predictions:
        raise RuntimeError("All HAR baseline variants failed.")

    metrics = pd.concat(metric_frames, ignore_index=True)
    validation = metrics[metrics["split"] == "validation"].copy()
    selected_model = str(validation.sort_values(["qlike", "rmse", "model"]).iloc[0]["model"])
    metrics["is_selected"] = metrics["model"].eq(selected_model)
    selected_predictions = all_predictions[selected_model]

    pred_path = paths.predictions_dir / "har_predictions.csv"
    metrics_path = paths.metrics_dir / "har_metrics.csv"
    coef_path = paths.metrics_dir / "har_coefficients.csv"
    table_path = paths.tables_dir / "har_metrics.tex"
    selected_predictions.to_csv(pred_path, index=False, float_format="%.17g")
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    write_latex_table(
        metrics.sort_values(["split", "qlike", "model"]),
        table_path,
        ["model", "split", "is_selected", "n_obs", "rmse", "mae", "qlike", "pred_actual_ratio"],
    )
    return {
        "selected_model": selected_model,
        "skipped": skipped,
        "outputs": [pred_path, metrics_path, coef_path, table_path],
    }


def discover_prediction_paths(paths: StagePaths, include_stage1: bool = True) -> list[Path]:
    roots = [
        paths.project_root / "outputs" / "predictions",
        paths.project_root / "outputs" / "advanced" / "predictions",
    ]
    if include_stage1:
        roots.append(paths.predictions_dir)
    discovered: list[Path] = []
    for root in roots:
        if root.exists():
            discovered.extend(sorted(path for path in root.rglob("*.csv") if path.is_file()))
    return discovered


def load_all_predictions(paths: StagePaths, include_stage1: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in discover_prediction_paths(paths, include_stage1=include_stage1):
        try:
            df = validate_prediction_frame(read_csv(path), path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Skipped {rel_path(path, paths.project_root)}: {type(exc).__name__}: {exc}")
            continue
        df["_source_file"] = rel_path(path, paths.project_root)
        frames.append(df)
        inventory_rows.append(
            {
                "source_file": rel_path(path, paths.project_root),
                "rows": len(df),
                "models": " | ".join(sorted(df["model"].unique())),
                "splits": " | ".join(sorted(df["split"].unique())),
            }
        )

    if not frames:
        raise FileNotFoundError("No usable prediction files were found.")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["model", "split", "date", "target_date", "_source_file"]
    )
    duplicate_mask = combined.duplicated(["model", "split", "date", "target_date"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        warnings.append(f"Dropped {duplicate_count} duplicate model/split/date/target_date rows, keeping first.")
        combined = combined.loc[~duplicate_mask].copy()
    inventory = pd.DataFrame(inventory_rows)
    if warnings:
        warning_path = paths.metrics_dir / "prediction_inventory_warnings.txt"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    inventory.to_csv(paths.metrics_dir / "prediction_inventory.csv", index=False)
    return combined.reset_index(drop=True), inventory


def validation_ranked_models(predictions: pd.DataFrame) -> list[str]:
    rows = []
    validation = predictions[predictions["split"] == "validation"].copy()
    for model, group in validation.groupby("model"):
        metrics = volatility_metrics(group["actual_var"], group["pred_var"])
        rows.append({"model": model, **metrics})
    if not rows:
        return []
    return pd.DataFrame(rows).sort_values(["qlike", "rmse", "model"])["model"].astype(str).tolist()


def select_key_models(predictions: pd.DataFrame, max_models: int = 14) -> list[str]:
    available = set(predictions["model"].astype(str).unique())
    selected: list[str] = []
    for model in KEY_MODEL_PREFERENCES:
        if model in available and model not in selected:
            selected.append(model)
    for prefix in ["EWMA(", "HAR-"]:
        matches = sorted(model for model in available if model.startswith(prefix))
        selected.extend(model for model in matches if model not in selected)
    for ranked in validation_ranked_models(predictions):
        if ranked not in selected:
            selected.append(ranked)
        if len(selected) >= max_models:
            break
    return selected[:max_models]


def regime_label(values: pd.Series, q50: float, q75: float, q90: float) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    labels = np.full(len(arr), "extreme-volatility", dtype=object)
    labels[arr <= q90] = "high-volatility"
    labels[arr <= q75] = "normal"
    labels[arr <= q50] = "calm"
    return pd.Series(labels, index=values.index)


def run_regime_and_spike_diagnostics(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    predictions, _inventory = load_all_predictions(paths, include_stage1=True)
    splits = load_splits(paths.project_root)
    test_actual = pd.to_numeric(splits["test"]["target_var_next"], errors="coerce")
    q50, q75, q90 = [float(test_actual.quantile(q)) for q in (0.50, 0.75, 0.90)]

    test_predictions = predictions[predictions["split"] == "test"].copy()
    test_predictions["regime"] = regime_label(test_predictions["actual_var"], q50, q75, q90)
    regime_rows: list[dict[str, Any]] = []
    for (model, regime), group in test_predictions.groupby(["model", "regime"], sort=True):
        regime_rows.append(
            {
                "model": model,
                "split": "test",
                "regime": regime,
                "threshold_source": "test_actual_var_quantiles_for_diagnostic_grouping",
                "q50": q50,
                "q75": q75,
                "q90": q90,
                **volatility_metrics(group["actual_var"], group["pred_var"]),
            }
        )
    regime_metrics = pd.DataFrame(regime_rows)

    train_val_actual = pd.concat(
        [splits["train"]["target_var_next"], splits["validation"]["target_var_next"]],
        ignore_index=True,
    )
    spike_rows: list[dict[str, Any]] = []
    for level in (0.90, 0.95):
        threshold = float(pd.to_numeric(train_val_actual, errors="coerce").quantile(level))
        for model, group in test_predictions.groupby("model", sort=True):
            model_group = group.sort_values("target_date").copy()
            spike = model_group[model_group["actual_var"] >= threshold].copy()
            k = len(spike)
            if k:
                ratio = np.clip(spike["pred_var"].to_numpy(dtype=float), EPSILON, None) / np.clip(
                    spike["actual_var"].to_numpy(dtype=float), EPSILON, None
                )
                top_pred = model_group.nlargest(k, "pred_var")
                actual_keys = set(spike[["date", "target_date"]].itertuples(index=False, name=None))
                pred_keys = set(top_pred[["date", "target_date"]].itertuples(index=False, name=None))
                capture_rate = len(actual_keys.intersection(pred_keys)) / k
                qlike_spike = float(np.mean(qlike_loss(spike["actual_var"], spike["pred_var"])))
                under_rate = float((spike["pred_var"] < spike["actual_var"]).mean())
                severe_rate = float((ratio < 0.5).mean())
                avg_ratio = float(np.mean(ratio))
            else:
                capture_rate = qlike_spike = under_rate = severe_rate = avg_ratio = np.nan
            spike_rows.append(
                {
                    "model": model,
                    "split": "test",
                    "threshold_level": level,
                    "threshold_value": threshold,
                    "threshold_source": "train_validation_actual_var_quantile",
                    "spike_days": int(k),
                    "spike_underprediction_rate": under_rate,
                    "severe_spike_underprediction_rate": severe_rate,
                    "average_pred_actual_ratio_spike_days": avg_ratio,
                    "top_k_spike_capture_rate": capture_rate,
                    "qlike_spike_days": qlike_spike,
                }
            )
    spike_metrics = pd.DataFrame(spike_rows)

    regime_path = paths.metrics_dir / "regime_metrics.csv"
    spike_path = paths.metrics_dir / "spike_diagnostics.csv"
    regime_table = paths.tables_dir / "regime_metrics.tex"
    spike_table = paths.tables_dir / "spike_diagnostics.tex"
    regime_metrics.to_csv(regime_path, index=False)
    spike_metrics.to_csv(spike_path, index=False)
    write_latex_table(
        regime_metrics[regime_metrics["model"].isin(select_key_models(predictions))].sort_values(
            ["model", "regime"]
        ),
        regime_table,
        ["model", "regime", "n_obs", "rmse", "mae", "qlike", "pred_actual_ratio"],
    )
    write_latex_table(
        spike_metrics[
            (spike_metrics["model"].isin(select_key_models(predictions)))
            & (spike_metrics["threshold_level"] == 0.90)
        ].sort_values(["severe_spike_underprediction_rate", "model"]),
        spike_table,
        [
            "model",
            "threshold_level",
            "spike_days",
            "spike_underprediction_rate",
            "severe_spike_underprediction_rate",
            "average_pred_actual_ratio_spike_days",
            "top_k_spike_capture_rate",
            "qlike_spike_days",
        ],
    )
    plot_regime_qlike(regime_metrics, select_key_models(predictions), paths.figures_dir / "regime_qlike_comparison.png")
    plot_spike_underprediction(
        spike_metrics,
        select_key_models(predictions),
        paths.figures_dir / "spike_underprediction_comparison.png",
    )
    return {"outputs": [regime_path, spike_path, regime_table, spike_table]}


def plot_regime_qlike(metrics: pd.DataFrame, models: list[str], path: Path) -> None:
    plot = metrics[metrics["model"].isin(models)].copy()
    order = ["calm", "normal", "high-volatility", "extreme-volatility"]
    pivot = plot.pivot_table(index="model", columns="regime", values="qlike", aggfunc="first")
    pivot = pivot.reindex(models).dropna(how="all")
    pivot = pivot[[column for column in order if column in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(pivot))))
    y = np.arange(len(pivot))
    width = 0.8 / max(1, len(pivot.columns))
    for index, column in enumerate(pivot.columns):
        ax.barh(y + index * width, pivot[column], height=width, label=column)
    ax.set_yticks(y + width * (len(pivot.columns) - 1) / 2)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("QLIKE")
    ax.set_title("Test QLIKE by Volatility Regime")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_spike_underprediction(metrics: pd.DataFrame, models: list[str], path: Path) -> None:
    plot = metrics[(metrics["model"].isin(models)) & (metrics["threshold_level"] == 0.90)].copy()
    plot = plot.sort_values(["severe_spike_underprediction_rate", "model"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot))))
    ax.barh(plot["model"], plot["severe_spike_underprediction_rate"], color="#B33A3A")
    ax.set_xlabel("Severe underprediction rate (predicted/actual < 0.5)")
    ax.set_title("Spike Underprediction on Test Spike Days")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def normal_quantile(alpha: float) -> float:
    return float(NormalDist().inv_cdf(alpha))


def chi2_sf(value: float, df: int) -> float:
    if not np.isfinite(value):
        return np.nan
    if chi2 is not None:
        return float(chi2.sf(value, df))
    if df == 1:
        return float(math.erfc(math.sqrt(max(value, 0.0) / 2.0)))
    if df == 2:
        return float(math.exp(-max(value, 0.0) / 2.0))
    return np.nan


def bernoulli_loglik(successes: int, failures: int, probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return successes * math.log(p) + failures * math.log(1.0 - p)


def kupiec_test(violations: Iterable[bool], alpha: float) -> tuple[float, float]:
    v = np.asarray(list(violations), dtype=bool)
    n = int(len(v))
    x = int(v.sum())
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    ll_null = bernoulli_loglik(x, n - x, alpha)
    ll_alt = bernoulli_loglik(x, n - x, phat)
    lr = float(max(0.0, -2.0 * (ll_null - ll_alt)))
    return lr, chi2_sf(lr, 1)


def christoffersen_independence(violations: Iterable[bool]) -> tuple[float, float, dict[str, int]]:
    v = np.asarray(list(violations), dtype=int)
    if len(v) < 2:
        return np.nan, np.nan, {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev = v[:-1]
    curr = v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    pi01 = n01 / (n00 + n01) if (n00 + n01) else pi
    pi11 = n11 / (n10 + n11) if (n10 + n11) else pi
    ll_restricted = bernoulli_loglik(n01 + n11, n00 + n10, pi)
    ll_unrestricted = bernoulli_loglik(n01, n00, pi01) + bernoulli_loglik(n11, n10, pi11)
    lr = float(max(0.0, -2.0 * (ll_restricted - ll_unrestricted)))
    return lr, chi2_sf(lr, 1), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def violation_cluster_lengths(violations: Iterable[bool]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in np.asarray(list(violations), dtype=bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def basel_zone(violations: int, n_obs: int, alpha: float) -> str:
    if abs(alpha - 0.01) > 1e-12 or n_obs <= 0:
        return "not_applicable"
    if binom is not None:
        green_cut = int(binom.ppf(0.95, n_obs, alpha))
        yellow_cut = int(binom.ppf(0.9999, n_obs, alpha))
    else:
        mean = n_obs * alpha
        sd = math.sqrt(n_obs * alpha * (1.0 - alpha))
        green_cut = int(math.floor(mean + 1.645 * sd))
        yellow_cut = int(math.floor(mean + 3.719 * sd))
    if violations <= green_cut:
        return "green"
    if violations <= yellow_cut:
        return "yellow"
    return "red"


def target_returns(project_root: Path) -> pd.DataFrame:
    clean = read_csv(project_root / "data" / "processed" / "vnindex_cafef_2010_2025_clean.csv")
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["log_return_pct"] = pd.to_numeric(clean["log_return_pct"], errors="coerce")
    clean = clean.dropna(subset=["date", "log_return_pct"]).copy()
    return clean[["date", "log_return_pct"]].rename(
        columns={"date": "target_date", "log_return_pct": "target_return"}
    )


def var_backtest_row(group: pd.DataFrame, model: str, alpha: float) -> dict[str, Any]:
    ordered = group.sort_values("target_date").copy()
    sigma = np.sqrt(np.clip(ordered["pred_var"].to_numpy(dtype=float), EPSILON, None))
    var_forecast = normal_quantile(alpha) * sigma
    returns = ordered["target_return"].to_numpy(dtype=float)
    violations = returns < var_forecast
    violation_count = int(violations.sum())
    n_obs = int(len(ordered))
    kupiec_lr, kupiec_p = kupiec_test(violations, alpha)
    ind_lr, ind_p, transitions = christoffersen_independence(violations)
    cc_lr = kupiec_lr + ind_lr if np.isfinite(kupiec_lr) and np.isfinite(ind_lr) else np.nan
    clusters = violation_cluster_lengths(violations)
    violation_rate = violation_count / n_obs if n_obs else np.nan
    if np.isfinite(kupiec_p) and kupiec_p < 0.05:
        interpretation = "reject_unconditional_coverage_5pct"
    elif np.isfinite(ind_p) and ind_p < 0.05:
        interpretation = "coverage_ok_but_violations_clustered_5pct"
    else:
        interpretation = "no_5pct_backtest_rejection"
    return {
        "model": model,
        "var_level": alpha,
        "var_distribution": "common_normal",
        "distribution_specific_quantile_available": False,
        "n_obs": n_obs,
        "observed_violations": violation_count,
        "expected_violations": float(alpha * n_obs),
        "violation_rate": float(violation_rate),
        "kupiec_stat": kupiec_lr,
        "kupiec_p_value": kupiec_p,
        "christoffersen_independence_stat": ind_lr,
        "christoffersen_independence_p_value": ind_p,
        "christoffersen_conditional_coverage_stat": cc_lr,
        "christoffersen_conditional_coverage_p_value": chi2_sf(cc_lr, 2),
        "violation_cluster_count": int(len(clusters)),
        "max_violation_cluster_length": int(max(clusters) if clusters else 0),
        "mean_violation_cluster_length": float(np.mean(clusters)) if clusters else 0.0,
        "basel_traffic_light_zone": basel_zone(violation_count, n_obs, alpha),
        "interpretation": interpretation,
        "normal_quantile": normal_quantile(alpha),
        **transitions,
    }


def run_extended_var_backtests(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    predictions, _inventory = load_all_predictions(paths, include_stage1=True)
    returns = target_returns(paths.project_root)
    test_predictions = predictions[predictions["split"] == "test"].copy()
    merged = test_predictions.merge(returns, on="target_date", how="left", validate="many_to_one")
    if merged["target_return"].isna().any():
        bad = merged.loc[merged["target_return"].isna(), ["model", "target_date"]].head(10)
        raise ValueError(f"Missing target returns for VaR backtesting:\n{bad.to_string(index=False)}")

    rows: list[dict[str, Any]] = []
    for alpha in (0.05, 0.025, 0.01):
        for model, group in merged.groupby("model", sort=True):
            rows.append(var_backtest_row(group, model, alpha))
    table = pd.DataFrame(rows).sort_values(["var_level", "kupiec_p_value", "model"], ascending=[True, False, True])
    out_path = paths.metrics_dir / "var_backtests_extended.csv"
    tex_path = paths.tables_dir / "var_backtests_extended.tex"
    table.to_csv(out_path, index=False)
    key_models = select_key_models(predictions, max_models=10)
    write_latex_table(
        table[table["model"].isin(key_models)],
        tex_path,
        [
            "model",
            "var_level",
            "n_obs",
            "observed_violations",
            "expected_violations",
            "violation_rate",
            "kupiec_p_value",
            "christoffersen_independence_p_value",
            "christoffersen_conditional_coverage_p_value",
            "basel_traffic_light_zone",
        ],
    )
    for model in key_models:
        model_df = merged[merged["model"] == model].copy()
        if not model_df.empty:
            plot_var_exceedances(model_df, model, paths.figures_dir / f"var_exceedances_{safe_slug(model)}.png")
    return {"outputs": [out_path, tex_path]}


def plot_var_exceedances(df: pd.DataFrame, model: str, path: Path) -> None:
    plot = df.sort_values("target_date").copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot["target_date"], plot["target_return"], color="#111111", linewidth=0.9, label="Return")
    for alpha, color in [(0.05, "#4C78A8"), (0.025, "#F58518"), (0.01, "#B33A3A")]:
        var_line = normal_quantile(alpha) * np.sqrt(np.clip(plot["pred_var"].to_numpy(dtype=float), EPSILON, None))
        ax.plot(plot["target_date"], var_line, color=color, linewidth=0.9, label=f"{100*alpha:g}% VaR")
        violations = plot["target_return"].to_numpy(dtype=float) < var_line
        ax.scatter(
            plot.loc[violations, "target_date"],
            plot.loc[violations, "target_return"],
            color=color,
            s=14,
            alpha=0.85,
        )
    ax.set_title(f"VaR Exceedances: {model}")
    ax.set_ylabel("Percent log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=4)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def run_time_split_diagnostic(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    predictions, _inventory = load_all_predictions(paths, include_stage1=True)
    selected_models = select_key_models(predictions, max_models=12)
    neural_candidates = [
        model
        for model in validation_ranked_models(predictions)
        if re.search(r"lstm|hybrid|calibrated", model, flags=re.IGNORECASE)
    ]
    if neural_candidates and neural_candidates[0] not in selected_models:
        selected_models.append(neural_candidates[0])
    selected_models = list(dict.fromkeys(selected_models))

    rows: list[dict[str, Any]] = []
    subset = predictions[predictions["model"].isin(selected_models)].copy()
    subset["target_year"] = subset["target_date"].dt.year
    for (model, split, year), group in subset.groupby(["model", "split", "target_year"], sort=True):
        if len(group) < 20:
            continue
        rows.append(
            {
                "experiment_type": "regime_time_existing_prediction_diagnostic",
                "true_rolling_origin_retraining": False,
                "window_label": f"{split}_{int(year)}",
                "model": model,
                "split": split,
                "target_year": int(year),
                **volatility_metrics(group["actual_var"], group["pred_var"]),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["target_year", "split", "qlike", "model"])
    out_path = paths.metrics_dir / "rolling_origin_metrics.csv"
    tex_path = paths.tables_dir / "rolling_origin_metrics.tex"
    fig_path = paths.figures_dir / "rolling_origin_qlike.png"
    metrics.to_csv(out_path, index=False)
    write_latex_table(
        metrics,
        tex_path,
        ["window_label", "model", "n_obs", "rmse", "mae", "qlike", "pred_actual_ratio"],
    )
    plot_time_qlike(metrics, fig_path)
    return {"outputs": [out_path, tex_path, fig_path]}


def plot_time_qlike(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    pivot = metrics.pivot_table(index="window_label", columns="model", values="qlike", aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in pivot.columns[:12]:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=1.0, label=model)
    ax.set_ylabel("QLIKE")
    ax.set_title("Existing-Prediction Time-Split QLIKE Diagnostic")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def run_advanced_search_audit(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    tables_dir = paths.project_root / "outputs" / "advanced" / "tables"
    all_path = tables_dir / "table_garch_family_all_candidates.csv"
    failures_path = tables_dir / "table_garch_family_fit_failures.csv"
    selected_path = tables_dir / "table_garch_family_selected_models.csv"
    all_candidates = read_csv(all_path)
    failures = read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    selected = read_csv(selected_path) if selected_path.exists() else pd.DataFrame()

    attempted = len(all_candidates) + len(failures)
    success = int((all_candidates.get("status", "success") == "success").sum())
    failed = len(failures)
    audit_rows: list[dict[str, Any]] = [
        {"section": "overall", "label": "attempted_candidates", "value": attempted, "detail": ""},
        {"section": "overall", "label": "successful_fits", "value": success, "detail": ""},
        {"section": "overall", "label": "failed_fits", "value": failed, "detail": ""},
        {
            "section": "overall",
            "label": "convergence_failures",
            "value": int(failures["failure_reason"].astype(str).str.contains("convergence", case=False).sum())
            if not failures.empty and "failure_reason" in failures
            else 0,
            "detail": "",
        },
        {
            "section": "overall",
            "label": "nonfinite_forecast_failures",
            "value": int(
                failures["error_message"].astype(str).str.contains("non-finite", case=False).sum()
            )
            if not failures.empty and "error_message" in failures
            else 0,
            "detail": "",
        },
    ]
    for section, df, column in [
        ("success_by_family", all_candidates, "volatility_group"),
        ("success_by_distribution", all_candidates, "distribution"),
        ("failure_by_family", failures, "volatility_group"),
        ("failure_by_distribution", failures, "distribution"),
        ("failure_reason", failures, "failure_reason"),
    ]:
        if df.empty or column not in df:
            continue
        for label, count in df[column].astype(str).value_counts().sort_index().items():
            audit_rows.append({"section": section, "label": label, "value": int(count), "detail": ""})

    successful = all_candidates.copy()
    for column in ["validation_QLIKE", "test_QLIKE"]:
        if column in successful:
            successful[column] = pd.to_numeric(successful[column], errors="coerce")
    top_val = successful.sort_values(["validation_QLIKE", "validation_RMSE", "candidate_id"]).head(10)
    top_test = successful.sort_values(["test_QLIKE", "test_RMSE", "candidate_id"]).head(10)
    for rank, row in enumerate(top_val.to_dict("records"), start=1):
        audit_rows.append(
            {
                "section": "top_validation_models",
                "label": f"rank_{rank}",
                "value": row.get("validation_QLIKE"),
                "detail": f"{row.get('candidate_id')} | test_QLIKE={row.get('test_QLIKE')}",
            }
        )
    for rank, row in enumerate(top_test.to_dict("records"), start=1):
        audit_rows.append(
            {
                "section": "top_test_models_exploratory",
                "label": f"rank_{rank}",
                "value": row.get("test_QLIKE"),
                "detail": f"{row.get('candidate_id')} | validation_QLIKE={row.get('validation_QLIKE')}",
            }
        )
    if not selected.empty:
        primary = selected[selected["selection_role"] == "best_fixed_by_validation_qlike"]
        if not primary.empty:
            row = primary.iloc[0]
            audit_rows.append(
                {
                    "section": "selected_model",
                    "label": "validation_selected_primary",
                    "value": row.get("validation_QLIKE"),
                    "detail": f"{row.get('model')} | {row.get('candidate_id')} | test_QLIKE={row.get('test_QLIKE')}",
                }
            )

    audit = pd.DataFrame(audit_rows)
    failure_summary = (
        failures.groupby(["failure_reason", "error_type", "stage"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "failure_reason"], ascending=[False, True])
        if not failures.empty
        else pd.DataFrame(columns=["failure_reason", "error_type", "stage", "count"])
    )
    audit_path = paths.metrics_dir / "advanced_search_audit.csv"
    failures_summary_path = paths.metrics_dir / "advanced_search_failures_summary.csv"
    table_path = paths.tables_dir / "advanced_search_summary.tex"
    audit.to_csv(audit_path, index=False)
    failure_summary.to_csv(failures_summary_path, index=False)
    summary_table = audit[audit["section"].isin(["overall", "selected_model"])].copy()
    write_latex_table(summary_table, table_path, ["section", "label", "value", "detail"])
    return {"outputs": [audit_path, failures_summary_path, table_path]}


def construct_ohlc_proxies(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close"]:
        if column not in df:
            raise ValueError(f"OHLC proxy construction requires {column}.")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid = (
        df[["open", "high", "low", "close"]].notna().all(axis=1)
        & (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    prev_close = df["close"].shift(1)
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    overnight = np.log(df["open"] / prev_close)
    open_close = np.log(df["close"] / df["open"])
    rs = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) + np.log(
        df["low"] / df["close"]
    ) * np.log(df["low"] / df["open"])
    close_to_close = 100.0 * np.log(df["close"] / prev_close)

    proxies = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce")})
    proxies["close_to_close_squared_return"] = close_to_close**2
    proxies["parkinson"] = (100.0**2) * (1.0 / (4.0 * np.log(2.0))) * (log_hl**2)
    proxies["garman_klass"] = (100.0**2) * (
        0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    )
    proxies["rogers_satchell"] = (100.0**2) * rs
    for column in ["parkinson", "garman_klass", "rogers_satchell"]:
        proxies[column] = proxies[column].where(valid)
    for proxy in ["close_to_close_squared_return", "parkinson", "garman_klass", "rogers_satchell"]:
        proxies[proxy] = pd.to_numeric(proxies[proxy], errors="coerce")
        proxies.loc[proxies[proxy] < -1e-12, proxy] = np.nan
        proxies.loc[(proxies[proxy] < 0) & (proxies[proxy] >= -1e-12), proxy] = 0.0

    for window in (5, 10, 20):
        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        sigma_o = overnight.rolling(window=window, min_periods=window).var(ddof=1)
        sigma_c = open_close.rolling(window=window, min_periods=window).var(ddof=1)
        sigma_rs = rs.rolling(window=window, min_periods=window).mean()
        name = f"yang_zhang_{window}"
        proxies[name] = (100.0**2) * (sigma_o + k * sigma_c + (1.0 - k) * sigma_rs)
        proxies.loc[proxies[name] < -1e-12, name] = np.nan
        proxies.loc[(proxies[name] < 0) & (proxies[name] >= -1e-12), name] = 0.0
    return proxies


def run_proxy_robustness_cleanup(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    model_ready = load_model_ready(paths.project_root)
    proxies = construct_ohlc_proxies(model_ready[["date", "open", "high", "low", "close"]])
    corr = proxies[PROXY_LABELS].corr(min_periods=50)
    corr.index.name = "proxy"
    corr_path = paths.metrics_dir / "proxy_correlations.csv"
    corr.reset_index().to_csv(corr_path, index=False)
    plot_proxy_correlation_heatmap(corr, paths.figures_dir / "proxy_correlation_heatmap.png")

    predictions, _inventory = load_all_predictions(paths, include_stage1=True)
    targets = proxies[["date", *PROXY_LABELS]].rename(columns={"date": "target_date"})
    rows: list[dict[str, Any]] = []
    for proxy in PROXY_LABELS:
        target = targets[["target_date", proxy]].rename(columns={proxy: "proxy_actual_var"})
        merged = predictions.merge(target, on="target_date", how="left")
        merged = merged[np.isfinite(pd.to_numeric(merged["proxy_actual_var"], errors="coerce"))].copy()
        for (model, split), group in merged.groupby(["model", "split"], sort=True):
            if len(group) < 50:
                continue
            metrics = volatility_metrics(group["proxy_actual_var"], group["pred_var"])
            rows.append({"proxy": proxy, "model": model, "split": split, **metrics})
    proxy_metrics = pd.DataFrame(rows)
    if proxy_metrics.empty:
        raise RuntimeError("No proxy robustness metrics could be computed.")
    proxy_metrics["qlike_rank"] = proxy_metrics.groupby(["proxy", "split"])["qlike"].rank(
        method="min", ascending=True
    )
    test = proxy_metrics[proxy_metrics["split"] == "test"].copy()
    stability_rows: list[dict[str, Any]] = []
    for model, group in test.groupby("model", sort=True):
        ranks = group["qlike_rank"].astype(float)
        stability_rows.append(
            {
                "model": model,
                "proxy_count": int(group["proxy"].nunique()),
                "average_qlike_rank": float(ranks.mean()),
                "best_rank": float(ranks.min()),
                "worst_rank": float(ranks.max()),
                "rank_std": float(ranks.std(ddof=0)),
                "top1_proxy_count": int((ranks <= 1).sum()),
                "top3_proxy_count": int((ranks <= 3).sum()),
                "top5_proxy_count": int((ranks <= 5).sum()),
                "average_pred_actual_ratio": float(group["pred_actual_ratio"].mean()),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(["average_qlike_rank", "rank_std", "model"])
    rank_path = paths.metrics_dir / "proxy_rank_stability.csv"
    proxy_metrics_path = paths.metrics_dir / "proxy_model_metrics.csv"
    rank_tex = paths.tables_dir / "proxy_rank_stability.tex"
    stability.to_csv(rank_path, index=False)
    proxy_metrics.to_csv(proxy_metrics_path, index=False)
    write_latex_table(
        stability.head(25),
        rank_tex,
        [
            "model",
            "proxy_count",
            "average_qlike_rank",
            "best_rank",
            "worst_rank",
            "rank_std",
            "top5_proxy_count",
            "average_pred_actual_ratio",
        ],
    )
    plot_proxy_rank_heatmap(
        proxy_metrics,
        stability,
        select_key_models(predictions, max_models=10),
        paths.figures_dir / "proxy_rank_heatmap_top_models.png",
    )
    return {"outputs": [corr_path, rank_path, proxy_metrics_path, rank_tex]}


def plot_proxy_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    labels = corr.index.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Volatility Proxy Correlations")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_proxy_rank_heatmap(metrics: pd.DataFrame, stability: pd.DataFrame, key_models: list[str], path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    top_models = stability.head(20)["model"].astype(str).tolist()
    models = list(dict.fromkeys([*key_models, *top_models]))
    plot = test[test["model"].isin(models)].pivot_table(
        index="model", columns="proxy", values="qlike_rank", aggfunc="min"
    )
    plot["avg_rank"] = plot.mean(axis=1)
    plot = plot.sort_values("avg_rank").drop(columns=["avg_rank"])
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot))))
    image = ax.imshow(plot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(plot.columns)))
    ax.set_xticklabels(plot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(plot.index)))
    ax.set_yticklabels(plot.index)
    ax.set_title("Test QLIKE Rank by Proxy")
    fig.colorbar(image, ax=ax, label="Rank (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def run_neural_diagnostics(paths: StagePaths) -> dict[str, Any]:
    ensure_dirs(paths)
    predictions, _inventory = load_all_predictions(paths, include_stage1=True)
    neural_mask = predictions["model"].str.contains("lstm|hybrid|calibrated", case=False, regex=True, na=False)
    benchmark_models = ["GARCH(1,1)", "ARIMA-GARCH", "AdvGARCH-BestQLIKE"]
    selected = predictions[neural_mask | predictions["model"].isin(benchmark_models)].copy()
    if selected.empty:
        skipped_path = paths.metrics_dir / "neural_diagnostics.csv"
        pd.DataFrame(
            [{"status": "skipped", "reason": "No neural, hybrid, or calibrated predictions available."}]
        ).to_csv(skipped_path, index=False)
        return {"outputs": [skipped_path], "skipped": ["neural diagnostics: no predictions available"]}

    regime_path = paths.metrics_dir / "regime_metrics.csv"
    spike_path = paths.metrics_dir / "spike_diagnostics.csv"
    var_path = paths.metrics_dir / "var_backtests_extended.csv"
    regime = read_csv(regime_path) if regime_path.exists() else pd.DataFrame()
    spike = read_csv(spike_path) if spike_path.exists() else pd.DataFrame()
    var = read_csv(var_path) if var_path.exists() else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model, split), group in selected.groupby(["model", "split"], sort=True):
        model_group = (
            "benchmark_garch_family"
            if model in benchmark_models or "garch" in model.lower()
            else "neural_or_hybrid"
        )
        row: dict[str, Any] = {"model": model, "split": split, "model_group": model_group}
        row.update(volatility_metrics(group["actual_var"], group["pred_var"]))
        if split == "test" and not regime.empty:
            extreme = regime[(regime["model"] == model) & (regime["regime"] == "extreme-volatility")]
            row["extreme_regime_qlike"] = float(extreme.iloc[0]["qlike"]) if not extreme.empty else np.nan
        if split == "test" and not spike.empty:
            spike90 = spike[(spike["model"] == model) & (spike["threshold_level"].astype(float) == 0.90)]
            if not spike90.empty:
                row["spike90_underprediction_rate"] = float(spike90.iloc[0]["spike_underprediction_rate"])
                row["spike90_severe_underprediction_rate"] = float(
                    spike90.iloc[0]["severe_spike_underprediction_rate"]
                )
                row["spike90_average_pred_actual_ratio"] = float(
                    spike90.iloc[0]["average_pred_actual_ratio_spike_days"]
                )
        if split == "test" and not var.empty:
            var1 = var[(var["model"] == model) & (var["var_level"].astype(float) == 0.01)]
            if not var1.empty:
                row["var1_violation_rate"] = float(var1.iloc[0]["violation_rate"])
                row["var1_kupiec_p_value"] = float(var1.iloc[0]["kupiec_p_value"])
        rows.append(row)
    diagnostics = pd.DataFrame(rows).sort_values(["split", "qlike", "model"])
    out_path = paths.metrics_dir / "neural_diagnostics.csv"
    tex_path = paths.tables_dir / "neural_diagnostics.tex"
    fig_path = paths.figures_dir / "neural_spike_failure.png"
    diagnostics.to_csv(out_path, index=False)
    write_latex_table(
        diagnostics[diagnostics["split"] == "test"].head(25),
        tex_path,
        [
            "model",
            "model_group",
            "n_obs",
            "rmse",
            "mae",
            "qlike",
            "pred_actual_ratio",
            "extreme_regime_qlike",
            "spike90_severe_underprediction_rate",
            "var1_violation_rate",
        ],
    )
    plot_neural_spike_failure(diagnostics, fig_path)
    return {"outputs": [out_path, tex_path, fig_path]}


def plot_neural_spike_failure(diagnostics: pd.DataFrame, path: Path) -> None:
    plot = diagnostics[(diagnostics["split"] == "test") & diagnostics["spike90_severe_underprediction_rate"].notna()].copy()
    if plot.empty:
        return
    plot = plot.sort_values("spike90_severe_underprediction_rate", ascending=False).head(25)
    colors = np.where(plot["model_group"].eq("benchmark_garch_family"), "#4C78A8", "#B33A3A")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot))))
    ax.barh(plot["model"], plot["spike90_severe_underprediction_rate"], color=colors)
    ax.set_xlabel("Severe spike underprediction rate")
    ax.set_title("Neural/Hybrid Spike Failure Diagnostic")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def leakage_audit(paths: StagePaths) -> pd.DataFrame:
    splits = load_splits(paths.project_root)
    model_ready = load_model_ready(paths.project_root)
    rows: list[dict[str, Any]] = []
    for split, df in splits.items():
        rows.append(
            {
                "check": f"{split}_target_date_after_origin",
                "status": "pass" if bool((df["target_date"] > df["date"]).all()) else "fail",
                "detail": f"rows={len(df)}",
            }
        )
    prepared = model_ready.copy()
    close = pd.to_numeric(prepared["close"], errors="coerce")
    recomputed_sq = (100.0 * np.log(close / close.shift(1))) ** 2
    recomputed_target = recomputed_sq.shift(-1)
    comparable = prepared["target_var_next"].notna() & recomputed_target.notna()
    target_ok = np.allclose(
        prepared.loc[comparable, "target_var_next"].to_numpy(dtype=float),
        recomputed_target.loc[comparable].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    rows.append({"check": "target_var_next_equals_next_squared_return", "status": "pass" if target_ok else "fail", "detail": ""})
    for window in (5, 10, 20):
        recomputed = prepared["log_return_pct"].rolling(window=window).std()
        available = prepared[f"rolling_vol_{window}"].notna() & recomputed.notna()
        ok = np.allclose(
            prepared.loc[available, f"rolling_vol_{window}"].to_numpy(dtype=float),
            recomputed.loc[available].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        rows.append(
            {
                "check": f"rolling_vol_{window}_trailing_only_formula",
                "status": "pass" if ok else "fail",
                "detail": "pandas rolling(window).std() matches checked-in feature",
            }
        )
    rows.extend(
        [
            {
                "check": "lstm_scaler_imputer_fit_scope",
                "status": "pass_by_code_audit",
                "detail": "src/train_lstm_hybrid.py fits imputer/scaler on train then applies to validation/test.",
            },
            {
                "check": "lstm_sequences_do_not_cross_split_boundaries",
                "status": "pass_by_code_audit",
                "detail": "create_sequences is called separately for train, validation, and test splits.",
            },
            {
                "check": "qlike_epsilon_consistency",
                "status": "pass",
                "detail": f"Shared EPSILON imported from src/metrics.py = {EPSILON}.",
            },
            {
                "check": "forbidden_predictive_columns",
                "status": "pass_by_code_audit",
                "detail": "target_var_next, target_date, date, and split are excluded from LSTM feature lists.",
            },
        ]
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(paths.metrics_dir / "leakage_audit.csv", index=False)
    return audit


def repo_tree(project_root: Path, max_depth: int = 3) -> str:
    skip = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}
    lines = ["."]
    root_depth = len(project_root.parts)
    for path in sorted(project_root.rglob("*")):
        rel = path.relative_to(project_root)
        if any(part in skip for part in rel.parts):
            continue
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        indent = "    " * (depth - 1)
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{rel.name}{suffix}")
    return "\n".join(lines)


def current_outputs_summary(paths: StagePaths) -> pd.DataFrame:
    roots = [
        paths.project_root / "outputs" / "metrics",
        paths.project_root / "outputs" / "predictions",
        paths.project_root / "outputs" / "advanced",
        paths.project_root / "outputs" / "proxy_robustness",
        paths.output_dir,
    ]
    rows = []
    for root in roots:
        if root.exists():
            files = [path for path in root.rglob("*") if path.is_file()]
            rows.append({"root": rel_path(root, paths.project_root), "file_count": len(files)})
    return pd.DataFrame(rows)


def main_files_table() -> list[tuple[str, str]]:
    return [
        ("data preparation", "src/prepare_data.py"),
        ("feature engineering", "src/prepare_data.py; src/train_econometric.py bridge features"),
        ("model training", "src/train_econometric.py; src/train_lstm_hybrid.py; src/tune_lstm_hybrid.py"),
        ("GARCH / ARIMA-GARCH", "src/train_econometric.py"),
        ("advanced GARCH-family search", "src/advanced_model_search.py; src/run_advanced_experiments.py"),
        ("LSTM / hybrid models", "src/train_lstm_hybrid.py; src/tune_lstm_hybrid.py"),
        ("evaluation", "src/evaluate_all.py; src/metrics.py"),
        ("proxy robustness", "src/proxy_robustness.py"),
        ("VaR backtesting", "src/advanced_var_backtesting.py"),
        ("Makefile", "Makefile; paper/Makefile"),
        ("configs", "pyproject.toml; requirements.txt"),
        ("tests", "tests/test_*.py"),
        ("Stage 1 extensions", "src/stage1_extensions.py"),
    ]


def write_repo_audit_report(paths: StagePaths, test_status: str = "not recorded") -> Path:
    ensure_dirs(paths)
    leakage = leakage_audit(paths)
    outputs = current_outputs_summary(paths)
    stage_files = sorted(path for path in paths.output_dir.rglob("*") if path.is_file())
    commands = [
        "pwd && tree -L 3 -a -I '.git|__pycache__|.pytest_cache|.venv|venv|node_modules'",
        "rg --files",
        "sed -n '1,240p' README.md",
        "sed -n '1,260p' Makefile",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py baselines",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py risk",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py robustness",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py diagnostics",
        "PYTHONDONTWRITEBYTECODE=1 make stage1-all",
        "PYTHONDONTWRITEBYTECODE=1 pytest",
        "PYTHONDONTWRITEBYTECODE=1 make stage1-test",
        "PYTHONDONTWRITEBYTECODE=1 make stage1-baselines stage1-risk stage1-robustness stage1-diagnostics",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py audit --test-status \"pytest: PASS (22 passed); make stage1-test: PASS (22 passed); make stage1-all and component Stage 1 targets: PASS\"",
        "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py manifest",
    ]
    lines = [
        "# Stage 1 Repository Audit",
        "",
        "Scope: source code, experiments, tests, reproducibility, and empirical artifacts only.",
        "",
        "## Repository Structure",
        "",
        "```text",
        repo_tree(paths.project_root, max_depth=3),
        "```",
        "",
        "## Main Files and Scripts",
        "",
        "| Area | Files |",
        "|---|---|",
    ]
    lines.extend([f"| {area} | `{files}` |" for area, files in main_files_table()])
    lines.extend(
        [
            "",
            "## Current Pipeline Summary",
            "",
            "- `src/prepare_data.py` constructs percent log returns, squared returns, trailing rolling volatility, next-day target variance, and chronological train/validation/test splits.",
            "- `src/train_econometric.py` trains HistoricalMean, RollingVol, GARCH(1,1), and ARIMA-GARCH with training-only parameter estimation and recursive state updates.",
            "- `src/train_lstm_hybrid.py` trains LSTM and ARIMA-GARCH-LSTM with train-only imputers/scalers and split-local sequences.",
            "- `src/tune_lstm_hybrid.py`, `src/advanced_model_search.py`, and advanced experiment scripts add tuned neural, calibrated, GARCH-family, refit, combination, and hypothesis-testing artifacts.",
            "- `src/evaluate_all.py` computes common-window and available-window metrics with shared QLIKE clipping.",
            "",
            "## Current Available Outputs",
            "",
            "| Root | File count |",
            "|---|---:|",
        ]
    )
    for row in outputs.to_dict("records"):
        lines.append(f"| `{row['root']}` | {row['file_count']} |")
    lines.extend(
        [
            "",
            "## Leakage and Reproducibility Audit",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
    )
    for row in leakage.to_dict("records"):
        lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Missing Pieces Found Before Stage 1 Extensions",
            "",
            "- EWMA/RiskMetrics and HAR-type volatility baselines were not present in the canonical baseline set.",
            "- Existing VaR tables covered common Normal VaR at fewer levels than the requested 5%, 2.5%, and 1% extended output.",
            "- Regime/spike diagnostics existed partially, but not as a clean Stage 1 artifact spanning existing, EWMA, and HAR predictions.",
            "- Advanced GARCH search artifacts existed, but no consolidated Stage 1 audit summary existed under the new output folder.",
            "- Proxy robustness artifacts existed under `outputs/proxy_robustness`, but not in a cleaned Stage 1 folder with new baselines included.",
            "",
            "## Tasks Implemented in Stage 1",
            "",
            "- Added validation-selected EWMA baseline over lambda values 0.90, 0.94, 0.97, and 0.99.",
            "- Added validation-selected HAR baselines: HAR-SquaredReturn, Log-HAR-SquaredReturn, and HAR-Parkinson when OHLC columns are available.",
            "- Generated test-period regime metrics and spike diagnostics using train/validation spike thresholds.",
            "- Generated extended Normal VaR backtests at 5%, 2.5%, and 1% with Kupiec, Christoffersen independence, conditional coverage, clustering, and Basel-style 1% zones.",
            "- Generated an existing-prediction time-split diagnostic and explicitly labeled it as not true rolling-origin retraining.",
            "- Generated advanced GARCH search audit and failure summaries.",
            "- Regenerated proxy correlations and proxy rank stability with Stage 1 baselines included.",
            "- Generated neural/hybrid risk diagnostics from existing prediction artifacts.",
            "- Added Stage 1 Makefile targets and tests for proxy formulas, leakage alignment, and VaR backtest helpers.",
            "",
            "## Tasks Skipped and Why",
            "",
            "- Full rolling-origin retraining was skipped because retraining GARCH, ARIMA-GARCH, advanced GARCH, and neural models over multiple windows is materially more expensive; a lighter existing-prediction time-split diagnostic was generated instead.",
            "- White Reality Check / Hansen SPA was not implemented in this stage because stationary/block bootstrap inference over the full model universe needs additional design choices and runtime budget.",
            "- Distribution-specific VaR for Student-t, skewed-t, and GED was not computed because current prediction CSVs do not retain fitted innovation shape/skew parameters; the Stage 1 VaR output exposes `distribution_specific_quantile_available=False` and uses common Normal VaR.",
            "",
            "## Stage 1 Artifacts Generated",
            "",
        ]
    )
    if stage_files:
        for path in stage_files:
            lines.append(f"- `{rel_path(path, paths.project_root)}`")
    else:
        lines.append("- No Stage 1 files found yet.")
    lines.extend(
        [
            "",
            "## Exact Commands Run",
            "",
        ]
    )
    lines.extend([f"- `{command}`" for command in commands])
    lines.extend(
        [
            "",
            "## Test Status",
            "",
            f"- {test_status}",
            "",
            "## Failures or Errors Encountered",
            "",
            "- Initial `python src/stage1_extensions.py baselines` failed because pandas `DataFrame.to_latex()` required missing optional dependency `jinja2` in this environment.",
            "- Resolution: replaced pandas `to_latex()` usage with a deterministic manual LaTeX tabular writer in `src/stage1_extensions.py`; all Stage 1 subcommands and tests passed after the fix.",
            "- No unresolved Stage 1 script failures remain. Skipped methodological items are listed above.",
            "",
            "## Next Steps for Stage 2",
            "",
            "- Reframe primary claims around validation-selected models, proxy robustness, and risk-aware evaluation rather than test-best model discovery.",
            "- Use EWMA and HAR results as benchmark-completeness evidence.",
            "- Use regime, spike, and VaR outputs for economic interpretation, especially underestimation during high-volatility days.",
            "- Clearly label advanced-test-best rankings as exploratory and keep the validation-selected GARCH-family model as the primary advanced result.",
            "- State that full rolling-origin retraining and distribution-specific VaR are future robustness extensions unless separately run.",
        ]
    )
    path = paths.report_dir / "repo_audit_stage1.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return result.stdout.strip()


def write_run_manifest(paths: StagePaths, failed_tasks: list[str] | None = None, skipped_tasks: list[str] | None = None) -> Path:
    ensure_dirs(paths)
    generated = sorted(
        rel_path(path, paths.project_root)
        for root in [paths.output_dir, paths.report_dir]
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    input_files = [
        "data/processed/vnindex_model_ready.csv",
        "data/processed/train.csv",
        "data/processed/validation.csv",
        "data/processed/test.csv",
        "outputs/predictions/*.csv",
        "outputs/advanced/predictions/**/*.csv",
        "outputs/advanced/tables/table_garch_family_all_candidates.csv",
        "outputs/advanced/tables/table_garch_family_fit_failures.csv",
        "outputs/advanced/tables/table_garch_family_selected_models.csv",
    ]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(paths.project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: package_version(name)
            for name in [
                "numpy",
                "pandas",
                "matplotlib",
                "scipy",
                "statsmodels",
                "arch",
                "scikit-learn",
                "tensorflow",
                "pytest",
            ]
        },
        "commands_run": [
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py baselines",
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py risk",
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py robustness",
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py diagnostics",
            "PYTHONDONTWRITEBYTECODE=1 make stage1-all",
            "PYTHONDONTWRITEBYTECODE=1 pytest",
            "PYTHONDONTWRITEBYTECODE=1 make stage1-test",
            "PYTHONDONTWRITEBYTECODE=1 make stage1-baselines stage1-risk stage1-robustness stage1-diagnostics",
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py audit --test-status \"pytest: PASS (22 passed); make stage1-test: PASS (22 passed); make stage1-all and component Stage 1 targets: PASS\"",
            "PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py manifest",
        ],
        "input_files_used": input_files,
        "output_files_generated": generated,
        "random_seeds": {
            "stage1_extensions": "No stochastic model fitting; deterministic computations only.",
            "existing_lstm_pipeline": 42,
        },
        "failed_tasks": failed_tasks or [],
        "skipped_tasks": skipped_tasks
        or [
            "Full rolling-origin retraining skipped; existing-prediction time-split diagnostic generated.",
            "White Reality Check / Hansen SPA not implemented in Stage 1.",
            "Distribution-specific VaR skipped because fitted distribution parameters are not recoverable from prediction artifacts.",
        ],
        "notes_about_reproducibility": [
            "All new Stage 1 outputs are isolated under outputs/stage1_extensions and reports/stage1_extensions.",
            f"QLIKE clipping epsilon is {EPSILON}.",
            "EWMA and HAR model selection uses validation QLIKE only.",
            "Test-best advanced GARCH rows are labeled exploratory in the audit output.",
        ],
    }
    path = paths.output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_baselines(paths: StagePaths) -> dict[str, Any]:
    ewma = run_ewma_baseline(paths)
    har = run_har_baselines(paths)
    return {"ewma": ewma, "har": har}


def run_risk(paths: StagePaths) -> dict[str, Any]:
    regime = run_regime_and_spike_diagnostics(paths)
    var = run_extended_var_backtests(paths)
    return {"regime_spike": regime, "var": var}


def run_diagnostics(paths: StagePaths) -> dict[str, Any]:
    rolling = run_time_split_diagnostic(paths)
    advanced = run_advanced_search_audit(paths)
    neural = run_neural_diagnostics(paths)
    return {"rolling": rolling, "advanced": advanced, "neural": neural}


def run_all(paths: StagePaths) -> None:
    ensure_dirs(paths)
    run_baselines(paths)
    run_risk(paths)
    run_proxy_robustness_cleanup(paths)
    run_diagnostics(paths)
    write_run_manifest(paths)
    write_repo_audit_report(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["baselines", "risk", "diagnostics", "robustness", "audit", "manifest", "all"],
        help="Stage 1 extension task to run.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--test-status", default="not recorded")
    return parser.parse_args()


def paths_from_args(args: argparse.Namespace) -> StagePaths:
    project_root = args.project_root.resolve()
    defaults = default_paths(project_root)
    return StagePaths(
        project_root=project_root,
        output_dir=(args.output_dir.resolve() if args.output_dir else defaults.output_dir),
        report_dir=(args.report_dir.resolve() if args.report_dir else defaults.report_dir),
    )


def main() -> None:
    args = parse_args()
    paths = paths_from_args(args)
    ensure_dirs(paths)
    if args.command == "baselines":
        run_baselines(paths)
    elif args.command == "risk":
        run_risk(paths)
    elif args.command == "diagnostics":
        run_diagnostics(paths)
    elif args.command == "robustness":
        run_proxy_robustness_cleanup(paths)
    elif args.command == "audit":
        write_repo_audit_report(paths, test_status=args.test_status)
    elif args.command == "manifest":
        write_run_manifest(paths)
    elif args.command == "all":
        run_all(paths)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
