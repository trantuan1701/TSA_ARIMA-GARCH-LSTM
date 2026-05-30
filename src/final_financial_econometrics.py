#!/usr/bin/env python3
"""Final audit and evidence tables for the VN-Index volatility study.

This script is intentionally conservative. It reads existing predictions and
experiment artifacts, rebuilds common-window evidence from model/date keys, and
writes final course-project outputs under ``outputs/final`` plus paper-ready
tables/figures. It does not treat any test-best model as primary evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2, jarque_bera, kurtosis, skew
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover
    from .metrics import EPSILON

try:
    from stage1_extensions import (
        build_har_feature_frame,
        fit_ols_predict,
        load_model_ready as load_stage1_model_ready,
        load_splits as load_stage1_splits,
    )
except ImportError:  # pragma: no cover
    from .stage1_extensions import (
        build_har_feature_frame,
        fit_ols_predict,
        load_model_ready as load_stage1_model_ready,
        load_splits as load_stage1_splits,
    )

try:
    from proxy_robustness import construct_proxies, parse_price_source
except ImportError:  # pragma: no cover
    from .proxy_robustness import construct_proxies, parse_price_source

try:
    from advanced_model_search import (
        CandidateSpec,
        forecast_on_context,
        original_arima_garch_spec,
        original_garch_spec,
        parse_candidate_from_row,
    )
except ImportError:  # pragma: no cover
    from .advanced_model_search import (
        CandidateSpec,
        forecast_on_context,
        original_arima_garch_spec,
        original_garch_spec,
        parse_candidate_from_row,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FINAL_DIR = PROJECT_ROOT / "outputs" / "final"
OUTPUT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
DATE_COLUMNS = ["date", "target_date"]
TARGET_PROXY = "next_day_squared_percentage_log_return"

PRIMARY_STATIC_SPECS = [
    (
        "HistoricalMean",
        "fixed_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_baseline_mean.csv",
    ),
    (
        "RollingVol-5",
        "fixed_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_rolling_vol_5.csv",
    ),
    (
        "RollingVol-10",
        "fixed_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_rolling_vol_10.csv",
    ),
    (
        "RollingVol-20",
        "fixed_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_rolling_vol_20.csv",
    ),
    (
        "GARCH(1,1)",
        "training_selected_econometric_benchmark",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_garch_11.csv",
    ),
    (
        "ARIMA-GARCH",
        "training_aic_mean_model_plus_garch_benchmark",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_arima_garch.csv",
    ),
    (
        "AdvGARCH-BestQLIKE",
        "validation_selected_primary_advanced_garch",
        PROJECT_ROOT
        / "outputs"
        / "advanced"
        / "predictions"
        / "garch_family"
        / "pred_advgarch_bestqlike.csv",
    ),
    (
        "AdvGARCH-BestAsymmetric",
        "validation_selected_asymmetric_garch_comparator",
        PROJECT_ROOT
        / "outputs"
        / "advanced"
        / "predictions"
        / "garch_family"
        / "pred_advgarch_bestasymmetric.csv",
    ),
    (
        "LSTM",
        "validation_monitored_neural_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_lstm_base.csv",
    ),
    (
        "ARIMA-GARCH-LSTM",
        "validation_monitored_hybrid_neural_baseline",
        PROJECT_ROOT / "outputs" / "predictions" / "pred_lstm_hybrid.csv",
    ),
    (
        "Hybrid-QLIKE",
        "validation_selected_tuned_hybrid_neural",
        PROJECT_ROOT / "outputs" / "predictions" / "lstm_tuned" / "pred_hybrid_qlike.csv",
    ),
    (
        "EWMA(lambda=0.90)",
        "validation_selected_ewma",
        PROJECT_ROOT / "outputs" / "stage1_extensions" / "predictions" / "ewma_predictions.csv",
    ),
    (
        "HAR-Parkinson",
        "validation_selected_har",
        PROJECT_ROOT / "outputs" / "stage1_extensions" / "predictions" / "har_predictions.csv",
    ),
]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    selection_role: str
    source_file: str


def ensure_dirs() -> None:
    for path in (OUTPUT_FINAL_DIR, OUTPUT_AUDIT_DIR, REPORTS_DIR, PAPER_TABLES_DIR, PAPER_FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {rel_path(path)}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def parse_prediction_frame(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    missing = [column for column in PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{rel_path(source_path)} is missing prediction columns: {missing}")
    parsed = df[PREDICTION_COLUMNS].copy()
    for column in DATE_COLUMNS:
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        if parsed[column].isna().any():
            raise ValueError(f"{rel_path(source_path)} has unparseable {column} values.")
    for column in ("actual_var", "pred_var"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        values = parsed[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{rel_path(source_path)} contains non-finite {column} values.")
    if (parsed["actual_var"] < 0).any():
        raise ValueError(f"{rel_path(source_path)} contains negative actual_var values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{rel_path(source_path)} contains non-positive pred_var values.")
    parsed["model"] = parsed["model"].astype(str).str.strip()
    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    parsed["_source_file"] = rel_path(source_path)
    return parsed.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def load_prediction_path(path: Path, expected_model: str | None = None) -> pd.DataFrame:
    frame = parse_prediction_frame(read_csv(path), path)
    if expected_model is not None and expected_model not in set(frame["model"]):
        raise ValueError(f"{rel_path(path)} does not contain expected model {expected_model!r}.")
    if expected_model is not None:
        frame = frame[frame["model"] == expected_model].copy()
    return frame


def find_prediction_for_model(root: Path, model: str) -> Path | None:
    if not root.exists():
        return None
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = parse_prediction_frame(read_csv(path), path)
        except Exception:
            continue
        if model in set(frame["model"]):
            return path
    return None


def volatility_metrics(actual_var: Iterable[float], pred_var: Iterable[float]) -> dict[str, float | int]:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    if len(actual) == 0:
        raise ValueError("No finite observations remain for metric computation.")
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
    return np.log(np.clip(pred[finite], EPSILON, None)) + actual[finite] / np.clip(pred[finite], EPSILON, None)


def loss_values(actual_var: Iterable[float], pred_var: Iterable[float], loss_type: str) -> np.ndarray:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    if loss_type == "qlike":
        return np.log(pred) + actual / pred
    if loss_type == "squared_error":
        return (actual - pred) ** 2
    if loss_type == "absolute_error":
        return np.abs(actual - pred)
    raise ValueError(f"Unknown loss type: {loss_type}")


def build_har_squared_return_predictions() -> pd.DataFrame:
    splits = load_stage1_splits(PROJECT_ROOT)
    model_ready = load_stage1_model_ready(PROJECT_ROOT)
    features = build_har_feature_frame(model_ready)
    predictions, _coef = fit_ols_predict(
        features,
        splits,
        feature_columns=["sr_d", "sr_w", "sr_m"],
        model_name="HAR-SquaredReturn",
        log_target=False,
    )
    predictions["_source_file"] = "generated_in_memory_from_stage1_har_features"
    return predictions


def selected_calibrated_neural_spec() -> tuple[ModelSpec | None, pd.DataFrame | None]:
    validation_path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_neural_calibration_validation.csv"
    files_path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_neural_calibration_prediction_files.csv"
    if not validation_path.exists() or not files_path.exists():
        return None, None
    validation = read_csv(validation_path)
    files = read_csv(files_path)
    candidates = validation[validation["method"].astype(str).ne("raw")].copy()
    if candidates.empty:
        return None, None
    candidates["QLIKE"] = pd.to_numeric(candidates["QLIKE"], errors="coerce")
    chosen = candidates.sort_values(["QLIKE", "RMSE", "calibrated_model"]).iloc[0]
    model = str(chosen["calibrated_model"])
    file_row = files[files["model"].astype(str).eq(model)]
    if file_row.empty:
        return None, None
    path = PROJECT_ROOT / str(file_row.iloc[0]["prediction_file"])
    if not path.exists():
        return None, None
    spec = ModelSpec(
        model=model,
        selection_role="validation_selected_calibrated_neural",
        source_file=rel_path(path),
    )
    return spec, load_prediction_path(path, model)


def selected_combination_spec() -> tuple[ModelSpec | None, pd.DataFrame | None]:
    validation_path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_forecast_combination_validation.csv"
    if not validation_path.exists():
        return None, None
    validation = read_csv(validation_path)
    if validation.empty:
        return None, None
    validation["QLIKE"] = pd.to_numeric(validation["QLIKE"], errors="coerce")
    chosen = validation.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]
    model = str(chosen["model"])
    path = find_prediction_for_model(
        PROJECT_ROOT / "outputs" / "advanced" / "predictions" / "forecast_combinations",
        model,
    )
    if path is None:
        return None, None
    spec = ModelSpec(
        model=model,
        selection_role="validation_selected_forecast_combination",
        source_file=rel_path(path),
    )
    return spec, load_prediction_path(path, model)


def primary_prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    specs: list[ModelSpec] = []
    missing: list[dict[str, str]] = []

    for model, role, path in PRIMARY_STATIC_SPECS:
        if not path.exists():
            missing.append({"model": model, "selection_role": role, "source_file": rel_path(path)})
            continue
        frame = load_prediction_path(path, model)
        frames.append(frame)
        specs.append(ModelSpec(model=model, selection_role=role, source_file=rel_path(path)))

    try:
        har_sr = build_har_squared_return_predictions()
        frames.append(har_sr)
        specs.append(
            ModelSpec(
                model="HAR-SquaredReturn",
                selection_role="predeclared_har_comparator",
                source_file="generated_in_memory_from_stage1_har_features",
            )
        )
    except Exception as exc:  # noqa: BLE001 - absence should be explicit in the audit.
        missing.append(
            {
                "model": "HAR-SquaredReturn",
                "selection_role": "predeclared_har_comparator",
                "source_file": f"not_available: {type(exc).__name__}: {exc}",
            }
        )

    calibrated_spec, calibrated_frame = selected_calibrated_neural_spec()
    if calibrated_spec is not None and calibrated_frame is not None:
        frames.append(calibrated_frame)
        specs.append(calibrated_spec)

    combo_spec, combo_frame = selected_combination_spec()
    if combo_spec is not None and combo_frame is not None:
        frames.append(combo_frame)
        specs.append(combo_spec)

    if not frames:
        raise RuntimeError("No primary prediction frames could be loaded.")

    predictions = pd.concat(frames, ignore_index=True)
    duplicates = predictions.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicates.any():
        examples = predictions.loc[duplicates, ["model", "split", "date", "target_date", "_source_file"]]
        raise ValueError(f"Primary predictions contain duplicate keys:\n{examples.head(20).to_string(index=False)}")
    specs_df = pd.DataFrame([spec.__dict__ for spec in specs])
    if missing:
        missing_df = pd.DataFrame(missing)
        missing_df["missing"] = True
        specs_df = pd.concat([specs_df, missing_df], ignore_index=True, sort=False)
    return predictions.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True), specs_df


def common_window_for_models(predictions: pd.DataFrame, *, split: str = "test") -> pd.DataFrame:
    split_df = predictions[predictions["split"].eq(split)].copy()
    model_names = sorted(split_df["model"].unique())
    key_sets = [
        set(split_df.loc[split_df["model"].eq(model), ["date", "target_date"]].itertuples(index=False, name=None))
        for model in model_names
    ]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    if not common_keys:
        raise ValueError(f"No common {split} window exists for primary models.")
    common_keys_df = pd.DataFrame(sorted(common_keys), columns=["date", "target_date"])
    common = split_df.merge(common_keys_df, on=["date", "target_date"], how="inner")
    counts = common.groupby("model").size()
    if counts.nunique() != 1:
        raise ValueError(f"Common-window row counts are unequal:\n{counts.to_string()}")
    actual_check = common.groupby(["date", "target_date"])["actual_var"].agg(["min", "max"])
    if not np.isclose(actual_check["min"], actual_check["max"], rtol=1e-10, atol=1e-12).all():
        raise ValueError("Common-window actual_var values differ across models.")
    return common.sort_values(["date", "target_date", "model"]).reset_index(drop=True)


def build_master_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    predictions, specs = primary_prediction_frames()
    common = common_window_for_models(predictions, split="test")
    rows: list[dict[str, Any]] = []
    for model, group in common.groupby("model", sort=False):
        metrics = volatility_metrics(group["actual_var"], group["pred_var"])
        spec_row = specs[specs["model"].eq(model)].iloc[0]
        rows.append(
            {
                "model": model,
                "selection_role": spec_row["selection_role"],
                "target_proxy": TARGET_PROXY,
                "split": "test",
                "origin_date_start": group["date"].min().strftime("%Y-%m-%d"),
                "origin_date_end": group["date"].max().strftime("%Y-%m-%d"),
                "target_date_start": group["target_date"].min().strftime("%Y-%m-%d"),
                "target_date_end": group["target_date"].max().strftime("%Y-%m-%d"),
                **metrics,
                "source_file": spec_row["source_file"],
            }
        )
    master = pd.DataFrame(rows).sort_values(["qlike", "rmse", "model"]).reset_index(drop=True)
    master.insert(0, "rank_qlike", np.arange(1, len(master) + 1))
    common_path = OUTPUT_FINAL_DIR / "common_window_predictions.parquet"
    csv_path = OUTPUT_FINAL_DIR / "common_window_predictions.csv"
    master_path = OUTPUT_FINAL_DIR / "common_window_master_results.csv"
    specs_path = OUTPUT_FINAL_DIR / "primary_model_inventory.csv"
    common.to_parquet(common_path, index=False)
    common.to_csv(csv_path, index=False)
    master.to_csv(master_path, index=False)
    specs.to_csv(specs_path, index=False)
    write_master_results_latex(master, PAPER_TABLES_DIR / "common_window_master_results.tex")
    plot_master_qlike(master, PAPER_FIGURES_DIR / "fig_common_window_master_qlike.png")
    return master, common, specs


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


def write_master_results_latex(master: pd.DataFrame, path: Path) -> None:
    display = master[
        [
            "rank_qlike",
            "model",
            "selection_role",
            "n_obs",
            "qlike",
            "rmse",
            "mae",
            "pred_actual_ratio",
        ]
    ].copy()
    display = display.rename(
        columns={
            "rank_qlike": "Rank",
            "model": "Model",
            "selection_role": "Selection role",
            "n_obs": "N",
            "qlike": "QLIKE",
            "rmse": "RMSE",
            "mae": "MAE",
            "pred_actual_ratio": "Pred./Actual",
        }
    )
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Primary common-window test results. All models use target dates 2023-02-07 to 2025-12-31 and the same next-day squared percentage log-return target.}",
        r"\label{tab:common-window-master}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}r L{0.22\textwidth} L{0.30\textwidth} r r r r r@{}}",
        r"\toprule",
        r"Rank & Model & Selection role & \(N\) & QLIKE & RMSE & MAE & Pred./Actual \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(row["Rank"])),
                    latex_escape(row["Model"]),
                    latex_escape(row["Selection role"]).replace(r"\_", r"\_"),
                    str(int(row["N"])),
                    f"{float(row['QLIKE']):.6f}",
                    f"{float(row['RMSE']):.6f}",
                    f"{float(row['MAE']):.6f}",
                    f"{float(row['Pred./Actual']):.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_master_qlike(master: pd.DataFrame, path: Path) -> None:
    plot = master.sort_values("qlike", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.36 * len(plot))))
    colors = ["#2F6F73" if "GARCH" in model or "HAR" in model else "#6B7280" for model in plot["model"]]
    ax.barh(plot["model"], plot["qlike"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("QLIKE")
    ax.set_title("Primary Common-Window QLIKE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def classify_garch_failure(row: pd.Series) -> tuple[str, str]:
    message = str(row.get("error_message", "")).lower()
    error_type = str(row.get("error_type", ""))
    if "non-finite forecast" in message:
        return "nonfinite_forecast", "Raw model forecast contained non-finite variance values and was rejected before metric computation."
    if "backcasting" in message and "start cannot be less" in message:
        return (
            "invalid_forecast_start_for_ar_mean_lag",
            "The fitted AR-mean specification could not produce forecasts from the requested context start; the candidate was rejected.",
        )
    if "convergence" in message:
        return "optimizer_or_convergence_error", "Optimizer/convergence failure; candidate rejected."
    return "other_numerical_error", "Unclassified numerical failure; candidate rejected."


def build_failure_taxonomy() -> pd.DataFrame:
    ensure_dirs()
    failures_path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_garch_family_fit_failures.csv"
    failures = read_csv(failures_path)
    rows = []
    for category, group in failures.groupby(failures.apply(lambda row: classify_garch_failure(row)[0], axis=1)):
        example = group.iloc[0]
        handling = classify_garch_failure(example)[1]
        rows.append(
            {
                "failure_category": category,
                "count": int(len(group)),
                "stage": "; ".join(sorted(group["stage"].astype(str).unique())),
                "error_type": "; ".join(sorted(group["error_type"].astype(str).unique())),
                "example_candidate_id": example["candidate_id"],
                "example_mean_label": example["mean_label"],
                "example_volatility_label": example["volatility_label"],
                "example_distribution": example["distribution"],
                "example_error_message": example["error_message"],
                "handling_rule": handling,
            }
        )
    taxonomy = pd.DataFrame(rows).sort_values(["failure_category"]).reset_index(drop=True)
    if int(taxonomy["count"].sum()) != len(failures):
        raise ValueError("Failure taxonomy counts do not sum to total failures.")
    out_path = OUTPUT_AUDIT_DIR / "advanced_garch_failure_taxonomy.csv"
    tex_path = PAPER_TABLES_DIR / "advanced_garch_failure_audit.tex"
    taxonomy.to_csv(out_path, index=False)
    write_failure_latex(taxonomy, tex_path)
    return taxonomy


def write_failure_latex(taxonomy: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Advanced GARCH failure taxonomy. Categories sum exactly to the 576 failed candidates.}",
        r"\label{tab:advanced-garch-failure-audit}",
        r"\scriptsize",
        r"\begin{tabular}{@{}L{0.48\columnwidth}rL{0.32\columnwidth}@{}}",
        r"\toprule",
        r"Failure category & Count & Example \\",
        r"\midrule",
    ]
    for _, row in taxonomy.iterrows():
        example = f"{row['example_mean_label']}, {row['example_volatility_label']}, {row['example_distribution']}"
        lines.append(
            f"{latex_escape(row['failure_category'])} & {int(row['count'])} & {latex_escape(example)} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_returns() -> pd.DataFrame:
    path = DATA_DIR / "vnindex_model_ready.csv"
    df = read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["log_return_pct", "squared_return", "abs_return", "target_var_next"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date", "log_return_pct", "squared_return"]).sort_values("date")


def build_descriptive_and_arch_diagnostics() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    df = clean_returns()
    returns = df["log_return_pct"].dropna().astype(float)
    jb = jarque_bera(returns)
    desc = pd.DataFrame(
        [
            {
                "series": "log_return_pct",
                "n_obs": int(returns.size),
                "mean": float(returns.mean()),
                "std": float(returns.std(ddof=1)),
                "min": float(returns.min()),
                "max": float(returns.max()),
                "skewness": float(skew(returns, bias=False)),
                "excess_kurtosis": float(kurtosis(returns, fisher=True, bias=False)),
                "jarque_bera_stat": float(jb.statistic),
                "jarque_bera_p_value": float(jb.pvalue),
            }
        ]
    )
    desc.to_csv(OUTPUT_FINAL_DIR / "return_descriptive_statistics.csv", index=False)
    write_desc_latex(desc, PAPER_TABLES_DIR / "return_descriptive_statistics.tex")

    rows = []
    series_map = {
        "returns": returns,
        "squared_returns": df["squared_return"].dropna().astype(float),
        "absolute_returns": df["abs_return"].dropna().astype(float),
    }
    for name, values in series_map.items():
        lb = acorr_ljungbox(values.to_numpy(dtype=float), lags=[5, 10, 20], return_df=True)
        for lag, row in lb.iterrows():
            rows.append(
                {
                    "test": "Ljung-Box",
                    "series": name,
                    "lags": int(lag),
                    "n_obs": int(values.size),
                    "statistic": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                }
            )
    lb_table = pd.DataFrame(rows)
    lb_table.to_csv(OUTPUT_FINAL_DIR / "ljung_box_diagnostics.csv", index=False)

    arch_rows = []
    demeaned = returns.to_numpy(dtype=float) - float(returns.mean())
    for lag in [5, 10, 20]:
        lm_stat, lm_p, f_stat, f_p = het_arch(demeaned, nlags=lag, ddof=0)
        arch_rows.append(
            {
                "test": "ARCH-LM",
                "series": "demeaned_returns",
                "lags": lag,
                "n_obs": int(returns.size),
                "lm_stat": float(lm_stat),
                "lm_p_value": float(lm_p),
                "f_stat": float(f_stat),
                "f_p_value": float(f_p),
            }
        )
    arch_table = pd.DataFrame(arch_rows)
    arch_table.to_csv(OUTPUT_FINAL_DIR / "arch_lm_diagnostics.csv", index=False)
    write_diagnostics_latex(lb_table, arch_table, PAPER_TABLES_DIR / "volatility_clustering_diagnostics.tex")
    plot_return_diagnostics(df)
    return {"descriptive": desc, "ljung_box": lb_table, "arch_lm": arch_table}


def write_desc_latex(desc: pd.DataFrame, path: Path) -> None:
    row = desc.iloc[0]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Descriptive statistics for daily VN-Index percentage log returns.}",
        r"\label{tab:return-descriptive-statistics}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        f"Observations & {int(row['n_obs'])} " + r"\\",
        f"Mean & {float(row['mean']):.6f} " + r"\\",
        f"Std. dev. & {float(row['std']):.6f} " + r"\\",
        f"Minimum & {float(row['min']):.6f} " + r"\\",
        f"Maximum & {float(row['max']):.6f} " + r"\\",
        f"Skewness & {float(row['skewness']):.6f} " + r"\\",
        f"Excess kurtosis & {float(row['excess_kurtosis']):.6f} " + r"\\",
        f"Jarque-Bera stat. & {float(row['jarque_bera_stat']):.3f} " + r"\\",
        f"Jarque-Bera p-value & {float(row['jarque_bera_p_value']):.3g} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostics_latex(lb: pd.DataFrame, arch: pd.DataFrame, path: Path) -> None:
    keep = pd.concat(
        [
            lb[lb["series"].isin(["returns", "squared_returns"])][["test", "series", "lags", "statistic", "p_value"]],
            arch[["test", "series", "lags", "lm_stat", "lm_p_value"]].rename(
                columns={"lm_stat": "statistic", "lm_p_value": "p_value"}
            ),
        ],
        ignore_index=True,
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Return-dependence and ARCH-effect diagnostics.}",
        r"\label{tab:volatility-clustering-diagnostics}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Test & Series & Lag & Statistic & \(p\)-value \\",
        r"\midrule",
    ]
    for _, row in keep.iterrows():
        lines.append(
            f"{latex_escape(row['test'])} & {latex_escape(row['series'])} & {int(row['lags'])} & "
            f"{float(row['statistic']):.3f} & {float(row['p_value']):.3g} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_return_diagnostics(df: pd.DataFrame) -> None:
    returns = df["log_return_pct"].astype(float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(df["date"], returns, linewidth=0.7, color="#2F6F73")
    axes[0].set_title("VN-Index Percentage Log Returns")
    axes[0].set_ylabel("Return (%)")
    axes[1].plot(df["date"], df["squared_return"], linewidth=0.7, color="#7A4E2D")
    axes[1].set_title("Squared Returns")
    axes[1].set_ylabel("Squared return")
    axes[2].hist(returns, bins=70, density=True, color="#6B7280", alpha=0.85)
    axes[2].set_title("Return Distribution")
    axes[2].set_xlabel("Return (%)")
    for ax in axes[:2]:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES_DIR / "fig_return_descriptive_diagnostics.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    plot_acf(returns, lags=40, ax=axes[0], title="ACF: returns")
    plot_acf(df["squared_return"].astype(float), lags=40, ax=axes[1], title="ACF: squared returns")
    plot_acf(df["abs_return"].astype(float), lags=40, ax=axes[2], title="ACF: absolute returns")
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES_DIR / "fig_return_acf_diagnostics.png", dpi=250)
    plt.close(fig)


def load_split_frames() -> dict[str, pd.DataFrame]:
    splits = {}
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"
        frame = read_csv(path)
        for column in DATE_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        for column in ["log_return_pct", "squared_return", "target_var_next"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        splits[split] = frame.sort_values("date").reset_index(drop=True)
    return splits


def load_model_ready_frame() -> pd.DataFrame:
    frame = read_csv(DATA_DIR / "vnindex_model_ready.csv")
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    numeric_columns = [
        "log_return_pct",
        "squared_return",
        "target_var_next",
        "open",
        "high",
        "low",
        "close",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "target_date", "log_return_pct", "squared_return"]).sort_values("date").reset_index(drop=True)


def selected_advanced_spec(selection_role: str) -> CandidateSpec:
    selected_path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_garch_family_selected_models.csv"
    selected = read_csv(selected_path)
    row = selected[selected["selection_role"].astype(str).eq(selection_role)]
    if row.empty:
        raise ValueError(f"Could not find advanced GARCH selection role {selection_role!r}.")
    return parse_candidate_from_row(row.iloc[0])


def arch_result_converged(result: Any) -> bool:
    if hasattr(result, "convergence_flag"):
        return int(result.convergence_flag) == 0
    opt = getattr(result, "optimization_result", None)
    if opt is not None and hasattr(opt, "success"):
        return bool(opt.success)
    return False


def param_value(series: pd.Series, names: Iterable[str], default: float | None = None) -> float | None:
    for name in names:
        if name in series.index:
            return float(series[name])
    lowered = {str(index).lower(): index for index in series.index}
    for name in names:
        key = name.lower()
        if key in lowered:
            return float(series[lowered[key]])
    return default


def garch11_persistence_metrics(params: pd.Series) -> dict[str, Any]:
    alpha = param_value(params, ["alpha[1]", "alpha1"])
    beta = param_value(params, ["beta[1]", "beta1"])
    omega = param_value(params, ["omega"])
    persistence = np.nan if alpha is None or beta is None else alpha + beta
    valid = bool(np.isfinite(persistence) and persistence > 0 and persistence < 1)
    unconditional = omega / (1.0 - persistence) if valid and omega is not None else np.nan
    half_life = math.log(0.5) / math.log(persistence) if valid else np.nan
    return {
        "persistence_measure": "alpha_plus_beta",
        "persistence_value": float(persistence) if np.isfinite(persistence) else np.nan,
        "unconditional_variance": float(unconditional) if np.isfinite(unconditional) else np.nan,
        "shock_half_life_days": float(half_life) if np.isfinite(half_life) else np.nan,
        "validity_flag": "stationary_garch" if valid else "persistence_not_in_unit_interval",
    }


def egarch_persistence_metrics(params: pd.Series) -> dict[str, Any]:
    beta_values = [float(value) for name, value in params.items() if str(name).startswith("beta[")]
    beta_sum = float(np.sum(beta_values)) if beta_values else np.nan
    valid = bool(np.isfinite(beta_sum) and abs(beta_sum) < 1)
    return {
        "persistence_measure": "sum_beta_log_variance",
        "persistence_value": beta_sum,
        "unconditional_variance": np.nan,
        "shock_half_life_days": np.nan,
        "validity_flag": "log_variance_persistence_below_one" if valid else "egarch_half_life_not_reported",
    }


def result_param_rows(
    *,
    model: str,
    model_role: str,
    fit_sample: str,
    n_fit_obs: int,
    model_spec: str,
    mean_equation: str,
    variance_equation: str,
    distribution: str,
    component: str,
    result: Any,
    persistence: dict[str, Any],
    interpretation: str,
) -> list[dict[str, Any]]:
    params = pd.Series(getattr(result, "params", pd.Series(dtype=float)))
    std_errors = pd.Series(getattr(result, "std_err", pd.Series(index=params.index, dtype=float)))
    tvalues = pd.Series(getattr(result, "tvalues", pd.Series(index=params.index, dtype=float)))
    pvalues = pd.Series(getattr(result, "pvalues", pd.Series(index=params.index, dtype=float)))
    rows = []
    for parameter, estimate in params.items():
        rows.append(
            {
                "model": model,
                "model_role": model_role,
                "fit_sample": fit_sample,
                "n_fit_obs": int(n_fit_obs),
                "model_spec": model_spec,
                "mean_equation": mean_equation,
                "variance_equation": variance_equation,
                "innovation_distribution": distribution,
                "component": component,
                "parameter": str(parameter),
                "estimate": float(estimate),
                "std_error": float(std_errors.get(parameter, np.nan)),
                "t_stat": float(tvalues.get(parameter, np.nan)),
                "p_value": float(pvalues.get(parameter, np.nan)),
                "converged": bool(arch_result_converged(result)),
                **persistence,
                "interpretation": interpretation,
            }
        )
    return rows


def residual_diagnostic_rows(model: str, std_resid: Iterable[float]) -> list[dict[str, Any]]:
    values = pd.to_numeric(pd.Series(std_resid), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = []
    if values.empty:
        return rows
    arr = values.to_numpy(dtype=float)
    for series_label, series_values in {
        "standardized_residual": arr,
        "squared_standardized_residual": arr**2,
    }.items():
        lb = acorr_ljungbox(series_values, lags=[5, 10, 20], return_df=True)
        for lag, row in lb.iterrows():
            rows.append(
                {
                    "model": model,
                    "diagnostic": "Ljung-Box",
                    "series": series_label,
                    "lags": int(lag),
                    "n_obs": int(len(series_values)),
                    "statistic": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                    "interpretation_flag": "remaining_dependence" if float(row["lb_pvalue"]) < 0.05 else "not_rejected",
                }
            )
    for lag in [5, 10, 20]:
        lm_stat, lm_p, _f_stat, _f_p = het_arch(arr, nlags=lag, ddof=0)
        rows.append(
            {
                "model": model,
                "diagnostic": "ARCH-LM",
                "series": "standardized_residual",
                "lags": lag,
                "n_obs": int(len(arr)),
                "statistic": float(lm_stat),
                "p_value": float(lm_p),
                "interpretation_flag": "remaining_arch" if float(lm_p) < 0.05 else "not_rejected",
            }
        )
    jb = jarque_bera(arr)
    rows.append(
        {
            "model": model,
            "diagnostic": "Jarque-Bera",
            "series": "standardized_residual",
            "lags": 0,
            "n_obs": int(len(arr)),
            "statistic": float(jb.statistic),
            "p_value": float(jb.pvalue),
            "interpretation_flag": "non_normal" if float(jb.pvalue) < 0.05 else "normality_not_rejected",
        }
    )
    return rows


def fit_arch_result_for_spec(spec: CandidateSpec, fit_df: pd.DataFrame) -> tuple[Any, str]:
    fit_y = pd.Series(fit_df["log_return_pct"].astype(float).to_numpy(), name="returns")
    if spec.uses_arima_mean:
        if spec.arima_p is None or spec.arima_q is None:
            raise ValueError("Two-step ARIMA-GARCH spec is missing ARIMA p/q.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_result = ARIMA(fit_y.to_numpy(dtype=float), order=(int(spec.arima_p), 0, int(spec.arima_q))).fit()
        arima_mean = np.asarray(arima_result.predict(start=0, end=len(fit_y) - 1), dtype=float)
        residuals = fit_y.to_numpy(dtype=float) - arima_mean
        model = arch_model(
            pd.Series(residuals, name="arima_resid"),
            mean="Zero",
            vol=spec.arch_vol,
            p=spec.p,
            o=spec.o,
            q=spec.q,
            dist=spec.distribution,
            rescale=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False, options={"maxiter": 400})
        mean_description = f"two-step ARIMA({int(spec.arima_p)},0,{int(spec.arima_q)}) residuals"
        return result, mean_description

    model = arch_model(
        fit_y,
        mean=spec.arch_mean,
        lags=spec.arch_lags if spec.arch_mean == "AR" else 0,
        vol=spec.arch_vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.distribution,
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off", show_warning=False, options={"maxiter": 400})
    return result, spec.mean_label


def build_garch_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    splits = load_split_frames()
    fit_df = splits["train"].dropna(subset=["log_return_pct"]).copy()
    specs: list[tuple[str, str, CandidateSpec, str]] = [
        (
            "GARCH(1,1)",
            "standard_training_selected_benchmark",
            original_garch_spec(),
            "Standard GARCH benchmark estimated on the training split.",
        ),
        (
            "AdvGARCH-BestQLIKE",
            "primary_validation_selected_advanced_garch",
            selected_advanced_spec("best_fixed_by_validation_qlike"),
            "Primary validation-selected advanced GARCH-family specification.",
        ),
        (
            "AdvGARCH-BestAsymmetric",
            "category_selected_asymmetric_garch_comparator",
            selected_advanced_spec("best_asymmetric_garch_by_validation_qlike"),
            "Asymmetric category-selected comparator; not promoted solely by test QLIKE.",
        ),
    ]
    param_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for model_name, role, spec, interpretation in specs:
        result, mean_description = fit_arch_result_for_spec(spec, fit_df)
        if spec.arch_vol.upper() == "GARCH" and spec.p == 1 and spec.o == 0 and spec.q == 1:
            persistence = garch11_persistence_metrics(pd.Series(result.params))
        elif spec.arch_vol.upper() == "EGARCH":
            persistence = egarch_persistence_metrics(pd.Series(result.params))
        else:
            persistence = {
                "persistence_measure": "not_reported_for_model_class",
                "persistence_value": np.nan,
                "unconditional_variance": np.nan,
                "shock_half_life_days": np.nan,
                "validity_flag": "not_reported",
            }
        param_rows.extend(
            result_param_rows(
                model=model_name,
                model_role=role,
                fit_sample="training_split",
                n_fit_obs=len(fit_df),
                model_spec=spec.candidate_id,
                mean_equation=mean_description,
                variance_equation=spec.volatility_label,
                distribution=spec.distribution,
                component="variance",
                result=result,
                persistence=persistence,
                interpretation=interpretation,
            )
        )
        residual_rows.extend(residual_diagnostic_rows(model_name, getattr(result, "std_resid", [])))

    params = pd.DataFrame(param_rows)
    residuals = pd.DataFrame(residual_rows)
    params.to_csv(OUTPUT_FINAL_DIR / "garch_parameter_diagnostics.csv", index=False)
    residuals.to_csv(OUTPUT_FINAL_DIR / "garch_residual_diagnostics.csv", index=False)
    write_garch_parameter_latex(params, PAPER_TABLES_DIR / "garch_parameter_diagnostics.tex")
    write_garch_residual_latex(residuals, PAPER_TABLES_DIR / "garch_residual_diagnostics.tex")
    plot_garch_residual_diagnostics(residuals, PAPER_FIGURES_DIR / "fig_garch_residual_diagnostics.png")
    return params, residuals


def write_garch_parameter_latex(params: pd.DataFrame, path: Path) -> None:
    keep_names = ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "beta[2]", "nu", "lambda"]
    display = params[params["parameter"].isin(keep_names)].copy()
    if display.empty:
        display = params.copy()
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Selected GARCH-family parameter diagnostics. Persistence and half-life are reported only where the formula is technically appropriate.}",
        r"\label{tab:garch-parameter-diagnostics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}L{0.20\textwidth}L{0.16\textwidth}L{0.10\textwidth}rrrrL{0.13\textwidth}@{}}",
        r"\toprule",
        r"Model & Variance equation & Parameter & Estimate & Std. err. & \(p\)-value & Persistence & Validity \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        persistence = float(row["persistence_value"]) if np.isfinite(row["persistence_value"]) else np.nan
        persistence_text = f"{persistence:.4f}" if np.isfinite(persistence) else ""
        validity = {
            "stationary_garch": "stationary",
            "log_variance_persistence_below_one": "EGARCH persist.",
            "egarch_half_life_not_reported": "not reported",
            "persistence_not_in_unit_interval": "not stationary",
        }.get(str(row["validity_flag"]), str(row["validity_flag"]))
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['variance_equation'])} & "
            f"{latex_escape(row['parameter'])} & {float(row['estimate']):.4f} & "
            f"{float(row['std_error']):.4f} & {float(row['p_value']):.3g} & "
            f"{persistence_text} & {latex_escape(validity)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_garch_residual_latex(residuals: pd.DataFrame, path: Path) -> None:
    keep = residuals[
        (residuals["diagnostic"].isin(["Ljung-Box", "ARCH-LM"]))
        & (residuals["series"].isin(["squared_standardized_residual", "standardized_residual"]))
    ].copy()
    keep = keep[
        ((keep["diagnostic"] == "Ljung-Box") & (keep["series"] == "squared_standardized_residual"))
        | ((keep["diagnostic"] == "ARCH-LM") & (keep["series"] == "standardized_residual"))
    ]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Post-fit residual diagnostics for selected GARCH-family models.}",
        r"\label{tab:garch-residual-diagnostics}",
        r"\scriptsize",
        r"\begin{tabular}{@{}L{0.20\textwidth}L{0.10\textwidth}L{0.19\textwidth}rrrr@{}}",
        r"\toprule",
        r"Model & Test & Series & Lag & Statistic & \(p\)-value & Flag \\",
        r"\midrule",
    ]
    for _, row in keep.iterrows():
        series = "Std. resid." if row["series"] == "standardized_residual" else "Squared std. resid."
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['diagnostic'])} & {series} & "
            f"{int(row['lags'])} & {float(row['statistic']):.3f} & {float(row['p_value']):.3g} & "
            f"{latex_escape(row['interpretation_flag'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_garch_residual_diagnostics(residuals: pd.DataFrame, path: Path) -> None:
    plot = residuals[
        (residuals["diagnostic"].eq("ARCH-LM"))
        & (residuals["lags"].eq(10))
    ].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(plot["model"], plot["p_value"], color="#2F6F73")
    ax.axhline(0.05, color="#8C2D19", linewidth=1.1, linestyle="--")
    ax.set_ylabel("ARCH-LM p-value, lag 10")
    ax.set_title("Remaining ARCH Effects After GARCH-Family Fits")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def normal_p_value(statistic: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(statistic))))


def newey_west_lrv(diff: np.ndarray) -> tuple[float, int]:
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    centered = diff - float(np.mean(diff))
    lag = min(int(np.floor(n ** (1.0 / 3.0))), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for current_lag in range(1, lag + 1):
        cov = float(np.dot(centered[current_lag:], centered[:-current_lag]) / n)
        weight = 1.0 - current_lag / (lag + 1.0)
        lrv += 2.0 * weight * cov
    return max(float(lrv), 0.0), lag


def dm_test(diff: np.ndarray) -> tuple[float, float, int]:
    diff = np.asarray(diff, dtype=float)
    lrv, lag = newey_west_lrv(diff)
    mean_diff = float(np.mean(diff))
    if lrv <= 0:
        return (0.0, 1.0, lag) if abs(mean_diff) <= 1e-14 else (np.nan, np.nan, lag)
    stat = mean_diff / math.sqrt(lrv / len(diff))
    return float(stat), normal_p_value(float(stat)), lag


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) if np.isfinite(float(value)) else np.nan for value in p_values]
    finite = sorted([(idx, value) for idx, value in enumerate(values) if np.isfinite(value)], key=lambda item: item[1])
    adjusted = [np.nan] * len(values)
    running = 0.0
    total = len(finite)
    for rank, (idx, value) in enumerate(finite, start=1):
        running = max(running, min(1.0, (total - rank + 1) * value))
        adjusted[idx] = running
    return adjusted


def build_common_dm_tests(common: pd.DataFrame, benchmark: str = "GARCH(1,1)") -> pd.DataFrame:
    rows = []
    if benchmark not in set(common["model"]):
        return pd.DataFrame()
    bench = common[common["model"].eq(benchmark)][["date", "target_date", "actual_var", "pred_var"]].rename(
        columns={"pred_var": "pred_benchmark"}
    )
    for model in sorted(set(common["model"]) - {benchmark}):
        model_df = common[common["model"].eq(model)][["date", "target_date", "actual_var", "pred_var"]].rename(
            columns={"pred_var": "pred_model"}
        )
        merged = model_df.merge(bench[["date", "target_date", "pred_benchmark"]], on=["date", "target_date"])
        for loss_type in ["qlike", "squared_error", "absolute_error"]:
            loss_model = loss_values(merged["actual_var"], merged["pred_model"], loss_type)
            loss_bench = loss_values(merged["actual_var"], merged["pred_benchmark"], loss_type)
            diff = loss_model - loss_bench
            stat, p_value, lag = dm_test(diff)
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "loss_type": loss_type,
                    "n_obs": int(len(diff)),
                    "mean_loss_model": float(np.mean(loss_model)),
                    "mean_loss_benchmark": float(np.mean(loss_bench)),
                    "mean_diff": float(np.mean(diff)),
                    "dm_stat": stat,
                    "p_value": p_value,
                    "hac_lag": lag,
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["holm_adjusted_p_value"] = np.nan
        for loss_type, index in table.groupby("loss_type").groups.items():
            table.loc[index, "holm_adjusted_p_value"] = holm_adjust(table.loc[index, "p_value"])
    table.to_csv(OUTPUT_FINAL_DIR / "common_window_dm_tests.csv", index=False)
    return table


def build_regime_and_spike(common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = common.groupby(["date", "target_date"], as_index=False)["actual_var"].first()
    q50, q75, q90 = [float(actual["actual_var"].quantile(q)) for q in [0.50, 0.75, 0.90]]
    regime_rows = []
    spike_rows = []
    common = common.copy()
    common["regime"] = np.where(
        common["actual_var"] <= q50,
        "calm",
        np.where(common["actual_var"] <= q75, "normal", np.where(common["actual_var"] <= q90, "high", "extreme")),
    )
    for (model, regime), group in common.groupby(["model", "regime"]):
        regime_rows.append({"model": model, "regime": regime, **volatility_metrics(group["actual_var"], group["pred_var"])})
    for model, group in common.groupby("model"):
        spike = group[group["actual_var"] >= q90].copy()
        if spike.empty:
            continue
        ratio = np.clip(spike["pred_var"], EPSILON, None) / np.clip(spike["actual_var"], EPSILON, None)
        spike_rows.append(
            {
                "model": model,
                "threshold_level": 0.90,
                "threshold_value": q90,
                "spike_days": int(len(spike)),
                "spike_underprediction_rate": float((spike["pred_var"] < spike["actual_var"]).mean()),
                "severe_spike_underprediction_rate": float((ratio < 0.5).mean()),
                "average_pred_actual_ratio_spike_days": float(ratio.mean()),
                "qlike_spike_days": float(np.mean(qlike_loss(spike["actual_var"], spike["pred_var"]))),
            }
        )
    regime = pd.DataFrame(regime_rows).sort_values(["regime", "qlike", "model"])
    spike_table = pd.DataFrame(spike_rows).sort_values(["severe_spike_underprediction_rate", "model"])
    regime.to_csv(OUTPUT_FINAL_DIR / "common_window_regime_metrics.csv", index=False)
    spike_table.to_csv(OUTPUT_FINAL_DIR / "common_window_spike_diagnostics.csv", index=False)
    write_spike_latex(spike_table, PAPER_TABLES_DIR / "common_window_spike_diagnostics.tex")
    return regime, spike_table


def write_spike_latex(spike_table: pd.DataFrame, path: Path) -> None:
    selected = [
        "AdvGARCH-BestAsymmetric",
        "AdvGARCH-BestQLIKE",
        "HAR-Parkinson",
        "GARCH(1,1)",
        "ARIMA-GARCH",
        "EWMA(lambda=0.90)",
        "LSTM",
        "ARIMA-GARCH-LSTM",
    ]
    display = spike_table[spike_table["model"].isin(selected)].copy()
    display["order"] = display["model"].map({model: idx for idx, model in enumerate(selected)})
    display = display.sort_values("order")
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Common-window spike-day diagnostics at the 90th percentile of actual variance.}",
        r"\label{tab:common-window-spike-diagnostics}",
        r"\scriptsize",
        r"\begin{tabular}{@{}L{0.24\textwidth}rrrr@{}}",
        r"\toprule",
        r"Model & Spike days & Underpred. & Severe & Spike QLIKE \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {int(row['spike_days'])} & "
            f"{100.0 * float(row['spike_underprediction_rate']):.1f}\\% & "
            f"{100.0 * float(row['severe_spike_underprediction_rate']):.1f}\\% & "
            f"{float(row['qlike_spike_days']):.3f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bernoulli_loglik(successes: int, failures: int, probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return successes * math.log(p) + failures * math.log(1.0 - p)


def kupiec_test(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    n = int(len(violations))
    x = int(np.sum(violations))
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    lr = -2.0 * (bernoulli_loglik(x, n - x, alpha) - bernoulli_loglik(x, n - x, phat))
    lr = max(float(lr), 0.0)
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence(violations: np.ndarray) -> tuple[float, float, dict[str, int]]:
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan, {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev, curr = v[:-1], v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    pi01 = n01 / (n00 + n01) if (n00 + n01) else pi
    pi11 = n11 / (n10 + n11) if (n10 + n11) else pi
    restricted = bernoulli_loglik(n01 + n11, n00 + n10, pi)
    unrestricted = bernoulli_loglik(n01, n00, pi01) + bernoulli_loglik(n11, n10, pi11)
    lr = max(float(-2.0 * (restricted - unrestricted)), 0.0)
    return lr, float(chi2.sf(lr, 1)), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def target_returns() -> pd.DataFrame:
    clean = read_csv(DATA_DIR / "vnindex_cafef_2010_2025_clean.csv")
    clean["target_date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["target_return"] = pd.to_numeric(clean["log_return_pct"], errors="coerce")
    return clean.dropna(subset=["target_date", "target_return"])[["target_date", "target_return"]]


def build_common_var(common: pd.DataFrame) -> pd.DataFrame:
    returns = target_returns()
    merged = common.merge(returns, on="target_date", how="left", validate="many_to_one")
    if merged["target_return"].isna().any():
        raise ValueError("Missing target returns for common-window VaR evaluation.")
    rows = []
    for alpha in [0.05, 0.025, 0.01]:
        z = NormalDist().inv_cdf(alpha)
        for model, group in merged.groupby("model"):
            ordered = group.sort_values("target_date")
            sigma = np.sqrt(np.clip(ordered["pred_var"].to_numpy(dtype=float), EPSILON, None))
            var_forecast = z * sigma
            returns_arr = ordered["target_return"].to_numpy(dtype=float)
            violations = returns_arr < var_forecast
            kupiec_lr, kupiec_p = kupiec_test(violations, alpha)
            ind_lr, ind_p, transitions = christoffersen_independence(violations)
            cc_lr = kupiec_lr + ind_lr if np.isfinite(kupiec_lr) and np.isfinite(ind_lr) else np.nan
            rows.append(
                {
                    "model": model,
                    "var_level": alpha,
                    "var_distribution": "common_normal",
                    "distribution_specific_quantile_available": False,
                    "n_obs": int(len(ordered)),
                    "observed_violations": int(violations.sum()),
                    "expected_violations": float(alpha * len(ordered)),
                    "violation_rate": float(violations.mean()),
                    "kupiec_stat": kupiec_lr,
                    "kupiec_p_value": kupiec_p,
                    "christoffersen_independence_stat": ind_lr,
                    "christoffersen_independence_p_value": ind_p,
                    "christoffersen_conditional_coverage_stat": cc_lr,
                    "christoffersen_conditional_coverage_p_value": float(chi2.sf(cc_lr, 2)) if np.isfinite(cc_lr) else np.nan,
                    **transitions,
                }
            )
    table = pd.DataFrame(rows).sort_values(["var_level", "kupiec_p_value", "model"], ascending=[True, False, True])
    table.to_csv(OUTPUT_FINAL_DIR / "common_window_var_backtests.csv", index=False)
    write_common_var_latex(table, PAPER_TABLES_DIR / "common_window_var_backtests.tex")
    return table


def write_common_var_latex(table: pd.DataFrame, path: Path) -> None:
    selected = [
        "AdvGARCH-BestAsymmetric",
        "AdvGARCH-BestQLIKE",
        "HAR-Parkinson",
        "GARCH(1,1)",
        "ARIMA-GARCH",
        "EWMA(lambda=0.90)",
    ]
    display = table[
        table["model"].isin(selected) & table["var_level"].isin([0.05, 0.01])
    ].copy()
    display["order"] = display["model"].map({model: idx for idx, model in enumerate(selected)})
    display = display.sort_values(["order", "var_level"], ascending=[True, False])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Common-window common-Normal VaR backtests.}",
        r"\label{tab:common-window-var-backtests}",
        r"\scriptsize",
        r"\begin{tabular}{@{}L{0.38\columnwidth}rrrrr@{}}",
        r"\toprule",
        r"Model & Level & \(N\) & Viol. & Expected & Kupiec \(p\) \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {100.0 * float(row['var_level']):.0f}\\% & "
            f"{int(row['n_obs'])} & {int(row['observed_violations'])} & "
            f"{float(row['expected_violations']):.2f} & {float(row['kupiec_p_value']):.3g} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REFIT_MODEL_SPECS = [
    ("GARCH(1,1)", "static_training_selected_benchmark"),
    ("ARIMA-GARCH", "static_training_aic_mean_plus_garch"),
    ("EWMA(lambda=0.90)", "validation_selected_ewma"),
    ("HAR-Parkinson", "validation_selected_har"),
    ("AdvGARCH-BestQLIKE", "primary_validation_selected_advanced_garch"),
    ("AdvGARCH-BestAsymmetric", "category_selected_asymmetric_garch_comparator"),
]


def refit_period_starts(target_df: pd.DataFrame) -> list[pd.Timestamp]:
    dates = target_df["date"].sort_values().reset_index(drop=True)
    groups = dates.groupby([dates.dt.year, dates.dt.month])
    return [pd.Timestamp(values.iloc[0]) for _, values in groups]


def make_refit_prediction_frame(
    target_df: pd.DataFrame,
    pred_var: Iterable[float],
    *,
    model: str,
    protocol: str,
) -> pd.DataFrame:
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(pred) != len(target_df):
        raise ValueError(f"{model} {protocol}: {len(pred)} predictions for {len(target_df)} rows.")
    if not np.isfinite(pred).all():
        raise ValueError(f"{model} {protocol} generated non-finite predictions.")
    out = target_df[["date", "target_date", "target_var_next"]].copy()
    out = out.rename(columns={"target_var_next": "actual_var"})
    out["pred_var"] = np.clip(pred, EPSILON, None)
    out["model"] = model
    out["protocol"] = protocol
    out["split"] = "test"
    return out[["date", "target_date", "actual_var", "pred_var", "model", "protocol", "split"]]


def static_refit_frames(common: pd.DataFrame) -> list[pd.DataFrame]:
    frames = []
    for model, _role in REFIT_MODEL_SPECS:
        model_common = common[common["model"].eq(model)].copy()
        if model_common.empty:
            continue
        frame = model_common[["date", "target_date", "actual_var", "pred_var", "model", "split"]].copy()
        frame["protocol"] = "static_train_only"
        frames.append(frame[["date", "target_date", "actual_var", "pred_var", "model", "protocol", "split"]])
    return frames


def candidate_spec_for_refit(model: str) -> CandidateSpec:
    if model == "GARCH(1,1)":
        return original_garch_spec()
    if model == "ARIMA-GARCH":
        return original_arima_garch_spec()
    if model == "AdvGARCH-BestQLIKE":
        return selected_advanced_spec("best_fixed_by_validation_qlike")
    if model == "AdvGARCH-BestAsymmetric":
        return selected_advanced_spec("best_asymmetric_garch_by_validation_qlike")
    raise ValueError(f"No ARCH candidate spec is defined for {model}.")


def expanding_monthly_arch_predictions(
    *,
    model: str,
    model_ready: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    spec = candidate_spec_for_refit(model)
    starts = refit_period_starts(target_df)
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    failures = 0
    for index, start_date in enumerate(starts):
        end_date = starts[index + 1] if index + 1 < len(starts) else pd.Timestamp.max
        segment = target_df[(target_df["date"] >= start_date) & (target_df["date"] < end_date)].copy()
        fit_df = model_ready[model_ready["date"] < start_date].dropna(subset=["log_return_pct"]).copy()
        if segment.empty:
            continue
        segment_start = time.perf_counter()
        try:
            context_df = pd.concat([fit_df, segment], ignore_index=True)
            pred, metadata = forecast_on_context(
                spec=spec,
                fit_df=fit_df,
                context_df=context_df,
                target_df=segment,
            )
            part = segment[["date", "target_date"]].copy()
            part["pred_var"] = pred
            parts.append(part)
            diagnostics.append(
                {
                    "model": model,
                    "protocol": "expanding_monthly",
                    "segment_start": start_date.strftime("%Y-%m-%d"),
                    "segment_end_exclusive": "end" if end_date == pd.Timestamp.max else end_date.strftime("%Y-%m-%d"),
                    "status": "success",
                    "fit_rows": int(len(fit_df)),
                    "segment_rows": int(len(segment)),
                    "runtime_seconds": float(time.perf_counter() - segment_start),
                    "message": json.dumps(metadata, sort_keys=True),
                }
            )
        except Exception as exc:  # noqa: BLE001 - failures are part of robustness evidence.
            failures += 1
            diagnostics.append(
                {
                    "model": model,
                    "protocol": "expanding_monthly",
                    "segment_start": start_date.strftime("%Y-%m-%d"),
                    "segment_end_exclusive": "end" if end_date == pd.Timestamp.max else end_date.strftime("%Y-%m-%d"),
                    "status": "failure",
                    "fit_rows": int(len(fit_df)),
                    "segment_rows": int(len(segment)),
                    "runtime_seconds": float(time.perf_counter() - segment_start),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    if failures:
        # Keep successful segments for operational diagnostics, but the result
        # will be marked as an incomplete common-window refit.
        pass
    if not parts:
        raise RuntimeError(f"{model} expanding-monthly refit produced no segments.")
    covered = target_df.merge(
        pd.concat(parts, ignore_index=True),
        on=["date", "target_date"],
        how="inner",
        validate="one_to_one",
    )
    frame = make_refit_prediction_frame(
        covered,
        covered["pred_var"].to_numpy(dtype=float),
        model=model,
        protocol="expanding_monthly",
    )
    return frame, diagnostics, failures


def ewma_expanding_monthly_predictions(
    *,
    model_ready: pd.DataFrame,
    target_df: pd.DataFrame,
    lambda_value: float = 0.90,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    starts = refit_period_starts(target_df)
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for index, start_date in enumerate(starts):
        end_date = starts[index + 1] if index + 1 < len(starts) else pd.Timestamp.max
        segment = target_df[(target_df["date"] >= start_date) & (target_df["date"] < end_date)].copy()
        fit_df = model_ready[model_ready["date"] < start_date].copy()
        if segment.empty:
            continue
        start_timer = time.perf_counter()
        h_current = float(np.clip(fit_df["squared_return"].mean(), EPSILON, None))
        for value in fit_df["squared_return"].astype(float).to_numpy():
            h_current = float(np.clip(lambda_value * h_current + (1.0 - lambda_value) * value, EPSILON, None))
        preds = []
        for value in segment["squared_return"].astype(float).to_numpy():
            h_current = float(np.clip(lambda_value * h_current + (1.0 - lambda_value) * value, EPSILON, None))
            preds.append(h_current)
        part = segment[["date", "target_date"]].copy()
        part["pred_var"] = preds
        parts.append(part)
        diagnostics.append(
            {
                "model": "EWMA(lambda=0.90)",
                "protocol": "expanding_monthly",
                "segment_start": start_date.strftime("%Y-%m-%d"),
                "segment_end_exclusive": "end" if end_date == pd.Timestamp.max else end_date.strftime("%Y-%m-%d"),
                "status": "success",
                "fit_rows": int(len(fit_df)),
                "segment_rows": int(len(segment)),
                "runtime_seconds": float(time.perf_counter() - start_timer),
                "message": f"lambda={lambda_value:.2f}",
            }
        )
    predictions = target_df[["date", "target_date"]].merge(
        pd.concat(parts, ignore_index=True),
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    frame = make_refit_prediction_frame(
        target_df,
        predictions["pred_var"].to_numpy(dtype=float),
        model="EWMA(lambda=0.90)",
        protocol="expanding_monthly",
    )
    return frame, diagnostics, 0


def har_parkinson_expanding_monthly_predictions(
    *,
    model_ready: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    features = build_har_feature_frame(model_ready)
    feature_columns = ["parkinson_d", "parkinson_w", "parkinson_m"]
    starts = refit_period_starts(target_df)
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    failures = 0
    for index, start_date in enumerate(starts):
        end_date = starts[index + 1] if index + 1 < len(starts) else pd.Timestamp.max
        segment = target_df[(target_df["date"] >= start_date) & (target_df["date"] < end_date)].copy()
        if segment.empty:
            continue
        start_timer = time.perf_counter()
        try:
            fit_keys = model_ready.loc[model_ready["target_date"] < start_date, ["date", "target_date"]].copy()
            fit = fit_keys.merge(features, on=["date", "target_date"], how="left", validate="one_to_one")
            fit_df = fit.dropna(subset=[*feature_columns, "target_var_next"]).copy()
            x_train = np.column_stack([np.ones(len(fit_df)), fit_df[feature_columns].to_numpy(dtype=float)])
            y_train = fit_df["target_var_next"].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
            seg_features = segment[["date", "target_date"]].merge(
                features,
                on=["date", "target_date"],
                how="left",
                validate="one_to_one",
            )
            if seg_features[feature_columns].isna().any().any():
                raise ValueError("Missing HAR-Parkinson features in a refit segment.")
            x_segment = np.column_stack([np.ones(len(seg_features)), seg_features[feature_columns].to_numpy(dtype=float)])
            pred = np.clip(x_segment @ beta, EPSILON, None)
            part = segment[["date", "target_date"]].copy()
            part["pred_var"] = pred
            parts.append(part)
            diagnostics.append(
                {
                    "model": "HAR-Parkinson",
                    "protocol": "expanding_monthly",
                    "segment_start": start_date.strftime("%Y-%m-%d"),
                    "segment_end_exclusive": "end" if end_date == pd.Timestamp.max else end_date.strftime("%Y-%m-%d"),
                    "status": "success",
                    "fit_rows": int(len(fit_df)),
                    "segment_rows": int(len(segment)),
                    "runtime_seconds": float(time.perf_counter() - start_timer),
                    "message": json.dumps(
                        {"intercept": float(beta[0]), **{col: float(val) for col, val in zip(feature_columns, beta[1:])}},
                        sort_keys=True,
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            diagnostics.append(
                {
                    "model": "HAR-Parkinson",
                    "protocol": "expanding_monthly",
                    "segment_start": start_date.strftime("%Y-%m-%d"),
                    "segment_end_exclusive": "end" if end_date == pd.Timestamp.max else end_date.strftime("%Y-%m-%d"),
                    "status": "failure",
                    "fit_rows": 0,
                    "segment_rows": int(len(segment)),
                    "runtime_seconds": float(time.perf_counter() - start_timer),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    if failures:
        raise RuntimeError(f"HAR-Parkinson expanding-monthly refit failed for {failures} monthly segments.")
    predictions = target_df[["date", "target_date"]].merge(
        pd.concat(parts, ignore_index=True),
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    frame = make_refit_prediction_frame(
        target_df,
        predictions["pred_var"].to_numpy(dtype=float),
        model="HAR-Parkinson",
        protocol="expanding_monthly",
    )
    return frame, diagnostics, failures


def refit_var_summary(group: pd.DataFrame) -> dict[str, Any]:
    returns = target_returns()
    merged = group.merge(returns, on="target_date", how="left", validate="many_to_one").sort_values("target_date")
    out: dict[str, Any] = {}
    for alpha in [0.05, 0.025, 0.01]:
        z = NormalDist().inv_cdf(alpha)
        sigma = np.sqrt(np.clip(merged["pred_var"].to_numpy(dtype=float), EPSILON, None))
        violations = merged["target_return"].to_numpy(dtype=float) < z * sigma
        out[f"var_{alpha:g}_violations"] = int(violations.sum())
        out[f"var_{alpha:g}_expected"] = float(alpha * len(violations))
        out[f"var_{alpha:g}_kupiec_p_value"] = kupiec_test(violations, alpha)[1]
    return out


def refit_result_rows(predictions: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    actual = predictions.groupby(["date", "target_date"], as_index=False)["actual_var"].first()
    q90 = float(actual["actual_var"].quantile(0.90))
    expected_by_model = (
        predictions[predictions["protocol"].eq("static_train_only")]
        .groupby("model")
        .size()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for (model, protocol), group in predictions.groupby(["model", "protocol"], sort=True):
        metrics = volatility_metrics(group["actual_var"], group["pred_var"])
        spike = group[group["actual_var"] >= q90].copy()
        ratio = np.clip(spike["pred_var"], EPSILON, None) / np.clip(spike["actual_var"], EPSILON, None)
        diag = diagnostics[(diagnostics["model"].eq(model)) & (diagnostics["protocol"].eq(protocol))]
        failed = int(diag["status"].eq("failure").sum()) if not diag.empty else 0
        runtime = float(pd.to_numeric(diag["runtime_seconds"], errors="coerce").sum()) if not diag.empty else 0.0
        rows.append(
            {
                "model": model,
                "protocol": protocol,
                "origin_date_start": group["date"].min().strftime("%Y-%m-%d"),
                "origin_date_end": group["date"].max().strftime("%Y-%m-%d"),
                "target_date_start": group["target_date"].min().strftime("%Y-%m-%d"),
                "target_date_end": group["target_date"].max().strftime("%Y-%m-%d"),
                **metrics,
                "complete_common_window": int(metrics["n_obs"]) == int(expected_by_model.get(model, metrics["n_obs"])),
                "extreme_q90_threshold": q90,
                "extreme_q90_qlike": float(np.mean(qlike_loss(spike["actual_var"], spike["pred_var"]))) if not spike.empty else np.nan,
                "spike_severe_underprediction_rate": float((ratio < 0.5).mean()) if not spike.empty else np.nan,
                "failed_refits": failed,
                "runtime_seconds": runtime,
                **refit_var_summary(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "qlike", "model"]).reset_index(drop=True)


def build_refit_protocol_robustness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    _master, common, _specs = build_master_results()
    model_ready = load_model_ready_frame()
    common_keys = common[["date", "target_date"]].drop_duplicates().sort_values("date")
    target_df = common_keys.merge(
        model_ready,
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if target_df["target_var_next"].isna().any():
        raise ValueError("Common-window target rows are missing from model-ready data.")

    frames = static_refit_frames(common)
    diagnostics: list[dict[str, Any]] = []
    for model, _role in REFIT_MODEL_SPECS:
        if model in {"GARCH(1,1)", "ARIMA-GARCH", "AdvGARCH-BestQLIKE", "AdvGARCH-BestAsymmetric"}:
            frame, segment_diag, _failures = expanding_monthly_arch_predictions(
                model=model,
                model_ready=model_ready,
                target_df=target_df,
            )
        elif model == "EWMA(lambda=0.90)":
            frame, segment_diag, _failures = ewma_expanding_monthly_predictions(
                model_ready=model_ready,
                target_df=target_df,
            )
        elif model == "HAR-Parkinson":
            frame, segment_diag, _failures = har_parkinson_expanding_monthly_predictions(
                model_ready=model_ready,
                target_df=target_df,
            )
        else:
            continue
        frames.append(frame)
        diagnostics.extend(segment_diag)

    predictions = pd.concat(frames, ignore_index=True).sort_values(["protocol", "model", "date", "target_date"])
    diagnostics_df = pd.DataFrame(diagnostics)
    results = refit_result_rows(predictions, diagnostics_df)
    dm_rows = []
    for model in sorted(predictions["model"].unique()):
        static = predictions[(predictions["model"].eq(model)) & (predictions["protocol"].eq("static_train_only"))]
        expanding = predictions[(predictions["model"].eq(model)) & (predictions["protocol"].eq("expanding_monthly"))]
        if static.empty or expanding.empty:
            continue
        merged = expanding[["date", "target_date", "actual_var", "pred_var"]].rename(columns={"pred_var": "pred_expanding"}).merge(
            static[["date", "target_date", "pred_var"]].rename(columns={"pred_var": "pred_static"}),
            on=["date", "target_date"],
            how="inner",
            validate="one_to_one",
        )
        diff = loss_values(merged["actual_var"], merged["pred_expanding"], "qlike") - loss_values(
            merged["actual_var"], merged["pred_static"], "qlike"
        )
        stat, p_value, lag = dm_test(diff)
        dm_rows.append(
            {
                "model": model,
                "comparison": "expanding_monthly_minus_static_train_only",
                "loss_type": "qlike",
                "n_obs": int(len(diff)),
                "mean_loss_diff": float(np.mean(diff)),
                "dm_stat": stat,
                "p_value": p_value,
                "hac_lag": lag,
            }
        )
    diagnostics_out = pd.concat([diagnostics_df, pd.DataFrame(dm_rows)], ignore_index=True, sort=False)
    predictions.to_parquet(OUTPUT_FINAL_DIR / "refit_protocol_predictions.parquet", index=False)
    predictions.to_csv(OUTPUT_FINAL_DIR / "refit_protocol_predictions.csv", index=False)
    results.to_csv(OUTPUT_FINAL_DIR / "refit_protocol_results.csv", index=False)
    diagnostics_out.to_csv(OUTPUT_FINAL_DIR / "refit_protocol_diagnostics.csv", index=False)
    write_refit_latex(results, PAPER_TABLES_DIR / "refit_protocol_results.tex")
    plot_refit_qlike(results, PAPER_FIGURES_DIR / "fig_refit_protocol_qlike.png")
    return results, predictions, diagnostics_out


def write_refit_latex(results: pd.DataFrame, path: Path) -> None:
    display = results[results["protocol"].isin(["static_train_only", "expanding_monthly"])].copy()
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Static versus expanding-window monthly refit robustness on the unified common window.}",
        r"\label{tab:refit-protocol-results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}L{0.18\textwidth}L{0.12\textwidth}rrrrrrrr@{}}",
        r"\toprule",
        r"Model & Protocol & \(N\) & Full & QLIKE & RMSE & MAE & Pred./Actual & Q90 QLIKE & Failed \\",
        r"\midrule",
    ]
    for _, row in display.sort_values(["model", "protocol"]).iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['protocol'])} & {int(row['n_obs'])} & "
            f"{'yes' if bool(row['complete_common_window']) else 'no'} & "
            f"{float(row['qlike']):.6f} & {float(row['rmse']):.6f} & {float(row['mae']):.6f} & "
            f"{float(row['pred_actual_ratio']):.3f} & {float(row['extreme_q90_qlike']):.3f} & "
            f"{int(row['failed_refits'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_refit_qlike(results: pd.DataFrame, path: Path) -> None:
    pivot = results.pivot_table(index="model", columns="protocol", values="qlike", aggfunc="first")
    keep_cols = [col for col in ["static_train_only", "expanding_monthly"] if col in pivot.columns]
    pivot = pivot[keep_cols].sort_values(keep_cols[0])
    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, column in enumerate(keep_cols):
        ax.bar(x + (idx - 0.5) * width, pivot[column], width=width, label=column.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("QLIKE")
    ax.set_title("Static and Expanding-Monthly Refit QLIKE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def build_proxy_robustness(common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_df, dataset, _mapping = parse_price_source()
    if price_df is None:
        raise RuntimeError("No OHLC price source is available for proxy robustness.")
    proxies, invalid = construct_proxies(price_df)
    invalid.to_csv(OUTPUT_FINAL_DIR / "proxy_invalid_rows.csv", index=False)
    targets = proxies.rename(columns={"date": "target_date"})
    daily_proxies = ["close_to_close_squared_return", "parkinson", "garman_klass", "rogers_satchell"]
    rolling_proxies = ["yang_zhang_5", "yang_zhang_10", "yang_zhang_20"]
    rows = []
    for proxy in [*daily_proxies, *rolling_proxies]:
        merged = common.merge(targets[["target_date", proxy]], on="target_date", how="left")
        merged = merged.rename(columns={proxy: "proxy_actual_var"})
        merged = merged[np.isfinite(pd.to_numeric(merged["proxy_actual_var"], errors="coerce"))].copy()
        for model, group in merged.groupby("model"):
            metrics = volatility_metrics(group["proxy_actual_var"], group["pred_var"])
            rows.append(
                {
                    "proxy": proxy,
                    "proxy_group": "daily_one_day" if proxy in daily_proxies else "rolling_sensitivity",
                    "model": model,
                    "price_source": dataset,
                    **metrics,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics["qlike_rank"] = metrics.groupby("proxy")["qlike"].rank(method="min")
    daily = metrics[metrics["proxy_group"].eq("daily_one_day")].copy()
    rolling = metrics[metrics["proxy_group"].eq("rolling_sensitivity")].copy()
    daily.to_csv(OUTPUT_FINAL_DIR / "daily_proxy_robustness_metrics.csv", index=False)
    rolling.to_csv(OUTPUT_FINAL_DIR / "rolling_proxy_sensitivity_metrics.csv", index=False)
    plot_proxy_rank_heatmap(daily, PAPER_FIGURES_DIR / "fig_daily_proxy_rank_heatmap_primary.png")
    return daily, rolling


def plot_proxy_rank_heatmap(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    pivot = metrics.pivot_table(index="model", columns="proxy", values="qlike_rank", aggfunc="min")
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([col.replace("_", "\n") for col in pivot.columns], fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="QLIKE rank")
    ax.set_title("Daily Proxy QLIKE Ranks: Primary Models")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def write_repo_audit_report(master: pd.DataFrame | None = None) -> Path:
    ensure_dirs()
    raw = PROJECT_ROOT / "data" / "raw" / "vnindex_cafef_2010_2025_raw.csv"
    clean = DATA_DIR / "vnindex_cafef_2010_2025_clean.csv"
    model_ready = DATA_DIR / "vnindex_model_ready.csv"
    split_lines = []
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"
        df = read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        split_lines.append(
            f"- `{rel_path(path)}`: {len(df)} rows, origins {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}, "
            f"targets {df['target_date'].min():%Y-%m-%d} to {df['target_date'].max():%Y-%m-%d}."
        )
    taxonomy_path = OUTPUT_AUDIT_DIR / "advanced_garch_failure_taxonomy.csv"
    taxonomy_note = "not yet generated"
    if taxonomy_path.exists():
        taxonomy = read_csv(taxonomy_path)
        taxonomy_note = "; ".join(f"{row.failure_category}={int(row.count)}" for row in taxonomy.itertuples())
    master_note = "not yet generated"
    if master is not None and not master.empty:
        master_note = (
            f"{len(master)} models; common N={int(master['n_obs'].iloc[0])}; "
            f"target dates {master['target_date_start'].iloc[0]} to {master['target_date_end'].iloc[0]}."
        )
    lines = [
        "# Financial Econometrics Repository Audit",
        "",
        "## Repository Structure Relevant To The Research Pipeline",
        "",
        "- `data/raw/`: raw CafeF export.",
        "- `data/processed/`: clean data, model-ready data, and chronological splits.",
        "- `src/`: data preparation, model training, evaluation, robustness, and final audit scripts.",
        "- `outputs/predictions/`: original econometric and neural prediction files.",
        "- `outputs/advanced/`: advanced GARCH, calibration, combination, refit, hypothesis-test, and VaR artifacts.",
        "- `outputs/stage1_extensions/`: EWMA, HAR, spike, proxy, and common-Normal VaR extension artifacts.",
        "- `outputs/final/`: authoritative final common-window artifacts generated by `src/final_financial_econometrics.py`.",
        "- `paper/`: LaTeX source, tables, figures, and compiled manuscript.",
        "",
        "## Data Pipeline",
        "",
        f"- Raw data: `{rel_path(raw)}` ({'present' if raw.exists() else 'missing'}).",
        f"- Clean data: `{rel_path(clean)}` ({'present' if clean.exists() else 'missing'}).",
        f"- Model-ready data: `{rel_path(model_ready)}` ({'present' if model_ready.exists() else 'missing'}).",
        "- Return formula in `src/prepare_data.py`: `log_return_pct = 100 * log(close / close.shift(1))`.",
        "- Target formula in `src/prepare_data.py`: `target_var_next = squared_return.shift(-1)` and `target_date = date.shift(-1)`.",
        *split_lines,
        "",
        "## Modeling And Result Sources",
        "",
        "- Baselines/GARCH/ARIMA-GARCH: `src/train_econometric.py` and `outputs/predictions/*.csv`.",
        "- LSTM/hybrid/tuned neural: `src/train_lstm_hybrid.py`, `src/tune_lstm_hybrid.py`, and `outputs/predictions/lstm_tuned/*.csv`.",
        "- EWMA/HAR/risk diagnostics: `src/stage1_extensions.py` and `outputs/stage1_extensions/`.",
        "- Advanced GARCH search: `src/advanced_model_search.py` and `outputs/advanced/tables/table_garch_family_*.csv`.",
        "- Calibration/combination: `src/advanced_forecast_combinations.py` and `outputs/advanced/predictions/`.",
        "- Final common-window outputs: `src/final_financial_econometrics.py` and `outputs/final/`.",
        "",
        "## Current Final Output Status",
        "",
        f"- Master results: {master_note}",
        f"- Advanced GARCH failure taxonomy: {taxonomy_note}.",
        "",
        "## Suspected Or Confirmed Issues",
        "",
        "- Mixed test windows existed before final reconciliation: neural artifacts have 726 test rows, while many econometric artifacts have 745.",
        "- The neural row loss comes from split-local sequence construction; using pre-test historical context would be leakage-safe but requires retraining predictions.",
        "- Previous advanced-search summaries undercounted failure categories by reporting 144 non-finite forecasts while total failures were 576.",
        "- Current prediction artifacts do not retain distribution parameters needed for full model-specific VaR across heavy-tailed/skewed GARCH variants.",
        "- Existing `requirements.txt` did not list every optional package used by final outputs, such as `scipy` and `pyarrow`.",
        "",
        "## Reproduction Commands",
        "",
        "- `python -m pip install -r requirements.txt`",
        "- `PYTHONDONTWRITEBYTECODE=1 python run_experiment.py --skip-paper`",
        "- `PYTHONDONTWRITEBYTECODE=1 python src/advanced_model_search.py --force`",
        "- `PYTHONDONTWRITEBYTECODE=1 python src/run_advanced_experiments.py --force`",
        "- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py all`",
        "- `PYTHONDONTWRITEBYTECODE=1 python src/final_financial_econometrics.py all`",
        "- `PYTHONDONTWRITEBYTECODE=1 pytest -q`",
        "- `make -C paper all`",
        "",
        "## Prioritized Remaining Work",
        "",
        "1. Treat `outputs/final/common_window_master_results.csv` as the only primary leaderboard.",
        "2. Regenerate neural predictions with leakage-safe pre-test context if runtime permits; otherwise keep the 726-row common window.",
        "3. Rewrite manuscript claims to remove mixed-window comparisons and test-best winners from primary evidence.",
        "4. Add distribution-specific VaR only after fitted innovation parameters are persisted or regenerated.",
    ]
    path = REPORTS_DIR / "repo_audit_financial_econometrics.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_all() -> None:
    ensure_dirs()
    master, common, _specs = build_master_results()
    build_failure_taxonomy()
    build_descriptive_and_arch_diagnostics()
    build_garch_diagnostics()
    build_common_dm_tests(common)
    build_regime_and_spike(common)
    build_common_var(common)
    build_proxy_robustness(common)
    build_refit_protocol_robustness()
    write_repo_audit_report(master)
    print(f"Wrote final outputs under {rel_path(OUTPUT_FINAL_DIR)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final VN-Index financial econometrics artifacts.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "audit",
            "failure-taxonomy",
            "master-results",
            "diagnostics",
            "garch-diagnostics",
            "stat-tests",
            "risk",
            "proxy-robustness",
            "refit-robustness",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.command == "audit":
        master = read_csv(OUTPUT_FINAL_DIR / "common_window_master_results.csv") if (OUTPUT_FINAL_DIR / "common_window_master_results.csv").exists() else None
        path = write_repo_audit_report(master)
        print(f"Wrote {rel_path(path)}")
    elif args.command == "failure-taxonomy":
        taxonomy = build_failure_taxonomy()
        print(f"Wrote taxonomy with {int(taxonomy['count'].sum())} failures")
    elif args.command == "master-results":
        master, _common, _specs = build_master_results()
        print(f"Wrote master results for {len(master)} models")
    elif args.command == "diagnostics":
        build_descriptive_and_arch_diagnostics()
        print("Wrote descriptive and ARCH diagnostics")
    elif args.command == "garch-diagnostics":
        build_garch_diagnostics()
        print("Wrote GARCH parameter and residual diagnostics")
    elif args.command == "stat-tests":
        _master, common, _specs = build_master_results()
        build_common_dm_tests(common)
        print("Wrote common-window DM tests")
    elif args.command == "risk":
        _master, common, _specs = build_master_results()
        build_regime_and_spike(common)
        build_common_var(common)
        print("Wrote common-window risk diagnostics")
    elif args.command == "proxy-robustness":
        _master, common, _specs = build_master_results()
        build_proxy_robustness(common)
        print("Wrote proxy robustness outputs")
    elif args.command == "refit-robustness":
        build_refit_protocol_robustness()
        print("Wrote refit protocol robustness outputs")
    else:
        build_all()


if __name__ == "__main__":
    main()
