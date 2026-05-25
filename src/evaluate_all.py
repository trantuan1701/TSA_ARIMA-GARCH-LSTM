#!/usr/bin/env python3
"""Final Stage 4 evaluation for VN-Index volatility forecasts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from metrics import EPSILON, evaluate_volatility_predictions
except ImportError:  # pragma: no cover - fallback for unusual execution contexts.
    EPSILON = 1e-8

    def evaluate_volatility_predictions(
        actual_var: pd.Series | np.ndarray | list[float],
        pred_var: pd.Series | np.ndarray | list[float],
        *,
        epsilon: float = EPSILON,
        label: str | None = None,
    ) -> dict[str, float | int]:
        actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
        if actual.shape[0] != pred.shape[0]:
            raise ValueError(
                f"actual_var and pred_var must have equal length; got {actual.shape[0]} "
                f"and {pred.shape[0]}."
            )
        finite_mask = np.isfinite(actual) & np.isfinite(pred)
        dropped = int((~finite_mask).sum())
        if dropped:
            metric_label = f" for {label}" if label else ""
            logging.warning(
                "Dropping %s non-finite volatility prediction rows%s before metric computation.",
                dropped,
                metric_label,
            )
        actual = actual[finite_mask]
        pred = pred[finite_mask]
        if actual.size == 0:
            raise ValueError("No finite observations remain for volatility metric computation.")
        pred_clipped = np.clip(pred, epsilon, None)
        errors = actual - pred_clipped
        mse = float(np.mean(errors**2))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(errors))),
            "qlike": float(np.mean(np.log(pred_clipped) + actual / pred_clipped)),
            "n_obs": int(actual.size),
            "mean_actual_var": float(np.mean(actual)),
            "mean_pred_var": float(np.mean(pred_clipped)),
        }


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
TUNED_LSTM_PREDICTIONS_DIR = PREDICTIONS_DIR / "lstm_tuned"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
KEY_COLUMNS = ["split", "date", "target_date"]
VALID_SPLITS = ["validation", "test"]

TUNED_LSTM_MODELS = [
    "LSTM-LogTarget",
    "Hybrid-LogTarget",
    "LSTM-QLIKE",
    "Hybrid-QLIKE",
    "LSTM-LogTarget-Small",
    "Hybrid-LogTarget-Small",
]

CANONICAL_MODELS = [
    "HistoricalMean",
    "RollingVol-5",
    "RollingVol-10",
    "RollingVol-20",
    "GARCH(1,1)",
    "ARIMA-GARCH",
    "LSTM",
    "ARIMA-GARCH-LSTM",
    *TUNED_LSTM_MODELS,
]

MODEL_FILE_SPECS = [
    ("HistoricalMean", "pred_baseline_mean.csv"),
    ("RollingVol-5", "pred_rolling_vol_5.csv"),
    ("RollingVol-10", "pred_rolling_vol_10.csv"),
    ("RollingVol-20", "pred_rolling_vol_20.csv"),
    ("GARCH(1,1)", "pred_garch_11.csv"),
    ("ARIMA-GARCH", "pred_arima_garch.csv"),
    ("LSTM", "pred_lstm_base.csv"),
    ("ARIMA-GARCH-LSTM", "pred_lstm_hybrid.csv"),
]

MODEL_TO_WIDE_COLUMN = {
    "HistoricalMean": "pred_HistoricalMean",
    "RollingVol-5": "pred_RollingVol_5",
    "RollingVol-10": "pred_RollingVol_10",
    "RollingVol-20": "pred_RollingVol_20",
    "GARCH(1,1)": "pred_GARCH_11",
    "ARIMA-GARCH": "pred_ARIMA_GARCH",
    "LSTM": "pred_LSTM",
    "ARIMA-GARCH-LSTM": "pred_ARIMA_GARCH_LSTM",
    "LSTM-LogTarget": "pred_LSTM_LogTarget",
    "Hybrid-LogTarget": "pred_Hybrid_LogTarget",
    "LSTM-QLIKE": "pred_LSTM_QLIKE",
    "Hybrid-QLIKE": "pred_Hybrid_QLIKE",
    "LSTM-LogTarget-Small": "pred_LSTM_LogTarget_Small",
    "Hybrid-LogTarget-Small": "pred_Hybrid_LogTarget_Small",
}

METRIC_COLUMNS = [
    "model",
    "split",
    "sample_type",
    "n_obs",
    "mse",
    "rmse",
    "mae",
    "qlike",
    "mean_actual_var",
    "mean_pred_var",
]

PAPER_COLUMNS = ["Rank", "Model", "RMSE", "MAE", "QLIKE", "N"]


def alias_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


MODEL_ALIASES = {
    alias_key(alias): canonical
    for canonical, aliases in {
        "HistoricalMean": [
            "HistoricalMean",
            "Historical Mean",
            "historical_mean",
            "baseline_mean",
            "BaselineMean",
            "mean",
        ],
        "RollingVol-5": [
            "RollingVol-5",
            "rolling_vol_5",
            "RollingVol5",
            "Rolling Vol 5",
            "rolling volatility 5",
        ],
        "RollingVol-10": [
            "RollingVol-10",
            "rolling_vol_10",
            "RollingVol10",
            "Rolling Vol 10",
            "rolling volatility 10",
        ],
        "RollingVol-20": [
            "RollingVol-20",
            "rolling_vol_20",
            "RollingVol20",
            "Rolling Vol 20",
            "rolling volatility 20",
        ],
        "GARCH(1,1)": ["GARCH(1,1)", "GARCH(1, 1)", "garch_11", "GARCH11", "GARCH 1 1"],
        "ARIMA-GARCH": ["ARIMA-GARCH", "arima_garch", "ARIMAGARCH", "ARIMA GARCH"],
        "LSTM": ["LSTM", "lstm_base", "base_lstm", "LSTM Base", "pred_lstm_base"],
        "ARIMA-GARCH-LSTM": [
            "ARIMA-GARCH-LSTM",
            "arima_garch_lstm",
            "ARIMAGARCHLSTM",
            "LSTM Hybrid",
            "hybrid_lstm",
            "lstm_hybrid",
            "pred_lstm_hybrid",
        ],
        "LSTM-LogTarget": ["LSTM-LogTarget", "lstm_logtarget", "pred_lstm_logtarget"],
        "Hybrid-LogTarget": ["Hybrid-LogTarget", "hybrid_logtarget", "pred_hybrid_logtarget"],
        "LSTM-QLIKE": ["LSTM-QLIKE", "lstm_qlike", "pred_lstm_qlike"],
        "Hybrid-QLIKE": ["Hybrid-QLIKE", "hybrid_qlike", "pred_hybrid_qlike"],
        "LSTM-LogTarget-Small": [
            "LSTM-LogTarget-Small",
            "lstm_logtarget_small",
            "pred_lstm_logtarget_small",
        ],
        "Hybrid-LogTarget-Small": [
            "Hybrid-LogTarget-Small",
            "hybrid_logtarget_small",
            "pred_hybrid_logtarget_small",
        ],
    }.items()
    for alias in aliases
}


def rel_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"WARNING: {message}")


def ensure_dirs() -> None:
    for path in (METRICS_DIR, TABLES_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ordered_models(models: pd.Series | list[str] | np.ndarray) -> list[str]:
    unique_models = sorted(set(str(model) for model in models))
    order = {model: index for index, model in enumerate(CANONICAL_MODELS)}
    return sorted(unique_models, key=lambda model: (order.get(model, len(order)), model))


def normalize_model_name(model_name: object, source_path: Path, warnings: list[str]) -> str:
    if pd.isna(model_name) or not str(model_name).strip():
        add_warning(
            warnings,
            f"{rel_path(source_path)} has a blank model name; keeping value as UnknownModel.",
        )
        return "UnknownModel"

    raw_name = str(model_name).strip()
    normalized = MODEL_ALIASES.get(alias_key(raw_name))
    if normalized:
        return normalized

    add_warning(
        warnings,
        f"{rel_path(source_path)} contains unknown model name {raw_name!r}; keeping original name.",
    )
    return raw_name


def validate_prediction_schema(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    missing = [column for column in PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{rel_path(source_path)} is missing required prediction columns: {missing}. "
            f"Required logical schema: {PREDICTION_COLUMNS}"
        )
    return df[PREDICTION_COLUMNS].copy()


def parse_prediction_frame(
    df: pd.DataFrame,
    source_path: Path,
    warnings: list[str],
) -> pd.DataFrame:
    parsed = validate_prediction_schema(df, source_path)

    for column in ("date", "target_date"):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        bad_dates = int(parsed[column].isna().sum())
        if bad_dates:
            raise ValueError(
                f"{rel_path(source_path)} has {bad_dates} rows with unparseable {column} values."
            )

    for column in ("actual_var", "pred_var"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")

    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(parsed.loc[~parsed["split"].isin(VALID_SPLITS), "split"]))
    if invalid_splits:
        raise ValueError(
            f"{rel_path(source_path)} has invalid split values {invalid_splits}; "
            f"allowed values are {VALID_SPLITS}."
        )

    parsed["model"] = parsed["model"].map(lambda value: normalize_model_name(value, source_path, warnings))
    parsed["_source_file"] = rel_path(source_path)
    parsed = parsed.sort_values(["date", "target_date", "model", "split"]).reset_index(drop=True)

    duplicate_mask = parsed.duplicated(["model", "split", "date", "target_date"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        add_warning(
            warnings,
            f"{rel_path(source_path)} has {duplicate_count} duplicate model/split/date/target_date rows; "
            "keeping the first occurrence after date sorting.",
        )
        parsed = parsed.loc[~duplicate_mask].reset_index(drop=True)

    return parsed


def load_prediction_files(warnings: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    found_paths: list[Path] = []

    for expected_model, file_name in MODEL_FILE_SPECS:
        path = PREDICTIONS_DIR / file_name
        if not path.exists():
            add_warning(warnings, f"Missing prediction file for {expected_model}: {rel_path(path)}")
            continue

        print(f"Found prediction file: {rel_path(path)}")
        raw = pd.read_csv(path, encoding="utf-8-sig")
        parsed = parse_prediction_frame(raw, path, warnings)
        found_models = ordered_models(parsed["model"].unique())
        if expected_model not in found_models:
            add_warning(
                warnings,
                f"{rel_path(path)} is expected to contain {expected_model}, "
                f"but contains {found_models}.",
            )
        frames.append(parsed)
        found_paths.append(path)

    tuned_paths = sorted(TUNED_LSTM_PREDICTIONS_DIR.glob("pred_*.csv"))
    for path in tuned_paths:
        print(f"Found tuned LSTM prediction file: {rel_path(path)}")
        raw = pd.read_csv(path, encoding="utf-8-sig")
        parsed = parse_prediction_frame(raw, path, warnings)
        frames.append(parsed)
        found_paths.append(path)

    if not frames:
        raise FileNotFoundError(f"No prediction files were found in {rel_path(PREDICTIONS_DIR)}.")

    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "target_date", "model", "split"]
    ).reset_index(drop=True)
    duplicate_mask = predictions.duplicated(["model", "split", "date", "target_date"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        add_warning(
            warnings,
            f"Combined predictions contain {duplicate_count} duplicate model/split/date/target_date rows; "
            "keeping the first occurrence after date sorting.",
        )
        predictions = predictions.loc[~duplicate_mask].reset_index(drop=True)

    models = ordered_models(predictions["model"].unique())
    if len(models) < 2:
        raise ValueError(f"At least two models are required for comparison; found {models}.")

    return predictions, found_paths


def load_split_targets(warnings: list[str]) -> pd.DataFrame:
    target_frames: list[pd.DataFrame] = []
    required_columns = ["date", "target_date", "target_var_next"]

    for split in VALID_SPLITS:
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            add_warning(warnings, f"Cannot validate actual_var because split file is missing: {rel_path(path)}")
            continue

        split_df = pd.read_csv(path, encoding="utf-8-sig")
        missing = [column for column in required_columns if column not in split_df.columns]
        if missing:
            add_warning(
                warnings,
                f"Cannot validate actual_var for {split} because {rel_path(path)} is missing {missing}.",
            )
            continue

        split_targets = split_df[required_columns].copy()
        split_targets["split"] = split
        for column in ("date", "target_date"):
            split_targets[column] = pd.to_datetime(split_targets[column], errors="coerce")
            bad_dates = int(split_targets[column].isna().sum())
            if bad_dates:
                raise ValueError(f"{rel_path(path)} has {bad_dates} unparseable {column} values.")
        split_targets["target_var_next"] = pd.to_numeric(
            split_targets["target_var_next"],
            errors="coerce",
        )
        target_frames.append(split_targets[["split", "date", "target_date", "target_var_next"]])

    if not target_frames:
        return pd.DataFrame(columns=["split", "date", "target_date", "target_var_next"])

    targets = pd.concat(target_frames, ignore_index=True)
    duplicate_mask = targets.duplicated(KEY_COLUMNS, keep="first")
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        add_warning(
            warnings,
            f"Split target files contain {duplicate_count} duplicate split/date/target_date rows; "
            "keeping the first occurrence for validation.",
        )
        targets = targets.loc[~duplicate_mask].reset_index(drop=True)
    return targets


def validate_actual_targets(predictions: pd.DataFrame, warnings: list[str]) -> None:
    targets = load_split_targets(warnings)
    if targets.empty:
        add_warning(warnings, "Skipping actual_var validation because no usable split target files were loaded.")
        return

    validated_splits = set(targets["split"].unique())
    predictions_to_check = predictions[predictions["split"].isin(validated_splits)].copy()
    merged = predictions_to_check.merge(targets, on=KEY_COLUMNS, how="left", indicator=True)

    missing_matches = merged["_merge"].eq("left_only")
    if missing_matches.any():
        examples = merged.loc[missing_matches, ["model", "split", "date", "target_date"]].head(5)
        raise ValueError(
            "Some predictions could not be matched to Stage 1 split targets for actual_var validation:\n"
            f"{examples.to_string(index=False)}"
        )

    actual = merged["actual_var"].to_numpy(dtype=float)
    target = merged["target_var_next"].to_numpy(dtype=float)
    both_nan = np.isnan(actual) & np.isnan(target)
    matches = np.isclose(actual, target, rtol=1e-10, atol=1e-12) | both_nan
    if not bool(matches.all()):
        examples = merged.loc[
            ~matches,
            ["model", "split", "date", "target_date", "actual_var", "target_var_next"],
        ].head(5)
        raise ValueError(
            "Prediction actual_var must equal target_var_next, but mismatches were found:\n"
            f"{examples.to_string(index=False)}"
        )


def row_counts_by_model_split(predictions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        predictions.groupby(["model", "split"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    counts["model_order"] = counts["model"].map({model: index for index, model in enumerate(CANONICAL_MODELS)})
    counts["model_order"] = counts["model_order"].fillna(len(CANONICAL_MODELS)).astype(int)
    counts["split_order"] = counts["split"].map({split: index for index, split in enumerate(VALID_SPLITS)})
    counts = counts.sort_values(["model_order", "split_order", "model"]).drop(
        columns=["model_order", "split_order"]
    )
    return counts.reset_index(drop=True)


def evaluate_predictions(df: pd.DataFrame, model: str, split: str, sample_type: str) -> dict[str, float | int | str]:
    metrics = evaluate_volatility_predictions(
        df["actual_var"],
        df["pred_var"],
        epsilon=EPSILON,
        label=f"{model}/{split}/{sample_type}",
    )
    return {
        "model": model,
        "split": split,
        "sample_type": sample_type,
        "n_obs": int(metrics["n_obs"]),
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "qlike": float(metrics["qlike"]),
        "mean_actual_var": float(metrics["mean_actual_var"]),
        "mean_pred_var": float(metrics["mean_pred_var"]),
    }


def compute_available_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split in VALID_SPLITS:
        split_df = predictions[predictions["split"] == split]
        for model in ordered_models(split_df["model"].unique()):
            model_df = split_df[split_df["model"] == model]
            if not model_df.empty:
                rows.append(evaluate_predictions(model_df, model, split, "available"))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def get_common_keys(
    predictions: pd.DataFrame,
    split: str,
    warnings: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    split_df = predictions[predictions["split"] == split]
    models = ordered_models(split_df["model"].unique())
    if len(models) < 2:
        add_warning(warnings, f"Skipping common-window evaluation for {split}: fewer than two models available.")
        return pd.DataFrame(columns=["date", "target_date"]), models

    key_sets: list[set[tuple[pd.Timestamp, pd.Timestamp]]] = []
    for model in models:
        model_keys = set(
            split_df.loc[split_df["model"] == model, ["date", "target_date"]].itertuples(index=False, name=None)
        )
        key_sets.append(model_keys)

    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError(f"No common date/target_date window exists for split {split}.")

    common_key_df = pd.DataFrame(sorted(common_keys), columns=["date", "target_date"])
    common_key_df = common_key_df.sort_values(["date", "target_date"]).reset_index(drop=True)
    return common_key_df, models


def filter_common_predictions(
    predictions: pd.DataFrame,
    split: str,
    common_keys: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    key_df = common_keys.copy()
    key_df["split"] = split
    filtered = predictions[
        (predictions["split"] == split) & (predictions["model"].isin(models))
    ].merge(key_df, on=["split", "date", "target_date"], how="inner")
    filtered = filtered.sort_values(["date", "target_date", "model"]).reset_index(drop=True)

    counts = filtered.groupby("model").size()
    expected_count = len(common_keys)
    bad_counts = counts[counts != expected_count]
    if not bad_counts.empty:
        raise ValueError(
            f"Common-window filtering for split {split} produced unequal row counts:\n"
            f"{bad_counts.to_string()}"
        )
    return filtered


def compute_common_metrics(
    predictions: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, int]]:
    rows: list[dict[str, float | int | str]] = []
    common_predictions: dict[str, pd.DataFrame] = {}
    common_counts: dict[str, int] = {}

    for split in VALID_SPLITS:
        common_keys, models = get_common_keys(predictions, split, warnings)
        if common_keys.empty:
            continue
        split_common = filter_common_predictions(predictions, split, common_keys, models)
        common_predictions[split] = split_common
        common_counts[split] = len(common_keys)

        for model in models:
            model_df = split_common[split_common["model"] == model]
            rows.append(evaluate_predictions(model_df, model, split, "common"))

    if not rows:
        raise ValueError("No common-window metrics could be computed.")

    return pd.DataFrame(rows, columns=METRIC_COLUMNS), common_predictions, common_counts


def sort_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["split"] == "test"].sort_values(["qlike", "rmse", "model"]).reset_index(drop=True)


def save_metric_outputs(available_metrics: pd.DataFrame, common_metrics: pd.DataFrame) -> dict[str, Path]:
    paths = {
        "available": METRICS_DIR / "final_model_comparison_available.csv",
        "available_test": METRICS_DIR / "final_model_comparison_available_test_only.csv",
        "common": METRICS_DIR / "final_model_comparison_common.csv",
        "common_test": METRICS_DIR / "final_model_comparison_common_test_only.csv",
    }
    available_metrics.to_csv(paths["available"], index=False)
    sort_test_metrics(available_metrics).to_csv(paths["available_test"], index=False)
    common_metrics.to_csv(paths["common"], index=False)
    sort_test_metrics(common_metrics).to_csv(paths["common_test"], index=False)
    return paths


def safe_prediction_column(model: str) -> str:
    if model in MODEL_TO_WIDE_COLUMN:
        return MODEL_TO_WIDE_COLUMN[model]
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
    return f"pred_{sanitized}"


def build_wide_predictions(common_split_predictions: pd.DataFrame, split: str, warnings: list[str]) -> pd.DataFrame:
    if common_split_predictions.empty:
        return pd.DataFrame()

    actual_range = common_split_predictions.groupby(["date", "target_date"])["actual_var"].agg(["min", "max"])
    inconsistent_actual = ~np.isclose(
        actual_range["min"].to_numpy(dtype=float),
        actual_range["max"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    if bool(inconsistent_actual.any()):
        add_warning(
            warnings,
            f"{split} common window has {int(inconsistent_actual.sum())} keys with inconsistent actual_var values; "
            "using the first actual_var in the wide table.",
        )

    actual = (
        common_split_predictions.groupby(["date", "target_date"], as_index=False)["actual_var"]
        .first()
        .sort_values(["date", "target_date"])
    )
    pred_wide = common_split_predictions.pivot(
        index=["date", "target_date"],
        columns="model",
        values="pred_var",
    ).reset_index()
    pred_wide.columns.name = None

    rename_map = {model: safe_prediction_column(model) for model in pred_wide.columns if model not in ["date", "target_date"]}
    pred_wide = pred_wide.rename(columns=rename_map)
    wide = actual.merge(pred_wide, on=["date", "target_date"], how="left")

    ordered_columns = ["date", "target_date", "actual_var"]
    for model in CANONICAL_MODELS:
        column = MODEL_TO_WIDE_COLUMN.get(model)
        if column in wide.columns:
            ordered_columns.append(column)
    extra_prediction_columns = [
        column for column in wide.columns if column.startswith("pred_") and column not in ordered_columns
    ]
    ordered_columns.extend(sorted(extra_prediction_columns))
    return wide[ordered_columns].sort_values(["date", "target_date"]).reset_index(drop=True)


def save_wide_predictions(
    common_predictions: dict[str, pd.DataFrame],
    warnings: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in VALID_SPLITS:
        path = METRICS_DIR / f"predictions_{split}_common_wide.csv"
        wide = build_wide_predictions(common_predictions.get(split, pd.DataFrame()), split, warnings)
        wide.to_csv(path, index=False)
        paths[split] = path
    return paths


def format_paper_table(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.sort_values(["qlike", "rmse", "model"]).reset_index(drop=True).copy()
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    table = table.rename(
        columns={
            "model": "Model",
            "rmse": "RMSE",
            "mae": "MAE",
            "qlike": "QLIKE",
            "n_obs": "N",
        }
    )
    table = table[["Rank", "Model", "RMSE", "MAE", "QLIKE", "N"]]
    formatted = table.copy()
    for column in ["RMSE", "MAE", "QLIKE"]:
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.6f}")
    formatted["Rank"] = formatted["Rank"].astype(int).astype(str)
    formatted["N"] = formatted["N"].astype(int).astype(str)
    return formatted


def save_markdown_table(table: pd.DataFrame, path: Path) -> None:
    rows = [PAPER_COLUMNS] + table[PAPER_COLUMNS].astype(str).values.tolist()
    widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(PAPER_COLUMNS))]

    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(PAPER_COLUMNS)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(PAPER_COLUMNS))) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in table[PAPER_COLUMNS].astype(str).values.tolist()
    ]
    path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")


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
    return "".join(replacements.get(character, character) for character in text)


def save_latex_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{r l r r r r}",
        r"\hline",
        r"Rank & Model & RMSE & MAE & QLIKE & N \\",
        r"\hline",
    ]
    for _, row in table[PAPER_COLUMNS].iterrows():
        values = [latex_escape(row[column]) for column in PAPER_COLUMNS]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_paper_tables(available_test: pd.DataFrame, common_test: pd.DataFrame) -> dict[str, Path]:
    paths = {
        "common_md": TABLES_DIR / "table_model_comparison_common.md",
        "common_tex": TABLES_DIR / "table_model_comparison_common.tex",
        "available_md": TABLES_DIR / "table_model_comparison_available.md",
        "available_tex": TABLES_DIR / "table_model_comparison_available.tex",
    }
    common_table = format_paper_table(common_test)
    available_table = format_paper_table(available_test)
    save_markdown_table(common_table, paths["common_md"])
    save_latex_table(common_table, paths["common_tex"])
    save_markdown_table(available_table, paths["available_md"])
    save_latex_table(available_table, paths["available_tex"])
    return paths


def plot_metric_bar(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    plot_df = metrics.sort_values([metric, "model"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(plot_df)), plot_df[metric].to_numpy(dtype=float), color="#3B6EA8")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["model"], rotation=35, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Common Test {metric.upper()} Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_actual_vs_predicted(wide: pd.DataFrame, path: Path, limit: int | None = None) -> None:
    if wide.empty:
        raise ValueError("Cannot plot actual vs predicted volatility because the common test wide table is empty.")

    plot_df = wide.tail(limit).copy() if limit is not None else wide.copy()
    selected_models = ["GARCH(1,1)", "ARIMA-GARCH", "LSTM", "ARIMA-GARCH-LSTM"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plot_df["target_date"],
        plot_df["actual_var"],
        label="Actual variance",
        color="#111111",
        linewidth=1.2,
    )
    colors = {
        "GARCH(1,1)": "#1F77B4",
        "ARIMA-GARCH": "#FF7F0E",
        "LSTM": "#2CA02C",
        "ARIMA-GARCH-LSTM": "#D62728",
    }
    for model in selected_models:
        column = MODEL_TO_WIDE_COLUMN[model]
        if column in plot_df.columns:
            ax.plot(
                plot_df["target_date"],
                plot_df[column],
                label=model,
                linewidth=1.0,
                alpha=0.85,
                color=colors[model],
            )

    ax.set_xlabel("Target date")
    ax.set_ylabel("Variance proxy")
    title_suffix = f" Last {len(plot_df)} Observations" if limit is not None else ""
    ax.set_title(f"Actual vs Predicted Volatility on Common Test Window{title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_figures(common_test: pd.DataFrame, test_wide: pd.DataFrame) -> dict[str, Path]:
    paths = {
        "qlike": FIGURES_DIR / "fig_qlike_comparison_common_test.png",
        "rmse": FIGURES_DIR / "fig_rmse_comparison_common_test.png",
        "mae": FIGURES_DIR / "fig_mae_comparison_common_test.png",
        "actual_pred": FIGURES_DIR / "fig_actual_vs_predicted_test_common.png",
        "actual_pred_zoom": FIGURES_DIR / "fig_actual_vs_predicted_test_common_zoom.png",
    }
    plot_metric_bar(common_test, "qlike", paths["qlike"])
    plot_metric_bar(common_test, "rmse", paths["rmse"])
    plot_metric_bar(common_test, "mae", paths["mae"])
    plot_actual_vs_predicted(test_wide, paths["actual_pred"], limit=None)
    plot_actual_vs_predicted(test_wide, paths["actual_pred_zoom"], limit=min(150, len(test_wide)))
    return paths


def best_model(metrics: pd.DataFrame, metric: str) -> tuple[str, float]:
    row = metrics.sort_values([metric, "model"]).iloc[0]
    return str(row["model"]), float(row[metric])


def comparison_line(metrics: pd.DataFrame, candidate: str, baseline: str, metric: str) -> str:
    candidate_rows = metrics.loc[metrics["model"] == candidate, metric]
    baseline_rows = metrics.loc[metrics["model"] == baseline, metric]
    label = metric.upper()
    if candidate_rows.empty or baseline_rows.empty:
        return f"{label}: not available because one or both models are missing"

    candidate_value = float(candidate_rows.iloc[0])
    baseline_value = float(baseline_rows.iloc[0])
    if candidate_value < baseline_value:
        verdict = "improves"
    elif np.isclose(candidate_value, baseline_value, rtol=1e-12, atol=1e-12):
        verdict = "ties"
    else:
        verdict = "does not improve"
    return f"{label}: {verdict} ({candidate_value:.6f} vs {baseline_value:.6f})"


def write_evaluation_summary(
    path: Path,
    found_paths: list[Path],
    models: list[str],
    row_counts: pd.DataFrame,
    common_counts: dict[str, int],
    common_test: pd.DataFrame,
    warnings: list[str],
) -> None:
    best_qlike_model, best_qlike_value = best_model(common_test, "qlike")
    best_rmse_model, best_rmse_value = best_model(common_test, "rmse")
    best_mae_model, best_mae_value = best_model(common_test, "mae")

    lines = [
        "Final Evaluation Summary",
        "========================",
        "",
        "Input prediction files found:",
        *[f"- {rel_path(file_path)}" for file_path in found_paths],
        "",
        "Models included:",
        *[f"- {model}" for model in models],
        "",
        "Available-sample row counts by model/split:",
        row_counts.to_string(index=False),
        "",
        "Common-window row counts by split:",
    ]
    lines.extend([f"- {split}: {count}" for split, count in common_counts.items()])
    lines.extend(
        [
            "",
            f"Best model by QLIKE on common test sample: {best_qlike_model} ({best_qlike_value:.6f})",
            f"Best model by RMSE on common test sample: {best_rmse_model} ({best_rmse_value:.6f})",
            f"Best model by MAE on common test sample: {best_mae_model} ({best_mae_value:.6f})",
            "",
            "ARIMA-GARCH-LSTM versus LSTM on common test sample:",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'LSTM', 'qlike')}",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'LSTM', 'rmse')}",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'LSTM', 'mae')}",
            "",
            "ARIMA-GARCH-LSTM versus GARCH(1,1) on common test sample:",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'GARCH(1,1)', 'qlike')}",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'GARCH(1,1)', 'rmse')}",
            f"- {comparison_line(common_test, 'ARIMA-GARCH-LSTM', 'GARCH(1,1)', 'mae')}",
            "",
            "Caution: use the common-window metrics for the main paper comparison because all models are "
            "evaluated on the same date/target_date observations.",
        ]
    )
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {message}" for message in warnings]])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_final_console_summary(
    found_paths: list[Path],
    models: list[str],
    row_counts: pd.DataFrame,
    common_counts: dict[str, int],
    common_test: pd.DataFrame,
    saved_paths: list[Path],
    warnings: list[str],
) -> None:
    print("\nPrediction files found:")
    for path in found_paths:
        print(f"- {rel_path(path)}")

    print("\nModels included:")
    for model in models:
        print(f"- {model}")

    print("\nRow counts per model/split:")
    print(row_counts.to_string(index=False))

    print("\nCommon-window row counts:")
    for split, count in common_counts.items():
        print(f"- {split}: {count}")

    display_table = common_test[["model", "n_obs", "rmse", "mae", "qlike"]].copy()
    print("\nFinal common test comparison table:")
    print(display_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    for metric in ["qlike", "rmse", "mae"]:
        model, value = best_model(common_test, metric)
        print(f"Best model by {metric.upper()}: {model} ({value:.6f})")

    print("\nSaved output paths:")
    for path in saved_paths:
        print(f"- {rel_path(path)}")

    if warnings:
        print("\nWarnings:")
        for message in warnings:
            print(f"- {message}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ensure_dirs()

    warnings: list[str] = []
    predictions, found_paths = load_prediction_files(warnings)
    validate_actual_targets(predictions, warnings)

    models = ordered_models(predictions["model"].unique())
    row_counts = row_counts_by_model_split(predictions)
    print("\nAvailable prediction row counts:")
    print(row_counts.to_string(index=False))

    available_metrics = compute_available_metrics(predictions)
    common_metrics, common_predictions, common_counts = compute_common_metrics(predictions, warnings)

    metric_paths = save_metric_outputs(available_metrics, common_metrics)
    wide_paths = save_wide_predictions(common_predictions, warnings)

    available_test = sort_test_metrics(available_metrics)
    common_test = sort_test_metrics(common_metrics)
    table_paths = save_paper_tables(available_test, common_test)

    test_wide = pd.read_csv(wide_paths["test"], parse_dates=["date", "target_date"])
    figure_paths = save_figures(common_test, test_wide)

    summary_path = METRICS_DIR / "final_evaluation_summary.txt"
    write_evaluation_summary(
        summary_path,
        found_paths,
        models,
        row_counts,
        common_counts,
        common_test,
        warnings,
    )

    saved_paths = [
        metric_paths["available"],
        metric_paths["available_test"],
        metric_paths["common"],
        metric_paths["common_test"],
        wide_paths["test"],
        wide_paths["validation"],
        summary_path,
        table_paths["common_md"],
        table_paths["common_tex"],
        table_paths["available_md"],
        table_paths["available_tex"],
        figure_paths["qlike"],
        figure_paths["rmse"],
        figure_paths["mae"],
        figure_paths["actual_pred"],
        figure_paths["actual_pred_zoom"],
    ]
    print_final_console_summary(
        found_paths,
        models,
        row_counts,
        common_counts,
        common_test,
        saved_paths,
        warnings,
    )


if __name__ == "__main__":
    main()
