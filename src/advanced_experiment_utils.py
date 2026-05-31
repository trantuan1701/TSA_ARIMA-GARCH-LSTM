#!/usr/bin/env python3
"""Shared helpers for advanced VN-Index volatility experiments.

The advanced layer is intentionally isolated from the original experiment
outputs.  Scripts in this layer may read the original predictions and split
files, but all new artifacts are written below ``outputs/advanced``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover
    from .metrics import EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def optional_project_path(env_name: str) -> Path | None:
    raw = os.environ.get(env_name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


ORIGINAL_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
NEURAL_OUTPUT_ROOT = optional_project_path("VNINDEX_NEURAL_OUTPUT_ROOT")
NEURAL_PREDICTIONS_DIR = (
    NEURAL_OUTPUT_ROOT / "predictions" if NEURAL_OUTPUT_ROOT is not None else ORIGINAL_PREDICTIONS_DIR
)
ADVANCED_DIR = optional_project_path("VNINDEX_ADVANCED_DIR") or PROJECT_ROOT / "outputs" / "advanced"
ADVANCED_PREDICTIONS_DIR = ADVANCED_DIR / "predictions"
ADVANCED_TABLES_DIR = ADVANCED_DIR / "tables"
ADVANCED_FIGURES_DIR = ADVANCED_DIR / "figures"
ADVANCED_AUDIT_DIR = ADVANCED_DIR / "audit"
ADVANCED_MODELS_DIR = ADVANCED_DIR / "models"

PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
SPLIT_NAMES = ("train", "validation", "test")
EVALUATION_SPLITS = ("validation", "test")
LOSS_TYPES = ("qlike", "squared_error", "absolute_error")


def neural_prediction_path(*parts: str) -> Path:
    return NEURAL_PREDICTIONS_DIR.joinpath(*parts)


ORIGINAL_MODEL_FILES = {
    "HistoricalMean": ORIGINAL_PREDICTIONS_DIR / "pred_baseline_mean.csv",
    "RollingVol-5": ORIGINAL_PREDICTIONS_DIR / "pred_rolling_vol_5.csv",
    "RollingVol-10": ORIGINAL_PREDICTIONS_DIR / "pred_rolling_vol_10.csv",
    "RollingVol-20": ORIGINAL_PREDICTIONS_DIR / "pred_rolling_vol_20.csv",
    "GARCH(1,1)": ORIGINAL_PREDICTIONS_DIR / "pred_garch_11.csv",
    "ARIMA-GARCH": ORIGINAL_PREDICTIONS_DIR / "pred_arima_garch.csv",
    "LSTM": neural_prediction_path("pred_lstm_base.csv"),
    "ARIMA-GARCH-LSTM": neural_prediction_path("pred_lstm_hybrid.csv"),
    "LSTM-QLIKE": neural_prediction_path("lstm_tuned", "pred_lstm_qlike.csv"),
    "Hybrid-QLIKE": neural_prediction_path("lstm_tuned", "pred_hybrid_qlike.csv"),
    "LSTM-LogTarget": neural_prediction_path("lstm_tuned", "pred_lstm_logtarget.csv"),
    "LSTM-LogTarget-Small": neural_prediction_path("lstm_tuned", "pred_lstm_logtarget_small.csv"),
    "Hybrid-LogTarget": neural_prediction_path("lstm_tuned", "pred_hybrid_logtarget.csv"),
    "Hybrid-LogTarget-Small": neural_prediction_path("lstm_tuned", "pred_hybrid_logtarget_small.csv"),
}


@dataclass(frozen=True)
class CommonWindow:
    """Aligned long and wide prediction frames for one split."""

    split: str
    long: pd.DataFrame
    wide: pd.DataFrame
    models: list[str]
    common_keys: pd.DataFrame


def rel_path(path: Path) -> str:
    """Return a repository-relative path string when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_advanced_dirs() -> None:
    for path in (
        ADVANCED_DIR,
        ADVANCED_PREDICTIONS_DIR,
        ADVANCED_TABLES_DIR,
        ADVANCED_FIGURES_DIR,
        ADVANCED_AUDIT_DIR,
        ADVANCED_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: Path, *, force: bool = False) -> None:
    """Write a CSV file without overwriting unless ``force`` is true."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{rel_path(path)} already exists; pass --force to overwrite.")
    df.to_csv(path, index=False)


def safe_write_text(text: str, path: Path, *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{rel_path(path)} already exists; pass --force to overwrite.")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str], *, force: bool = False) -> None:
    safe_write_text("\n".join(lines), path, force=force)


def safe_write_json(data: dict[str, Any] | list[Any], path: Path, *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{rel_path(path)} already exists; pass --force to overwrite.")
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def slugify(value: object, *, max_length: int = 140) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return (slug or "item")[:max_length]


def parse_bool(value: Any) -> bool | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return pd.NA


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {rel_path(path)}")
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_date_columns(df: pd.DataFrame, *, source: Path | str) -> pd.DataFrame:
    parsed = df.copy()
    for column in ("date", "target_date"):
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
            bad_count = int(parsed[column].isna().sum())
            if bad_count:
                raise ValueError(f"{source} has {bad_count} bad {column} values.")
    return parsed


def load_splits() -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    required = {"date", "target_date", "log_return_pct", "squared_return", "target_var_next"}
    for split in SPLIT_NAMES:
        path = DATA_DIR / f"{split}.csv"
        df = parse_date_columns(_read_csv(path), source=path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{rel_path(path)} is missing required columns: {missing}")
        for column in required - {"date", "target_date"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{rel_path(path)} column {column} contains non-finite values.")
        splits[split] = df.sort_values("date").reset_index(drop=True)
    return splits


def load_model_ready() -> pd.DataFrame:
    path = DATA_DIR / "vnindex_model_ready.csv"
    df = parse_date_columns(_read_csv(path), source=path)
    required = {"date", "target_date", "log_return_pct", "squared_return", "target_var_next"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rel_path(path)} is missing required columns: {missing}")
    for column in required - {"date", "target_date"}:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if df.duplicated(["date", "target_date"]).any():
        raise ValueError(f"{rel_path(path)} has duplicate date/target_date rows.")
    return df


def combine_splits_with_gaps(splits: dict[str, pd.DataFrame], model_ready: pd.DataFrame) -> pd.DataFrame:
    """Return the full chronological stream needed for recursive state updates."""

    train_last = splits["train"]["date"].max()
    full = model_ready.loc[model_ready["date"] <= splits["test"]["date"].max()].copy()
    if not set(splits["train"]["date"]).issubset(set(full["date"])):
        raise ValueError("Model-ready data does not contain every training origin date.")
    if full.loc[full["date"] > train_last].empty:
        raise ValueError("No post-training observations are available for advanced forecasts.")
    return full.sort_values("date").reset_index(drop=True)


def prediction_frame(
    split_df: pd.DataFrame,
    pred_var: Iterable[float] | np.ndarray | pd.Series,
    *,
    model: str,
    split: str,
    clip_for_schema: bool = True,
) -> pd.DataFrame:
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(pred) != len(split_df):
        raise ValueError(
            f"{model} {split} predictions have length {len(pred)} but split has {len(split_df)} rows."
        )
    if not np.isfinite(pred).all():
        raise ValueError(f"{model} {split} predictions contain non-finite values.")
    if (pred <= 0).any() and not clip_for_schema:
        raise ValueError(f"{model} {split} predictions contain non-positive values.")
    if clip_for_schema:
        pred = np.clip(pred, EPSILON, None)
    frame = pd.DataFrame(
        {
            "date": split_df["date"],
            "target_date": split_df["target_date"],
            "actual_var": split_df["target_var_next"].astype(float),
            "pred_var": pred,
            "model": model,
            "split": split,
        }
    )
    return frame[PREDICTION_COLUMNS]


def validate_prediction_frame(df: pd.DataFrame, *, source: Path | str = "<frame>") -> pd.DataFrame:
    if list(df.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"{source} has columns {list(df.columns)}, expected {PREDICTION_COLUMNS}.")
    parsed = parse_date_columns(df, source=source)
    for column in ("actual_var", "pred_var"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        values = parsed[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains non-finite {column} values.")
    if (parsed["actual_var"] < 0).any():
        raise ValueError(f"{source} contains negative actual_var values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{source} contains non-positive pred_var values.")
    parsed["model"] = parsed["model"].astype(str).str.strip()
    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    if parsed["model"].eq("").any():
        raise ValueError(f"{source} contains blank model names.")
    invalid_splits = sorted(set(parsed.loc[~parsed["split"].isin(EVALUATION_SPLITS), "split"]))
    if invalid_splits:
        raise ValueError(f"{source} contains invalid split values: {invalid_splits}")
    duplicate_mask = parsed.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicate_mask.any():
        examples = parsed.loc[
            duplicate_mask, ["model", "split", "date", "target_date"]
        ].head(10)
        raise ValueError(f"{source} has duplicate prediction keys:\n{examples.to_string(index=False)}")
    return parsed[PREDICTION_COLUMNS].sort_values(["split", "date", "target_date", "model"])


def save_prediction_csv(df: pd.DataFrame, path: Path, *, force: bool = False) -> None:
    parsed = validate_prediction_frame(df, source=path)
    safe_write_csv(parsed, path, force=force)


def original_prediction_paths() -> list[Path]:
    return sorted(path for path in ORIGINAL_MODEL_FILES.values() if path.exists())


def advanced_prediction_paths() -> list[Path]:
    if not ADVANCED_PREDICTIONS_DIR.exists():
        return []
    return sorted(path for path in ADVANCED_PREDICTIONS_DIR.rglob("*.csv") if path.is_file())


def load_prediction_file(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    parsed = validate_prediction_frame(df, source=path)
    parsed["_source_file"] = rel_path(path)
    return parsed


def load_original_predictions(models: list[str] | None = None) -> pd.DataFrame:
    wanted = set(models) if models is not None else None
    frames = []
    for model, path in ORIGINAL_MODEL_FILES.items():
        if wanted is not None and model not in wanted:
            continue
        if path.exists():
            frames.append(load_prediction_file(path))
    if not frames:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "_source_file"])
    return pd.concat(frames, ignore_index=True)


def load_advanced_predictions() -> pd.DataFrame:
    frames = [load_prediction_file(path) for path in advanced_prediction_paths()]
    if not frames:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "_source_file"])
    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicate_mask.any():
        examples = combined.loc[
            duplicate_mask, ["_source_file", "model", "split", "date", "target_date"]
        ].head(20)
        raise ValueError(
            "Advanced predictions contain duplicate model/split/date/target_date rows:\n"
            f"{examples.to_string(index=False)}"
        )
    return combined


def load_all_predictions(include_advanced: bool = True) -> pd.DataFrame:
    frames = [load_original_predictions()]
    if include_advanced:
        advanced = load_advanced_predictions()
        if not advanced.empty:
            frames.append(advanced)
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "_source_file"])
    duplicate_mask = combined.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicate_mask.any():
        examples = combined.loc[
            duplicate_mask, ["_source_file", "model", "split", "date", "target_date"]
        ].head(20)
        raise ValueError(
            "Combined original and advanced predictions contain duplicate keys:\n"
            f"{examples.to_string(index=False)}"
        )
    return combined.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def loss_values(
    actual_var: pd.Series | np.ndarray,
    pred_var: pd.Series | np.ndarray,
    loss_type: str,
) -> np.ndarray:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(actual) != len(pred):
        raise ValueError("actual_var and pred_var must have equal length.")
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    if loss_type == "qlike":
        return np.log(pred) + actual / pred
    if loss_type == "squared_error":
        return (actual - pred) ** 2
    if loss_type == "absolute_error":
        return np.abs(actual - pred)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_metrics(df: pd.DataFrame, *, pred_column: str = "pred_var") -> dict[str, float | int]:
    if df.empty:
        return {
            "n_obs": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "QLIKE": np.nan,
            "mean_actual_var": np.nan,
            "mean_pred_var": np.nan,
        }
    actual = pd.to_numeric(df["actual_var"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df[pred_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = pred[finite]
    if len(actual) == 0:
        return {
            "n_obs": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "QLIKE": np.nan,
            "mean_actual_var": np.nan,
            "mean_pred_var": np.nan,
        }
    pred_clipped = np.clip(pred, EPSILON, None)
    return {
        "n_obs": int(len(actual)),
        "RMSE": float(np.sqrt(np.mean((actual - pred_clipped) ** 2))),
        "MAE": float(np.mean(np.abs(actual - pred_clipped))),
        "QLIKE": float(np.mean(np.log(pred_clipped) + actual / pred_clipped)),
        "mean_actual_var": float(np.mean(actual)),
        "mean_pred_var": float(np.mean(pred_clipped)),
    }


def metric_row(df: pd.DataFrame, *, model: str, split: str) -> dict[str, Any]:
    metrics = compute_metrics(df)
    return {
        "model": model,
        "split": split,
        **metrics,
        "date_min": df["date"].min().strftime("%Y-%m-%d") if not df.empty else "not_available",
        "date_max": df["date"].max().strftime("%Y-%m-%d") if not df.empty else "not_available",
        "target_date_min": df["target_date"].min().strftime("%Y-%m-%d")
        if not df.empty
        else "not_available",
        "target_date_max": df["target_date"].max().strftime("%Y-%m-%d")
        if not df.empty
        else "not_available",
    }


def per_observation_losses(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "target_date", "actual_var", "pred_var", "model", "split"]].copy()
    out["qlike_loss_t"] = loss_values(out["actual_var"], out["pred_var"], "qlike")
    out["squared_error_loss_t"] = loss_values(out["actual_var"], out["pred_var"], "squared_error")
    out["absolute_error_loss_t"] = loss_values(out["actual_var"], out["pred_var"], "absolute_error")
    return out


def common_window(predictions: pd.DataFrame, *, split: str, models: list[str] | None = None) -> CommonWindow:
    split_predictions = predictions[predictions["split"].astype(str).str.lower() == split].copy()
    if models is not None:
        split_predictions = split_predictions[split_predictions["model"].isin(models)].copy()
    if split_predictions.empty:
        raise ValueError(f"No {split} predictions are available.")
    model_list = sorted(split_predictions["model"].astype(str).unique())
    key_sets = {
        model: set(
            split_predictions.loc[
                split_predictions["model"] == model, ["date", "target_date"]
            ].itertuples(index=False, name=None)
        )
        for model in model_list
    }
    common_key_set = set.intersection(*key_sets.values()) if key_sets else set()
    if not common_key_set:
        raise ValueError(f"No common {split} window exists for models: {model_list}")
    common_keys = pd.DataFrame(sorted(common_key_set), columns=["date", "target_date"])
    common_long = split_predictions.merge(common_keys, on=["date", "target_date"], how="inner")
    actual = (
        common_long.groupby(["date", "target_date"], as_index=False)["actual_var"]
        .first()
        .sort_values(["date", "target_date"])
    )
    actual_check = common_long.groupby(["date", "target_date"])["actual_var"].agg(["min", "max"])
    if not np.isclose(actual_check["min"], actual_check["max"], rtol=1e-10, atol=1e-12).all():
        raise ValueError(f"{split} common window has inconsistent actual_var values.")
    wide = common_long.pivot(index=["date", "target_date"], columns="model", values="pred_var")
    wide = wide.reset_index()
    wide.columns.name = None
    wide = actual.merge(wide, on=["date", "target_date"], how="left")
    common_long = common_long.sort_values(["date", "target_date", "model"]).reset_index(drop=True)
    return CommonWindow(
        split=split,
        long=common_long,
        wide=wide,
        models=model_list,
        common_keys=common_keys.sort_values(["date", "target_date"]).reset_index(drop=True),
    )


def align_two_models(
    predictions: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    split: str,
) -> pd.DataFrame:
    a = predictions[
        (predictions["model"] == model_a) & (predictions["split"].astype(str).str.lower() == split)
    ][["date", "target_date", "actual_var", "pred_var"]].rename(columns={"pred_var": "pred_a"})
    b = predictions[
        (predictions["model"] == model_b) & (predictions["split"].astype(str).str.lower() == split)
    ][["date", "target_date", "pred_var"]].rename(columns={"pred_var": "pred_b"})
    merged = a.merge(b, on=["date", "target_date"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"No common {split} rows between {model_a} and {model_b}.")
    return merged.sort_values(["date", "target_date"]).reset_index(drop=True)


def align_model_list(predictions: pd.DataFrame, *, models: list[str], split: str) -> pd.DataFrame:
    common = common_window(predictions, split=split, models=models)
    return common.wide


def underprediction_stats(df: pd.DataFrame, *, p90: float | None = None) -> dict[str, Any]:
    if df.empty:
        return {
            "pred_actual_ratio": np.nan,
            "spike_underprediction_rate": np.nan,
            "extreme_threshold_p90": np.nan,
        }
    metrics = compute_metrics(df)
    ratio = (
        float(metrics["mean_pred_var"]) / float(metrics["mean_actual_var"])
        if float(metrics["mean_actual_var"]) > 0
        else np.nan
    )
    threshold = float(p90) if p90 is not None else float(df["actual_var"].quantile(0.90))
    extreme = df[df["actual_var"] > threshold]
    spike_rate = float((extreme["pred_var"] < extreme["actual_var"]).mean()) if not extreme.empty else np.nan
    return {
        "pred_actual_ratio": ratio,
        "spike_underprediction_rate": spike_rate,
        "extreme_threshold_p90": threshold,
    }


def save_failure_rows(rows: list[dict[str, Any]], path: Path, *, force: bool = False) -> None:
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "candidate_id",
                "stage",
                "failure_reason",
                "error_type",
                "error_message",
                "metadata",
            ]
        )
    safe_write_csv(df, path, force=force)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(p) if np.isfinite(float(p)) else np.nan for p in p_values]
    finite_items = [(idx, value) for idx, value in enumerate(values) if np.isfinite(value)]
    adjusted = [np.nan] * len(values)
    m = len(finite_items)
    running_max = 0.0
    for rank, (idx, value) in enumerate(sorted(finite_items, key=lambda item: item[1]), start=1):
        adj = min(1.0, (m - rank + 1) * value)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def summarize_file_hashes(paths: Iterable[Path]) -> pd.DataFrame:
    """Return size and modification metadata for audit-friendly output checks."""

    rows = []
    for path in sorted(paths):
        if path.exists() and path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "path": rel_path(path),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return pd.DataFrame(rows)
