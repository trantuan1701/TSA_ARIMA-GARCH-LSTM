#!/usr/bin/env python3
"""Train Stage 3 LSTM and ARIMA-GARCH-LSTM volatility models."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", str(os.environ.get("VNINDEX_RANDOM_SEED", "42")))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "The 'joblib' package is required for Stage 3. Install it with: "
        "python -m pip install joblib"
    ) from exc

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "The 'scikit-learn' package is required for Stage 3. Install it with: "
        "python -m pip install scikit-learn"
    ) from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "TensorFlow is required for Stage 3 LSTM models. Install it with: "
        "python -m pip install tensorflow"
    ) from exc

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
        if len(actual) != len(pred):
            raise ValueError(
                f"actual_var and pred_var must have equal length; got {len(actual)} and {len(pred)}."
            )
        finite_mask = np.isfinite(actual) & np.isfinite(pred)
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def optional_project_path(env_name: str) -> Path | None:
    raw = os.environ.get(env_name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


NEURAL_OUTPUT_ROOT = optional_project_path("VNINDEX_NEURAL_OUTPUT_ROOT")
if NEURAL_OUTPUT_ROOT is None:
    PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
    METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
    FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
    MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
else:
    PREDICTIONS_DIR = NEURAL_OUTPUT_ROOT / "predictions"
    METRICS_DIR = NEURAL_OUTPUT_ROOT / "metrics"
    FIGURES_DIR = NEURAL_OUTPUT_ROOT / "figures"
    MODELS_DIR = NEURAL_OUTPUT_ROOT / "models"

SEED = int(os.environ.get("VNINDEX_RANDOM_SEED", "42"))
SEQ_LEN = 20
EPOCHS = 100
BATCH_SIZE = 32

BASE_MODEL_NAME = "LSTM"
HYBRID_MODEL_NAME = "ARIMA-GARCH-LSTM"

BASE_FEATURES_REQUIRED = [
    "log_return_pct",
    "squared_return",
    "abs_return",
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
]
OPTIONAL_MARKET_FEATURES = ["volume", "trading_value"]
REQUIRED_ECON_FEATURES = [
    "garch_11_pred_var",
    "garch_11_pred_vol",
    "arima_garch_pred_var",
    "arima_garch_pred_vol",
]
OPTIONAL_ECON_FEATURES = ["arima_mean_pred", "arima_resid_proxy"]
FORBIDDEN_FEATURES = {"target_var_next", "target_date", "date", "split"}

BASE_REQUIRED_COLUMNS = [
    "date",
    "target_date",
    "log_return_pct",
    "squared_return",
    "abs_return",
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
    "target_var_next",
]
PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
METRIC_COLUMNS = [
    "model",
    "split",
    "n_obs",
    "mse",
    "rmse",
    "mae",
    "qlike",
    "mean_actual_var",
    "mean_pred_var",
]


@dataclass
class SequenceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_meta: pd.DataFrame
    x_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame
    counts: dict[str, int]


@dataclass
class ModelRunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    history: pd.DataFrame
    sequence_counts: dict[str, int]
    final_val_loss: float
    paths: dict[str, Path]
    feature_names: list[str]
    imputation_messages: list[str]


def ensure_dirs() -> None:
    for path in (PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], source_path: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")


def parse_dates_and_sort(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    parsed = df.copy()
    for column in ("date", "target_date"):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        if parsed[column].isna().any():
            bad_rows = parsed.index[parsed[column].isna()].tolist()[:10]
            raise ValueError(f"Could not parse {column} in {source_path}; bad row indexes: {bad_rows}")
    parsed = parsed.sort_values("date").reset_index(drop=True)
    if not parsed["date"].is_monotonic_increasing:
        raise ValueError(f"{source_path} is not sorted chronologically after date parsing.")
    return parsed


def validate_numeric_target(df: pd.DataFrame, split_name: str, source_path: Path) -> pd.DataFrame:
    validated = df.copy()
    validated["target_var_next"] = pd.to_numeric(validated["target_var_next"], errors="coerce")
    target_values = validated["target_var_next"].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        bad_count = int((~np.isfinite(target_values)).sum())
        raise ValueError(
            f"{source_path} split {split_name} has {bad_count} missing or non-finite target_var_next values."
        )
    return validated


def load_split_file(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 1 split file does not exist: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"target_var_next": "string"})
    validate_required_columns(df, BASE_REQUIRED_COLUMNS, path)
    df["_target_var_next_raw"] = df["target_var_next"].astype(str)
    df = parse_dates_and_sort(df, path)
    return validate_numeric_target(df, split_name, path)


def load_stage_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_paths = {
        "train": PROCESSED_DIR / "train.csv",
        "validation": PROCESSED_DIR / "validation.csv",
        "test": PROCESSED_DIR / "test.csv",
    }
    base_splits = {split: load_split_file(path, split) for split, path in base_paths.items()}

    bridge_path = PROCESSED_DIR / "vnindex_with_econometric_features.csv"
    if not bridge_path.exists():
        raise FileNotFoundError(f"Required Stage 2 bridge file does not exist: {bridge_path}")
    bridge = pd.read_csv(bridge_path, encoding="utf-8-sig")
    validate_required_columns(bridge, BASE_REQUIRED_COLUMNS + ["split"], bridge_path)
    bridge = parse_dates_and_sort(bridge, bridge_path)
    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(bridge["split"].dropna().astype(str).unique())
    missing_splits = sorted(allowed_splits - observed_splits)
    if missing_splits:
        raise ValueError(f"{bridge_path} is missing split values: {missing_splits}")
    unexpected_splits = sorted(observed_splits - allowed_splits)
    if unexpected_splits:
        raise ValueError(f"{bridge_path} has unexpected split values: {unexpected_splits}")

    bridge_splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        split_df = bridge.loc[bridge["split"] == split].copy().reset_index(drop=True)
        split_df = validate_numeric_target(split_df, split, bridge_path)
        canonical = base_splits[split].reset_index(drop=True)
        if len(split_df) != len(canonical):
            raise ValueError(
                f"{bridge_path} split {split} has {len(split_df)} rows but the canonical "
                f"Stage 1 split has {len(canonical)} rows."
            )
        if not (
            split_df["date"].reset_index(drop=True).equals(canonical["date"])
            and split_df["target_date"].reset_index(drop=True).equals(canonical["target_date"])
        ):
            raise ValueError(
                f"{bridge_path} split {split} date/target_date rows do not align with the "
                "canonical Stage 1 split."
            )
        split_df["target_var_next"] = canonical["target_var_next"].to_numpy(dtype=float)
        split_df["_target_var_next_raw"] = canonical["_target_var_next_raw"].to_numpy()
        bridge_splits[split] = split_df

    return base_splits, bridge_splits


def get_base_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    validate_required_columns(df, BASE_FEATURES_REQUIRED, Path("base split"))
    features = list(BASE_FEATURES_REQUIRED)
    messages: list[str] = []
    for column in OPTIONAL_MARKET_FEATURES:
        if column in df.columns:
            features.append(column)
        else:
            messages.append(f"Optional market feature '{column}' is missing and was excluded.")
    return features, messages


def get_hybrid_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features, messages = get_base_features(df)
    missing_required = [column for column in REQUIRED_ECON_FEATURES if column not in df.columns]
    if missing_required:
        raise ValueError(
            "Hybrid model requires Stage 2 econometric bridge columns, but these are missing: "
            f"{missing_required}"
        )
    features.extend(REQUIRED_ECON_FEATURES)
    for column in OPTIONAL_ECON_FEATURES:
        if column in df.columns:
            features.append(column)
        else:
            messages.append(f"Optional econometric feature '{column}' is missing and was excluded.")
    return features, messages


def assert_no_forbidden_features(features: list[str], model_name: str) -> None:
    forbidden_used = sorted(FORBIDDEN_FEATURES.intersection(features))
    if forbidden_used:
        raise ValueError(f"{model_name} feature list contains forbidden predictors: {forbidden_used}")


def validate_split_lengths(splits: dict[str, pd.DataFrame], seq_len: int, model_name: str) -> None:
    for split, df in splits.items():
        if len(df) < seq_len:
            raise ValueError(
                f"{model_name} {split} split has {len(df)} rows, fewer than seq_len={seq_len}."
            )


def coerce_feature_columns(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    model_name: str,
) -> dict[str, pd.DataFrame]:
    coerced: dict[str, pd.DataFrame] = {}
    for split, df in splits.items():
        missing = [column for column in feature_names if column not in df.columns]
        if missing:
            raise ValueError(f"{model_name} {split} split is missing feature columns: {missing}")
        current = df.copy()
        for column in feature_names:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        coerced[split] = current
    return coerced


def missing_feature_counts(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in splits.items():
        for column in feature_names:
            values = df[column].to_numpy(dtype=float)
            missing_count = int((~np.isfinite(values)).sum())
            if missing_count:
                rows.append({"split": split, "feature": column, "missing_or_nonfinite": missing_count})
    return pd.DataFrame(rows, columns=["split", "feature", "missing_or_nonfinite"])


def fit_imputer_values(train_df: pd.DataFrame, feature_names: list[str], model_name: str) -> dict[str, float]:
    imputer_values: dict[str, float] = {}
    for column in feature_names:
        values = train_df[column].replace([np.inf, -np.inf], np.nan)
        median_value = float(values.median(skipna=True))
        if not np.isfinite(median_value):
            raise ValueError(f"Cannot fit {model_name} imputer: training median for {column} is not finite.")
        imputer_values[column] = median_value
    return imputer_values


def apply_imputation(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    imputer_values: dict[str, float],
) -> dict[str, pd.DataFrame]:
    imputed: dict[str, pd.DataFrame] = {}
    for split, df in splits.items():
        current = df.copy()
        for column in feature_names:
            current[column] = current[column].replace([np.inf, -np.inf], np.nan)
            current[column] = current[column].fillna(imputer_values[column])
        imputed[split] = current
    return imputed


def fit_scaler(train_df: pd.DataFrame, feature_names: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names].to_numpy(dtype=float))
    return scaler


def transform_features(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    scaler: StandardScaler,
) -> dict[str, np.ndarray]:
    return {
        split: scaler.transform(df[feature_names].to_numpy(dtype=float)).astype(np.float32)
        for split, df in splits.items()
    }


def create_sequences(
    split_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    seq_len: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(split_df) != len(feature_matrix):
        raise ValueError(
            f"{split_name} split has {len(split_df)} rows but feature matrix has {len(feature_matrix)} rows."
        )
    if len(split_df) < seq_len:
        raise ValueError(f"{split_name} split has {len(split_df)} rows, fewer than seq_len={seq_len}.")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    targets = split_df["target_var_next"].to_numpy(dtype=float)

    for end_idx in range(seq_len - 1, len(split_df)):
        start_idx = end_idx - seq_len + 1
        x_values.append(feature_matrix[start_idx : end_idx + 1])
        y_values.append(float(targets[end_idx]))
        row = split_df.iloc[end_idx]
        meta_rows.append(
            {
                "date": row["date"],
                "target_date": row["target_date"],
                "target_var_next": float(row["target_var_next"]),
                "target_var_next_raw": row["_target_var_next_raw"],
            }
        )

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        pd.DataFrame(meta_rows),
    )


def create_sequences_with_context(
    context_df: pd.DataFrame,
    context_features: np.ndarray,
    target_df: pd.DataFrame,
    seq_len: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create target-split sequences using prior chronological context rows.

    For validation and test forecasts, using rows before the split start as
    lagged context is not leakage: the sequence ends at the forecast origin and
    never includes target columns as model inputs.
    """

    if len(context_df) != len(context_features):
        raise ValueError(
            f"{split_name} context has {len(context_df)} rows but feature matrix has "
            f"{len(context_features)} rows."
        )
    if context_df.duplicated(["date", "target_date"]).any():
        raise ValueError(f"{split_name} context has duplicate date/target_date rows.")

    indexed = context_df.reset_index().rename(columns={"index": "_context_index"})
    target_keys = target_df[["date", "target_date"]].copy()
    target_positions = target_keys.merge(
        indexed[["date", "target_date", "_context_index"]],
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if target_positions["_context_index"].isna().any():
        missing = target_positions.loc[target_positions["_context_index"].isna(), ["date", "target_date"]].head(5)
        raise ValueError(f"{split_name} target rows missing from context:\n{missing.to_string(index=False)}")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    context_targets = context_df["target_var_next"].to_numpy(dtype=float)

    for _, position_row in target_positions.iterrows():
        end_idx = int(position_row["_context_index"])
        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            continue
        origin_row = context_df.iloc[end_idx]
        sequence_dates = context_df.iloc[start_idx : end_idx + 1]["date"]
        if sequence_dates.max() > origin_row["date"]:
            raise ValueError(f"{split_name} context sequence includes a future date.")
        x_values.append(context_features[start_idx : end_idx + 1])
        y_values.append(float(context_targets[end_idx]))
        meta_rows.append(
            {
                "date": origin_row["date"],
                "target_date": origin_row["target_date"],
                "target_var_next": float(origin_row["target_var_next"]),
                "target_var_next_raw": origin_row["_target_var_next_raw"],
            }
        )

    if not x_values:
        raise ValueError(f"{split_name} produced no context-aware sequences.")

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        pd.DataFrame(meta_rows),
    )


def build_sequence_data(
    splits: dict[str, pd.DataFrame],
    transformed: dict[str, np.ndarray],
    seq_len: int,
) -> SequenceData:
    x_train, y_train, _train_meta = create_sequences(splits["train"], transformed["train"], seq_len, "train")
    context_df = pd.concat(
        [splits["train"], splits["validation"], splits["test"]],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)
    context_features = np.concatenate(
        [transformed["train"], transformed["validation"], transformed["test"]],
        axis=0,
    )
    x_validation, y_validation, validation_meta = create_sequences_with_context(
        context_df,
        context_features,
        splits["validation"],
        seq_len,
        "validation",
    )
    x_test, y_test, test_meta = create_sequences_with_context(
        context_df,
        context_features,
        splits["test"],
        seq_len,
        "test",
    )
    return SequenceData(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        validation_meta=validation_meta,
        x_test=x_test,
        y_test=y_test,
        test_meta=test_meta,
        counts={
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
    )


def build_lstm_model(seq_len: int, num_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, num_features)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="softplus"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    sequence_data: SequenceData,
) -> tf.keras.callbacks.History:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    return model.fit(
        sequence_data.x_train,
        sequence_data.y_train,
        validation_data=(sequence_data.x_validation, sequence_data.y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )


def predict_model(model: tf.keras.Model, x_values: np.ndarray) -> np.ndarray:
    pred = model.predict(x_values, verbose=0).reshape(-1)
    if not np.isfinite(pred).all():
        bad_count = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Model produced {bad_count} non-finite predictions.")
    return np.clip(pred.astype(float), EPSILON, None)


def make_prediction_frame(
    meta: pd.DataFrame,
    pred_var: np.ndarray,
    *,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    if len(meta) != len(pred_var):
        raise ValueError(
            f"{model_name} {split} metadata has {len(meta)} rows but predictions have {len(pred_var)} rows."
        )
    actual_var = meta["target_var_next_raw"].astype(str)
    actual_numeric = pd.to_numeric(actual_var, errors="raise").to_numpy(dtype=float)
    if not np.array_equal(actual_numeric, meta["target_var_next"].to_numpy(dtype=float)):
        raise ValueError(f"{model_name} {split} actual_var source text does not match target_var_next.")

    frame = pd.DataFrame(
        {
            "date": meta["date"],
            "target_date": meta["target_date"],
            "actual_var": actual_var,
            "pred_var": np.clip(np.asarray(pred_var, dtype=float), EPSILON, None),
            "model": model_name,
            "split": split,
        }
    )
    if not np.array_equal(
        pd.to_numeric(frame["actual_var"], errors="raise").to_numpy(dtype=float),
        meta["target_var_next"].to_numpy(dtype=float),
    ):
        raise ValueError(f"{model_name} {split} actual_var does not equal target_var_next.")
    if not np.isfinite(frame["pred_var"].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_name} {split} predictions contain non-finite values after clipping.")
    if (frame["pred_var"] <= 0).any():
        raise ValueError(f"{model_name} {split} predictions contain non-positive values after clipping.")
    return frame[PREDICTION_COLUMNS]


def evaluate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, split), group in predictions.groupby(["model", "split"], sort=False):
        metrics = evaluate_volatility_predictions(
            group["actual_var"],
            group["pred_var"],
            epsilon=EPSILON,
            label=f"{model_name} {split}",
        )
        row = {"model": model_name, "split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)[METRIC_COLUMNS]


def save_history(history: tf.keras.callbacks.History, path: Path) -> pd.DataFrame:
    history_df = pd.DataFrame(history.history)
    rename_map = {
        "mean_absolute_error": "mae",
        "val_mean_absolute_error": "val_mae",
    }
    history_df = history_df.rename(columns=rename_map)
    expected_columns = ["loss", "mae", "val_loss", "val_mae"]
    missing = [column for column in expected_columns if column not in history_df.columns]
    if missing:
        raise ValueError(f"Training history is missing expected columns: {missing}")
    history_df = history_df[expected_columns].copy()
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(path, index=False)
    return history_df


def plot_training_history(history_df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["loss"], label="Training loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_feature_list(feature_names: list[str], path: Path) -> None:
    path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")


def save_imputer_values(imputer_values: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(imputer_values, indent=2, sort_keys=True), encoding="utf-8")


def final_validation_loss(history_df: pd.DataFrame) -> float:
    return float(history_df["val_loss"].iloc[-1])


def run_lstm_pipeline(
    *,
    model_name: str,
    output_stem: str,
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> ModelRunResult:
    assert_no_forbidden_features(feature_names, model_name)
    validate_split_lengths(splits, SEQ_LEN, model_name)

    coerced = coerce_feature_columns(splits, feature_names, model_name)
    missing_counts = missing_feature_counts(coerced, feature_names)
    imputation_messages: list[str] = []
    if missing_counts.empty:
        imputation_messages.append(f"{model_name}: no missing or non-finite feature values found.")
    else:
        for row in missing_counts.to_dict("records"):
            imputation_messages.append(
                f"{model_name}: {row['split']} feature {row['feature']} has "
                f"{row['missing_or_nonfinite']} missing/non-finite values."
            )

    imputer_values = fit_imputer_values(coerced["train"], feature_names, model_name)
    imputed = apply_imputation(coerced, feature_names, imputer_values)
    scaler = fit_scaler(imputed["train"], feature_names)
    transformed = transform_features(imputed, feature_names, scaler)
    sequence_data = build_sequence_data(imputed, transformed, SEQ_LEN)

    model = build_lstm_model(SEQ_LEN, len(feature_names))
    history = train_model(model, sequence_data)

    paths = {
        "predictions": PREDICTIONS_DIR / f"pred_{output_stem}.csv",
        "history": METRICS_DIR / f"{output_stem}_training_history.csv",
        "figure": FIGURES_DIR / f"fig_{output_stem}_loss.png",
        "model": MODELS_DIR / f"{output_stem}.keras",
        "scaler": MODELS_DIR / f"{output_stem}_scaler.pkl",
        "features": MODELS_DIR / f"{output_stem}_features.txt",
        "imputer": MODELS_DIR / f"{output_stem}_imputer_values.json",
    }

    history_df = save_history(history, paths["history"])
    plot_training_history(history_df, paths["figure"], f"{model_name} training loss")

    validation_pred = predict_model(model, sequence_data.x_validation)
    test_pred = predict_model(model, sequence_data.x_test)
    predictions = pd.concat(
        [
            make_prediction_frame(
                sequence_data.validation_meta,
                validation_pred,
                model_name=model_name,
                split="validation",
            ),
            make_prediction_frame(
                sequence_data.test_meta,
                test_pred,
                model_name=model_name,
                split="test",
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(paths["predictions"], index=False, float_format="%.17g")

    metrics = evaluate_prediction_frame(predictions)
    model.save(paths["model"])
    joblib.dump(scaler, paths["scaler"])
    save_feature_list(feature_names, paths["features"])
    save_imputer_values(imputer_values, paths["imputer"])

    return ModelRunResult(
        predictions=predictions,
        metrics=metrics,
        history=history_df,
        sequence_counts=sequence_data.counts,
        final_val_loss=final_validation_loss(history_df),
        paths=paths,
        feature_names=feature_names,
        imputation_messages=imputation_messages,
    )


def print_metrics(metrics: pd.DataFrame, title: str) -> None:
    display = metrics.copy()
    for column in ["mse", "rmse", "mae", "qlike", "mean_actual_var", "mean_pred_var"]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    print(title)
    print(display.to_string(index=False))


def leakage_report(base_features: list[str], hybrid_features: list[str]) -> list[str]:
    for model_name, features in ((BASE_MODEL_NAME, base_features), (HYBRID_MODEL_NAME, hybrid_features)):
        assert_no_forbidden_features(features, model_name)
    return [
        "target_var_next is not in either feature list.",
        "target_date is not in either feature list.",
        "date is not in either feature list.",
        "split is not in either feature list.",
        "Scalers are fit only on training rows for each model pipeline.",
        "Validation and test sequences may use earlier chronological context, but every sequence ends at the forecast origin.",
        "Test data is not passed to model.fit or early stopping.",
        "Prediction actual_var is assigned directly from target_var_next and checked before saving.",
    ]


def main() -> None:
    ensure_dirs()
    set_seeds(SEED)

    base_splits, bridge_splits = load_stage_data()
    base_features, base_feature_messages = get_base_features(base_splits["train"])
    hybrid_features, hybrid_feature_messages = get_hybrid_features(bridge_splits["train"])

    print("Loaded row counts:")
    for split in ("train", "validation", "test"):
        print(
            f"  {split}: base={len(base_splits[split])}, "
            f"bridge={len(bridge_splits[split])}"
        )
    print(f"Sequence length: {SEQ_LEN}")
    print(f"{BASE_MODEL_NAME} features ({len(base_features)}): {', '.join(base_features)}")
    print(f"{HYBRID_MODEL_NAME} features ({len(hybrid_features)}): {', '.join(hybrid_features)}")
    print("Architecture: Input -> LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(1, softplus)")

    for message in base_feature_messages + hybrid_feature_messages:
        print(f"Feature decision: {message}")

    print("Training LSTM...")
    base_result = run_lstm_pipeline(
        model_name=BASE_MODEL_NAME,
        output_stem="lstm_base",
        splits=base_splits,
        feature_names=base_features,
    )
    print(
        f"{BASE_MODEL_NAME} sequence counts: "
        f"train={base_result.sequence_counts['train']}, "
        f"validation={base_result.sequence_counts['validation']}, "
        f"test={base_result.sequence_counts['test']}"
    )
    print(f"{BASE_MODEL_NAME} final validation loss: {base_result.final_val_loss:.6g}")
    print_metrics(base_result.metrics, f"{BASE_MODEL_NAME} metrics:")

    print("Training ARIMA-GARCH-LSTM...")
    hybrid_result = run_lstm_pipeline(
        model_name=HYBRID_MODEL_NAME,
        output_stem="lstm_hybrid",
        splits=bridge_splits,
        feature_names=hybrid_features,
    )
    print(
        f"{HYBRID_MODEL_NAME} sequence counts: "
        f"train={hybrid_result.sequence_counts['train']}, "
        f"validation={hybrid_result.sequence_counts['validation']}, "
        f"test={hybrid_result.sequence_counts['test']}"
    )
    print(f"{HYBRID_MODEL_NAME} final validation loss: {hybrid_result.final_val_loss:.6g}")
    print_metrics(hybrid_result.metrics, f"{HYBRID_MODEL_NAME} metrics:")

    all_metrics = pd.concat([base_result.metrics, hybrid_result.metrics], ignore_index=True)
    all_metrics.to_csv(METRICS_DIR / "lstm_metrics.csv", index=False)
    test_only = (
        all_metrics.loc[all_metrics["split"] == "test"]
        .sort_values(["qlike", "rmse"], ascending=[True, True])
        .reset_index(drop=True)
    )
    test_only.to_csv(METRICS_DIR / "lstm_metrics_test_only.csv", index=False)

    feature_summary = pd.DataFrame(
        [
            {
                "model": BASE_MODEL_NAME,
                "num_features": len(base_result.feature_names),
                "feature_names": ",".join(base_result.feature_names),
                "seq_len": SEQ_LEN,
                "train_sequences": base_result.sequence_counts["train"],
                "validation_sequences": base_result.sequence_counts["validation"],
                "test_sequences": base_result.sequence_counts["test"],
            },
            {
                "model": HYBRID_MODEL_NAME,
                "num_features": len(hybrid_result.feature_names),
                "feature_names": ",".join(hybrid_result.feature_names),
                "seq_len": SEQ_LEN,
                "train_sequences": hybrid_result.sequence_counts["train"],
                "validation_sequences": hybrid_result.sequence_counts["validation"],
                "test_sequences": hybrid_result.sequence_counts["test"],
            },
        ]
    )
    feature_summary.to_csv(METRICS_DIR / "lstm_feature_summary.csv", index=False)

    print("Imputation decisions:")
    for message in base_result.imputation_messages + hybrid_result.imputation_messages:
        print(f"  {message}")
    print("Leakage checks:")
    for message in leakage_report(base_features, hybrid_features):
        print(f"  PASS: {message}")

    saved_paths = [
        base_result.paths["predictions"],
        hybrid_result.paths["predictions"],
        METRICS_DIR / "lstm_metrics.csv",
        METRICS_DIR / "lstm_metrics_test_only.csv",
        base_result.paths["history"],
        hybrid_result.paths["history"],
        METRICS_DIR / "lstm_feature_summary.csv",
        base_result.paths["figure"],
        hybrid_result.paths["figure"],
        base_result.paths["model"],
        hybrid_result.paths["model"],
        base_result.paths["scaler"],
        hybrid_result.paths["scaler"],
        base_result.paths["features"],
        hybrid_result.paths["features"],
    ]
    print("Saved output paths:")
    for path in saved_paths:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print_metrics(test_only, "Test metrics sorted by QLIKE then RMSE:")


if __name__ == "__main__":
    main()
