#!/usr/bin/env python3
"""Stage 3B neural tuning for VN-Index volatility forecasts.

This script trains validation-selected LSTM variants without touching the
audited Stage 1-3 artifacts. All outputs are written under ``lstm_tuned``
subdirectories.
"""

from __future__ import annotations

import os

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:  # pragma: no cover - dependency guard.
    raise ImportError("The 'joblib' package is required. Install it with: python -m pip install joblib") from exc

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - dependency guard.
    raise ImportError(
        "The 'scikit-learn' package is required. Install it with: python -m pip install scikit-learn"
    ) from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - dependency guard.
    raise ImportError("TensorFlow is required. Install it with: python -m pip install tensorflow") from exc

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
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions" / "lstm_tuned"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics" / "lstm_tuned"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "lstm_tuned"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "lstm_tuned"

SEQ_LEN = 20
EPOCHS = 200
BATCH_SIZE = 32

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


@dataclass(frozen=True)
class VariantConfig:
    model_name: str
    slug: str
    feature_set: str
    target_mode: str
    loss_name: str
    seq_len: int = SEQ_LEN
    units: int = 64
    dropout: float = 0.2
    dense_units: int = 32


@dataclass
class SequenceData:
    x_train: np.ndarray
    y_train_raw: np.ndarray
    x_validation: np.ndarray
    y_validation_raw: np.ndarray
    validation_meta: pd.DataFrame
    x_test: np.ndarray
    y_test_raw: np.ndarray
    test_meta: pd.DataFrame
    counts: dict[str, int]


@dataclass
class PreparedFeatures:
    transformed: dict[str, np.ndarray]
    scaler: StandardScaler
    imputer: SimpleImputer
    imputation_messages: list[str]


@dataclass
class ModelRunResult:
    config: VariantConfig
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    history: pd.DataFrame
    sequence_counts: dict[str, int]
    paths: dict[str, Path]
    feature_names: list[str]
    imputation_messages: list[str]


VARIANTS = [
    VariantConfig(
        model_name="LSTM-LogTarget",
        slug="lstm_logtarget",
        feature_set="base",
        target_mode="log",
        loss_name="mse_log",
    ),
    VariantConfig(
        model_name="Hybrid-LogTarget",
        slug="hybrid_logtarget",
        feature_set="hybrid",
        target_mode="log",
        loss_name="mse_log",
    ),
    VariantConfig(
        model_name="LSTM-QLIKE",
        slug="lstm_qlike",
        feature_set="base",
        target_mode="raw",
        loss_name="qlike",
    ),
    VariantConfig(
        model_name="Hybrid-QLIKE",
        slug="hybrid_qlike",
        feature_set="hybrid",
        target_mode="raw",
        loss_name="qlike",
    ),
    VariantConfig(
        model_name="LSTM-LogTarget-Small",
        slug="lstm_logtarget_small",
        feature_set="base",
        target_mode="log",
        loss_name="mse_log",
        units=32,
        dropout=0.1,
        dense_units=16,
    ),
    VariantConfig(
        model_name="Hybrid-LogTarget-Small",
        slug="hybrid_logtarget_small",
        feature_set="hybrid",
        target_mode="log",
        loss_name="mse_log",
        units=32,
        dropout=0.1,
        dense_units=16,
    ),
]


@tf.keras.utils.register_keras_serializable(package="VNIndex")
def qlike_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """QLIKE loss on variance scale, matching the project metric definition."""

    epsilon = tf.cast(EPSILON, y_pred.dtype)
    pred_clipped = tf.maximum(y_pred, epsilon)
    return tf.reduce_mean(tf.math.log(pred_clipped) + y_true / pred_clipped)


def ensure_dirs() -> None:
    for path in (PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def variant_paths(config: VariantConfig) -> dict[str, Path]:
    return {
        "predictions": PREDICTIONS_DIR / f"pred_{config.slug}.csv",
        "history": METRICS_DIR / f"history_{config.slug}.csv",
        "figure": FIGURES_DIR / f"fig_loss_{config.slug}.png",
        "model": MODELS_DIR / f"model_{config.slug}.keras",
        "scaler": MODELS_DIR / f"scaler_{config.slug}.pkl",
        "features": MODELS_DIR / f"features_{config.slug}.txt",
        "imputer": MODELS_DIR / f"imputer_{config.slug}.pkl",
    }


def cleanup_previous_outputs(configs: list[VariantConfig]) -> None:
    metric_files = [
        METRICS_DIR / "lstm_tuned_metrics.csv",
        METRICS_DIR / "lstm_tuned_metrics_test_only.csv",
        METRICS_DIR / "lstm_tuned_selection_summary.csv",
        METRICS_DIR / "lstm_tuned_failed_variants.csv",
    ]
    figure_files = [FIGURES_DIR / "fig_lstm_tuned_qlike_comparison.png"]
    paths = [*metric_files, *figure_files]
    for config in configs:
        paths.extend(variant_paths(config).values())

    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], source_path: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{rel_path(source_path)} is missing required columns: {missing}")


def parse_dates_and_sort(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    parsed = df.copy()
    for column in ("date", "target_date"):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        if parsed[column].isna().any():
            bad_rows = parsed.index[parsed[column].isna()].tolist()[:10]
            raise ValueError(f"Could not parse {column} in {rel_path(source_path)}; bad rows: {bad_rows}")
    parsed = parsed.sort_values("date").reset_index(drop=True)
    if not parsed["date"].is_monotonic_increasing:
        raise ValueError(f"{rel_path(source_path)} is not chronological after date parsing.")
    return parsed


def validate_numeric_target(df: pd.DataFrame, split_name: str, source_path: Path) -> pd.DataFrame:
    validated = df.copy()
    validated["target_var_next"] = pd.to_numeric(validated["target_var_next"], errors="coerce")
    target_values = validated["target_var_next"].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        bad_count = int((~np.isfinite(target_values)).sum())
        raise ValueError(
            f"{rel_path(source_path)} split {split_name} has {bad_count} non-finite target_var_next rows."
        )
    if (target_values < 0).any():
        bad_count = int((target_values < 0).sum())
        raise ValueError(
            f"{rel_path(source_path)} split {split_name} has {bad_count} negative target_var_next rows."
        )
    return validated


def load_split_file(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split file does not exist: {rel_path(path)}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"target_var_next": "string"})
    validate_required_columns(df, BASE_REQUIRED_COLUMNS, path)
    df["_target_var_next_raw"] = df["target_var_next"].astype(str)
    df = parse_dates_and_sort(df, path)
    return validate_numeric_target(df, split_name, path)


def validate_chronological_split_order(splits: dict[str, pd.DataFrame]) -> None:
    if not splits["train"]["date"].max() < splits["validation"]["date"].min():
        raise ValueError("Train and validation split dates overlap or are out of order.")
    if not splits["validation"]["date"].max() < splits["test"]["date"].min():
        raise ValueError("Validation and test split dates overlap or are out of order.")


def load_stage_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_paths = {
        "train": PROCESSED_DIR / "train.csv",
        "validation": PROCESSED_DIR / "validation.csv",
        "test": PROCESSED_DIR / "test.csv",
    }
    base_splits = {split: load_split_file(path, split) for split, path in base_paths.items()}
    validate_chronological_split_order(base_splits)

    bridge_path = PROCESSED_DIR / "vnindex_with_econometric_features.csv"
    if not bridge_path.exists():
        raise FileNotFoundError(f"Required econometric feature bridge does not exist: {rel_path(bridge_path)}")
    bridge = pd.read_csv(bridge_path, encoding="utf-8-sig")
    validate_required_columns(bridge, BASE_REQUIRED_COLUMNS + ["split"], bridge_path)
    bridge = parse_dates_and_sort(bridge, bridge_path)

    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(bridge["split"].dropna().astype(str).unique())
    missing_splits = sorted(allowed_splits - observed_splits)
    if missing_splits:
        raise ValueError(f"{rel_path(bridge_path)} is missing split values: {missing_splits}")
    unexpected_splits = sorted(observed_splits - allowed_splits)
    if unexpected_splits:
        raise ValueError(f"{rel_path(bridge_path)} has unexpected split values: {unexpected_splits}")

    bridge_splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        split_df = bridge.loc[bridge["split"] == split].copy().reset_index(drop=True)
        split_df = validate_numeric_target(split_df, split, bridge_path)
        canonical = base_splits[split].reset_index(drop=True)
        if len(split_df) != len(canonical):
            raise ValueError(
                f"{rel_path(bridge_path)} split {split} has {len(split_df)} rows but "
                f"{rel_path(PROCESSED_DIR / f'{split}.csv')} has {len(canonical)} rows."
            )
        if not (
            split_df["date"].reset_index(drop=True).equals(canonical["date"])
            and split_df["target_date"].reset_index(drop=True).equals(canonical["target_date"])
        ):
            raise ValueError(
                f"{rel_path(bridge_path)} split {split} date/target_date rows do not align with Stage 1 split."
            )
        split_df["target_var_next"] = canonical["target_var_next"].to_numpy(dtype=float)
        split_df["_target_var_next_raw"] = canonical["_target_var_next_raw"].to_numpy()
        bridge_splits[split] = split_df

    validate_chronological_split_order(bridge_splits)
    return base_splits, bridge_splits


def get_base_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    validate_required_columns(df, BASE_FEATURES_REQUIRED, Path("data/processed/train.csv"))
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
        raise ValueError(f"Hybrid feature bridge is missing required econometric columns: {missing_required}")
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
            raise ValueError(f"{model_name} {split} split has {len(df)} rows, fewer than seq_len={seq_len}.")


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


def clean_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return df[feature_names].replace([np.inf, -np.inf], np.nan)


def missing_feature_counts(splits: dict[str, pd.DataFrame], feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in splits.items():
        clean = clean_feature_frame(df, feature_names)
        for column in feature_names:
            missing_count = int(clean[column].isna().sum())
            if missing_count:
                rows.append({"split": split, "feature": column, "missing_or_nonfinite": missing_count})
    return pd.DataFrame(rows, columns=["split", "feature", "missing_or_nonfinite"])


def fit_train_imputer(train_df: pd.DataFrame, feature_names: list[str], model_name: str) -> SimpleImputer:
    train_features = clean_feature_frame(train_df, feature_names)
    medians = train_features.median(skipna=True)
    bad_columns = [column for column in feature_names if not np.isfinite(float(medians[column]))]
    if bad_columns:
        raise ValueError(f"Cannot fit {model_name} imputer; non-finite training medians for {bad_columns}.")

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_features)
    if len(imputer.statistics_) != len(feature_names) or not np.isfinite(imputer.statistics_).all():
        raise ValueError(f"Cannot fit {model_name} imputer; learned statistics are invalid.")
    return imputer


def prepare_features(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    model_name: str,
    seq_len: int,
) -> PreparedFeatures:
    assert_no_forbidden_features(feature_names, model_name)
    validate_split_lengths(splits, seq_len, model_name)

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

    imputer = fit_train_imputer(coerced["train"], feature_names, model_name)
    imputed_arrays = {
        split: imputer.transform(clean_feature_frame(df, feature_names)).astype(float)
        for split, df in coerced.items()
    }
    scaler = StandardScaler()
    scaler.fit(imputed_arrays["train"])
    transformed = {
        split: scaler.transform(values).astype(np.float32)
        for split, values in imputed_arrays.items()
    }
    return PreparedFeatures(
        transformed=transformed,
        scaler=scaler,
        imputer=imputer,
        imputation_messages=imputation_messages,
    )


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


def build_sequence_data(
    splits: dict[str, pd.DataFrame],
    transformed: dict[str, np.ndarray],
    seq_len: int,
) -> SequenceData:
    x_train, y_train, _train_meta = create_sequences(splits["train"], transformed["train"], seq_len, "train")
    x_validation, y_validation, validation_meta = create_sequences(
        splits["validation"], transformed["validation"], seq_len, "validation"
    )
    x_test, y_test, test_meta = create_sequences(splits["test"], transformed["test"], seq_len, "test")
    return SequenceData(
        x_train=x_train,
        y_train_raw=y_train,
        x_validation=x_validation,
        y_validation_raw=y_validation,
        validation_meta=validation_meta,
        x_test=x_test,
        y_test_raw=y_test,
        test_meta=test_meta,
        counts={
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
    )


def fit_target_values(y_raw: np.ndarray, target_mode: str) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.float32)
    if target_mode == "log":
        return np.log(y + EPSILON).astype(np.float32)
    if target_mode == "raw":
        return y.astype(np.float32)
    raise ValueError(f"Unsupported target mode: {target_mode}")


def softplus_inverse(value: float) -> float:
    value = max(float(value), EPSILON)
    if value > 20:
        return value
    return float(np.log(np.expm1(value)))


def build_lstm_model(config: VariantConfig, num_features: int, output_bias_value: float) -> tf.keras.Model:
    output_activation = "linear" if config.target_mode == "log" else "softplus"
    loss: str | Any = "mse" if config.loss_name == "mse_log" else qlike_loss

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.seq_len, num_features)),
            tf.keras.layers.LSTM(config.units, return_sequences=False),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(config.dense_units, activation="relu"),
            tf.keras.layers.Dense(
                1,
                activation=output_activation,
                bias_initializer=tf.keras.initializers.Constant(output_bias_value),
            ),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_model(
    model: tf.keras.Model,
    sequence_data: SequenceData,
    config: VariantConfig,
) -> tf.keras.callbacks.History:
    y_train = fit_target_values(sequence_data.y_train_raw, config.target_mode)
    y_validation = fit_target_values(sequence_data.y_validation_raw, config.target_mode)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    return model.fit(
        sequence_data.x_train,
        y_train,
        validation_data=(sequence_data.x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )


def inverse_predictions(raw_predictions: np.ndarray, target_mode: str) -> np.ndarray:
    pred = np.asarray(raw_predictions, dtype=float).reshape(-1)
    if not np.isfinite(pred).all():
        bad_count = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Model produced {bad_count} non-finite raw predictions.")
    if target_mode == "log":
        pred_log = np.clip(pred, np.log(EPSILON), 30.0)
        pred = np.exp(pred_log) - EPSILON
    elif target_mode != "raw":
        raise ValueError(f"Unsupported target mode: {target_mode}")
    pred = np.clip(pred, EPSILON, None)
    if not np.isfinite(pred).all():
        bad_count = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Model produced {bad_count} non-finite variance predictions.")
    return pred.astype(float)


def predict_model(model: tf.keras.Model, x_values: np.ndarray, config: VariantConfig) -> np.ndarray:
    raw_predictions = model.predict(x_values, verbose=0).reshape(-1)
    return inverse_predictions(raw_predictions, config.target_mode)


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
        raise ValueError(f"{model_name} {split} predictions contain non-finite values.")
    if (frame["pred_var"] <= 0).any():
        raise ValueError(f"{model_name} {split} predictions contain non-positive values.")
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
    history_df = pd.DataFrame(history.history).copy()
    if history_df.empty:
        raise ValueError("Training history is empty.")
    history_df = history_df.rename(columns={"learning_rate": "lr"})
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(path, index=False)
    return history_df


def plot_training_history(history_df: pd.DataFrame, path: Path, config: VariantConfig) -> None:
    if "loss" not in history_df.columns or "val_loss" not in history_df.columns:
        raise ValueError(f"{config.model_name} history is missing loss/val_loss columns.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["loss"], label="Training loss", linewidth=1.5)
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=1.5)
    ax.set_title(f"{config.model_name} training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("QLIKE loss" if config.loss_name == "qlike" else "MSE on log variance")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_feature_list(feature_names: list[str], path: Path) -> None:
    path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")


def run_variant(
    config: VariantConfig,
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> ModelRunResult:
    tf.keras.backend.clear_session()
    set_seeds(SEED)

    prepared = prepare_features(splits, feature_names, config.model_name, config.seq_len)
    sequence_data = build_sequence_data(splits, prepared.transformed, config.seq_len)
    y_train_fit = fit_target_values(sequence_data.y_train_raw, config.target_mode)
    if config.target_mode == "log":
        output_bias = float(np.mean(y_train_fit))
    else:
        output_bias = softplus_inverse(float(np.mean(sequence_data.y_train_raw)))

    model = build_lstm_model(config, len(feature_names), output_bias)
    history = train_model(model, sequence_data, config)

    paths = variant_paths(config)
    history_df = save_history(history, paths["history"])
    plot_training_history(history_df, paths["figure"], config)

    validation_pred = predict_model(model, sequence_data.x_validation, config)
    test_pred = predict_model(model, sequence_data.x_test, config)
    predictions = pd.concat(
        [
            make_prediction_frame(
                sequence_data.validation_meta,
                validation_pred,
                model_name=config.model_name,
                split="validation",
            ),
            make_prediction_frame(
                sequence_data.test_meta,
                test_pred,
                model_name=config.model_name,
                split="test",
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(paths["predictions"], index=False, float_format="%.17g")

    metrics = evaluate_prediction_frame(predictions)
    model.save(paths["model"])
    joblib.dump(prepared.scaler, paths["scaler"])
    joblib.dump(prepared.imputer, paths["imputer"])
    save_feature_list(feature_names, paths["features"])

    return ModelRunResult(
        config=config,
        predictions=predictions,
        metrics=metrics,
        history=history_df,
        sequence_counts=sequence_data.counts,
        paths=paths,
        feature_names=feature_names,
        imputation_messages=prepared.imputation_messages,
    )


def print_metrics(metrics: pd.DataFrame, title: str) -> None:
    display = metrics.copy()
    for column in ["mse", "rmse", "mae", "qlike", "mean_actual_var", "mean_pred_var"]:
        display[column] = display[column].map(lambda value: f"{float(value):.6g}")
    print(title)
    print(display.to_string(index=False))


def plot_qlike_comparison(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    plot_df = metrics[["model", "split", "qlike"]].copy()
    pivot = plot_df.pivot(index="model", columns="split", values="qlike")
    ordered_models = (
        metrics.loc[metrics["split"] == "validation"]
        .sort_values(["qlike", "model"])["model"]
        .drop_duplicates()
        .tolist()
    )
    pivot = pivot.reindex(ordered_models)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot))
    width = 0.38
    if "validation" in pivot.columns:
        ax.bar(x - width / 2, pivot["validation"], width, label="Validation", color="#3B6EA8")
    if "test" in pivot.columns:
        ax.bar(x + width / 2, pivot["test"], width, label="Test", color="#E07A5F")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("QLIKE")
    ax.set_title("Tuned LSTM QLIKE comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def select_best(metrics: pd.DataFrame, model_names: list[str], split: str = "validation") -> pd.Series:
    candidates = metrics[(metrics["model"].isin(model_names)) & (metrics["split"] == split)].copy()
    if candidates.empty:
        raise ValueError(f"No {split} metrics available for candidates: {model_names}")
    return candidates.sort_values(["qlike", "rmse", "mae", "model"]).iloc[0]


def row_for(metrics: pd.DataFrame, model_name: str, split: str) -> pd.Series:
    rows = metrics[(metrics["model"] == model_name) & (metrics["split"] == split)]
    if rows.empty:
        raise ValueError(f"No metrics row for {model_name} {split}.")
    return rows.iloc[0]


def metric_less(candidate: pd.Series, baseline: pd.Series, metric: str) -> bool:
    return bool(float(candidate[metric]) < float(baseline[metric]))


def load_garch_reference() -> tuple[str | None, pd.Series | None]:
    candidates = [
        PROJECT_ROOT / "outputs" / "metrics" / "final_model_comparison_common_test_only.csv",
        PROJECT_ROOT / "outputs" / "metrics" / "econometric_metrics_test_only.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "model" not in frame.columns:
            continue
        rows = frame[frame["model"].astype(str) == "GARCH(1,1)"]
        if rows.empty:
            continue
        return rel_path(path), rows.iloc[0]
    return None, None


def prefixed_metrics(prefix: str, row: pd.Series) -> dict[str, Any]:
    return {
        f"{prefix}_n_obs": int(row["n_obs"]),
        f"{prefix}_mse": float(row["mse"]),
        f"{prefix}_rmse": float(row["rmse"]),
        f"{prefix}_mae": float(row["mae"]),
        f"{prefix}_qlike": float(row["qlike"]),
        f"{prefix}_mean_actual_var": float(row["mean_actual_var"]),
        f"{prefix}_mean_pred_var": float(row["mean_pred_var"]),
    }


def build_selection_summary(
    metrics: pd.DataFrame,
    results: list[ModelRunResult],
) -> pd.DataFrame:
    base_models = [result.config.model_name for result in results if result.config.feature_set == "base"]
    hybrid_models = [result.config.model_name for result in results if result.config.feature_set == "hybrid"]
    best_base_val = select_best(metrics, base_models, "validation")
    best_hybrid_val = select_best(metrics, hybrid_models, "validation")
    best_base_test = row_for(metrics, str(best_base_val["model"]), "test")
    best_hybrid_test = row_for(metrics, str(best_hybrid_val["model"]), "test")

    row: dict[str, Any] = {
        "selection_rule": "best validation QLIKE; test metrics are reported only after selection",
        "best_base_model": str(best_base_val["model"]),
        "best_hybrid_model": str(best_hybrid_val["model"]),
    }
    row.update(prefixed_metrics("best_base_validation", best_base_val))
    row.update(prefixed_metrics("best_base_test", best_base_test))
    row.update(prefixed_metrics("best_hybrid_validation", best_hybrid_val))
    row.update(prefixed_metrics("best_hybrid_test", best_hybrid_test))

    for metric in ("qlike", "rmse", "mae"):
        row[f"best_hybrid_beats_best_base_test_{metric}"] = metric_less(
            best_hybrid_test,
            best_base_test,
            metric,
        )

    garch_source, garch_row = load_garch_reference()
    row["garch_reference_source"] = garch_source or "not_available"
    if garch_row is not None:
        row.update(prefixed_metrics("garch_11_test_reference", garch_row))
        for metric in ("qlike", "rmse", "mae"):
            row[f"best_base_beats_garch_11_test_{metric}"] = metric_less(best_base_test, garch_row, metric)
            row[f"best_hybrid_beats_garch_11_test_{metric}"] = metric_less(best_hybrid_test, garch_row, metric)
            row[f"either_best_neural_beats_garch_11_test_{metric}"] = bool(
                row[f"best_base_beats_garch_11_test_{metric}"]
                or row[f"best_hybrid_beats_garch_11_test_{metric}"]
            )
    else:
        for metric in ("qlike", "rmse", "mae"):
            row[f"best_base_beats_garch_11_test_{metric}"] = pd.NA
            row[f"best_hybrid_beats_garch_11_test_{metric}"] = pd.NA
            row[f"either_best_neural_beats_garch_11_test_{metric}"] = pd.NA

    return pd.DataFrame([row])


def save_failed_variants(failures: list[dict[str, str]]) -> None:
    columns = ["model", "slug", "feature_set", "target_mode", "loss_name", "error", "traceback"]
    pd.DataFrame(failures, columns=columns).to_csv(
        METRICS_DIR / "lstm_tuned_failed_variants.csv",
        index=False,
    )


def leakage_report(base_features: list[str], hybrid_features: list[str]) -> list[str]:
    assert_no_forbidden_features(base_features, "base tuned LSTM")
    assert_no_forbidden_features(hybrid_features, "hybrid tuned LSTM")
    return [
        "target_var_next, target_date, date, and split are excluded from all feature lists.",
        "Median imputers are fit only on training feature rows.",
        "StandardScalers are fit only on training imputed feature rows.",
        "Validation and test sequences are created independently within their own split.",
        "Test data is not passed to model.fit, EarlyStopping, or ReduceLROnPlateau.",
        "Best tuned base and hybrid models are selected by validation QLIKE only.",
        "Prediction actual_var is copied from target_var_next and checked before saving.",
    ]


def main() -> None:
    ensure_dirs()
    cleanup_previous_outputs(VARIANTS)
    set_seeds(SEED)

    base_splits, bridge_splits = load_stage_data()
    base_features, base_feature_messages = get_base_features(base_splits["train"])
    hybrid_features, hybrid_feature_messages = get_hybrid_features(bridge_splits["train"])

    print("Loaded row counts:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: base={len(base_splits[split])}, bridge={len(bridge_splits[split])}")
    print(f"Default sequence length for named variants: {SEQ_LEN}")
    print(f"Training epochs={EPOCHS}, batch_size={BATCH_SIZE}, shuffle=False")
    print(f"Base features ({len(base_features)}): {', '.join(base_features)}")
    print(f"Hybrid features ({len(hybrid_features)}): {', '.join(hybrid_features)}")
    print("Optional seq_len/unit/dropout grid is not run by default to keep runtime bounded.")
    for message in base_feature_messages + hybrid_feature_messages:
        print(f"Feature decision: {message}")

    results: list[ModelRunResult] = []
    failures: list[dict[str, str]] = []
    for config in VARIANTS:
        print(
            f"\nTraining {config.model_name}: "
            f"features={config.feature_set}, target={config.target_mode}, loss={config.loss_name}, "
            f"seq_len={config.seq_len}, units={config.units}, dropout={config.dropout}, "
            f"dense={config.dense_units}"
        )
        try:
            splits = base_splits if config.feature_set == "base" else bridge_splits
            features = base_features if config.feature_set == "base" else hybrid_features
            result = run_variant(config, splits, features)
            results.append(result)
            print(
                f"Completed {config.model_name}: "
                f"epochs={len(result.history)}, "
                f"train_seq={result.sequence_counts['train']}, "
                f"validation_seq={result.sequence_counts['validation']}, "
                f"test_seq={result.sequence_counts['test']}"
            )
            print_metrics(result.metrics, f"{config.model_name} metrics:")
        except Exception as exc:  # pragma: no cover - intended runtime robustness.
            failure = {
                "model": config.model_name,
                "slug": config.slug,
                "feature_set": config.feature_set,
                "target_mode": config.target_mode,
                "loss_name": config.loss_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"FAILED {config.model_name}: {exc}")

    save_failed_variants(failures)

    if not results:
        raise RuntimeError("All tuned LSTM variants failed. See lstm_tuned_failed_variants.csv.")
    if not any(result.config.feature_set == "base" for result in results):
        raise RuntimeError("No tuned base LSTM variant completed successfully.")
    if not any(result.config.feature_set == "hybrid" for result in results):
        raise RuntimeError("No tuned hybrid LSTM variant completed successfully.")

    all_metrics = pd.concat([result.metrics for result in results], ignore_index=True)
    all_metrics.to_csv(METRICS_DIR / "lstm_tuned_metrics.csv", index=False)
    test_only = (
        all_metrics.loc[all_metrics["split"] == "test"]
        .sort_values(["qlike", "rmse", "mae", "model"])
        .reset_index(drop=True)
    )
    test_only.to_csv(METRICS_DIR / "lstm_tuned_metrics_test_only.csv", index=False)
    selection_summary = build_selection_summary(all_metrics, results)
    selection_summary.to_csv(METRICS_DIR / "lstm_tuned_selection_summary.csv", index=False)
    plot_qlike_comparison(all_metrics, FIGURES_DIR / "fig_lstm_tuned_qlike_comparison.png")

    print("\nImputation decisions:")
    for result in results:
        for message in result.imputation_messages:
            print(f"  {message}")

    print("\nLeakage checks:")
    for message in leakage_report(base_features, hybrid_features):
        print(f"  PASS: {message}")

    print_metrics(all_metrics.sort_values(["split", "qlike", "rmse"]), "\nAll tuned metrics:")
    print_metrics(test_only, "\nTest metrics sorted by QLIKE then RMSE:")

    print("\nSelection summary:")
    display_cols = [
        "best_base_model",
        "best_base_validation_qlike",
        "best_base_test_qlike",
        "best_hybrid_model",
        "best_hybrid_validation_qlike",
        "best_hybrid_test_qlike",
        "best_hybrid_beats_best_base_test_qlike",
        "garch_reference_source",
    ]
    print(selection_summary[display_cols].to_string(index=False))

    saved_paths = [
        METRICS_DIR / "lstm_tuned_metrics.csv",
        METRICS_DIR / "lstm_tuned_metrics_test_only.csv",
        METRICS_DIR / "lstm_tuned_selection_summary.csv",
        METRICS_DIR / "lstm_tuned_failed_variants.csv",
        FIGURES_DIR / "fig_lstm_tuned_qlike_comparison.png",
    ]
    for result in results:
        saved_paths.extend(result.paths.values())

    print("\nSaved output paths:")
    for path in saved_paths:
        print(f"  {rel_path(path)}")

    if failures:
        print("\nFailed variants were logged:")
        for failure in failures:
            print(f"  {failure['model']}: {failure['error']}")
    else:
        print("\nFailed variants: none")


if __name__ == "__main__":
    main()
