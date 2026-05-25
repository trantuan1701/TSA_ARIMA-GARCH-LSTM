#!/usr/bin/env python3
"""Train Stage 2 econometric volatility forecasting models."""

from __future__ import annotations

import itertools
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "The 'arch' package is required for Stage 2. Install it with: "
        "python -m pip install arch"
    ) from exc

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise ImportError(
        "The 'statsmodels' package is required for Stage 2. Install it with: "
        "python -m pip install statsmodels"
    ) from exc

from metrics import EPSILON, evaluate_volatility_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_READY_PATH = PROCESSED_DIR / "vnindex_model_ready.csv"

REQUIRED_COLUMNS = [
    "date",
    "target_date",
    "log_return_pct",
    "squared_return",
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

MODEL_FILES = {
    "HistoricalMean": "pred_baseline_mean.csv",
    "RollingVol-5": "pred_rolling_vol_5.csv",
    "RollingVol-10": "pred_rolling_vol_10.csv",
    "RollingVol-20": "pred_rolling_vol_20.csv",
    "GARCH(1,1)": "pred_garch_11.csv",
    "ARIMA-GARCH": "pred_arima_garch.csv",
}

CONTRACT_TEXT = """# Experiment Contract

## 1. Research Objective

This project forecasts one-step-ahead VN-Index volatility, not the VN-Index price level.

## 2. Target Definition

`target_var_next = squared_return.shift(-1)`

In every prediction file, `actual_var` must equal `target_var_next`.

## 3. Date Semantics

- `date` = prediction origin date, meaning the day whose information is available.
- `target_date` = date being forecasted.
- `target_date` is usually the next trading day.
- For row `t`, features must only use information available up to `date t`.

## 4. Forbidden Predictors

The following columns must never be used as model input features:

- `target_var_next`
- `target_date`

They can be used only for labels, alignment, and evaluation.

## 5. Chronological Split

Use the existing Stage 1 files:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

Do not reshuffle.

Do not create random train/test splits.

## 6. No Leakage Rules

- Do not use test data for model selection.
- Do not fit scalers on validation or test data.
- Rolling features must be trailing-only.
- For econometric models, parameters should be estimated using training data only.
- Validation and test forecasts may use past observed returns recursively, but must not refit parameters on validation/test data unless explicitly documented as a rolling refit experiment.
- For Stage 2, use fixed-parameter forecasting as the default.

## 7. Prediction File Schema

Every prediction CSV must use this schema:

```text
date,target_date,actual_var,pred_var,model,split
```

Where:

- `actual_var = target_var_next`
- `pred_var` = predicted next-day variance
- `model` = canonical model name
- `split` = `validation` or `test`

## 8. Canonical Model Names

Use exactly these model names:

- `HistoricalMean`
- `RollingVol-5`
- `RollingVol-10`
- `RollingVol-20`
- `GARCH(1,1)`
- `ARIMA-GARCH`
- `LSTM`
- `ARIMA-GARCH-LSTM`

## 9. Evaluation Metrics

Compute:

```text
RMSE = sqrt(mean((actual_var - pred_var)^2))
MAE = mean(abs(actual_var - pred_var))
QLIKE = mean(log(pred_var) + actual_var / pred_var)
```

Before computing QLIKE, clip `pred_var` using `epsilon = 1e-8`.

## 10. Output Directories

Use:

- `outputs/predictions/`
- `outputs/metrics/`
- `outputs/models/`
- `data/processed/`

Create directories automatically.

## 11. Econometric Feature Bridge for Stage 3

Stage 2 must also save a bridge file for LSTM/hybrid modeling:

```text
data/processed/vnindex_with_econometric_features.csv
```

This file should preserve all rows from train, validation, and test and include econometric volatility features generated without using future information.
"""


@dataclass
class GarchParams:
    mean: float
    omega: float
    alpha: float
    beta: float


@dataclass
class GarchForecasts:
    train_pred_var: np.ndarray
    validation_pred_var: np.ndarray
    test_pred_var: np.ndarray
    params: GarchParams
    converged: bool
    warnings: list[str]


@dataclass
class ArimaGarchForecasts:
    train_pred_var: np.ndarray
    validation_pred_var: np.ndarray
    test_pred_var: np.ndarray
    train_arima_mean: np.ndarray
    validation_arima_mean: np.ndarray
    test_arima_mean: np.ndarray
    train_resid_proxy: np.ndarray
    validation_resid_proxy: np.ndarray
    test_resid_proxy: np.ndarray
    garch_params: GarchParams
    garch_converged: bool
    mean_method: str
    warnings: list[str]


def ensure_dirs() -> None:
    for path in (PREDICTIONS_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_experiment_contract() -> Path:
    path = DOCS_DIR / "experiment_contract.md"
    path.write_text(CONTRACT_TEXT, encoding="utf-8")
    return path


def load_splits() -> dict[str, pd.DataFrame]:
    paths = {
        "train": PROCESSED_DIR / "train.csv",
        "validation": PROCESSED_DIR / "validation.csv",
        "test": PROCESSED_DIR / "test.csv",
    }
    splits: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 1 split file does not exist: {path}")
        split = pd.read_csv(path, encoding="utf-8-sig")
        splits[name] = validate_inputs(split, name, path)
    return splits


def load_model_ready() -> pd.DataFrame:
    if not MODEL_READY_PATH.exists():
        raise FileNotFoundError(f"Required Stage 1 model-ready file does not exist: {MODEL_READY_PATH}")
    model_ready = pd.read_csv(MODEL_READY_PATH, encoding="utf-8-sig")
    return validate_inputs(model_ready, "model_ready", MODEL_READY_PATH)


def validate_inputs(df: pd.DataFrame, split_name: str, source_path: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required columns for {split_name}: {missing}")

    validated = df.copy()
    for column in ("date", "target_date"):
        validated[column] = pd.to_datetime(validated[column], errors="coerce")
        if validated[column].isna().any():
            examples = validated.loc[validated[column].isna()].index.tolist()[:10]
            raise ValueError(
                f"Could not parse {column} values in {source_path}; bad row indexes: {examples}"
            )

    numeric_columns = [column for column in REQUIRED_COLUMNS if column not in {"date", "target_date"}]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
            bad_count = int((~np.isfinite(validated[column].to_numpy(dtype=float))).sum())
            raise ValueError(
                f"{source_path} column {column} contains {bad_count} missing or non-finite values."
            )

    validated = validated.sort_values("date").reset_index(drop=True)
    if not validated["date"].is_monotonic_increasing:
        raise ValueError(f"{split_name} split is not sorted chronologically after date parsing.")
    return validated


def train_historical_baseline(train: pd.DataFrame) -> float:
    mean_train_var = float(train["squared_return"].mean())
    if not np.isfinite(mean_train_var) or mean_train_var <= 0:
        raise ValueError(f"Historical mean variance is invalid: {mean_train_var}")
    return mean_train_var


def make_prediction_frame(
    split_df: pd.DataFrame,
    pred_var: np.ndarray | pd.Series | list[float],
    *,
    model: str,
    split: str,
) -> pd.DataFrame:
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    if len(pred) != len(split_df):
        raise ValueError(
            f"{model} {split} predictions have length {len(pred)} but split has {len(split_df)} rows."
        )
    if not np.isfinite(pred).all():
        bad_count = int((~np.isfinite(pred)).sum())
        raise ValueError(f"{model} {split} predictions contain {bad_count} non-finite values.")

    frame = pd.DataFrame(
        {
            "date": split_df["date"],
            "target_date": split_df["target_date"],
            "actual_var": split_df["target_var_next"].astype(float),
            "pred_var": np.clip(pred, EPSILON, None),
            "model": model,
            "split": split,
        }
    )
    return frame[PREDICTION_COLUMNS]


def build_baseline_predictions(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    mean_train_var = train_historical_baseline(train)
    predictions: dict[str, pd.DataFrame] = {
        "HistoricalMean": pd.concat(
            [
                make_prediction_frame(
                    validation,
                    np.repeat(mean_train_var, len(validation)),
                    model="HistoricalMean",
                    split="validation",
                ),
                make_prediction_frame(
                    test,
                    np.repeat(mean_train_var, len(test)),
                    model="HistoricalMean",
                    split="test",
                ),
            ],
            ignore_index=True,
        )
    }

    for window in (5, 10, 20):
        model_name = f"RollingVol-{window}"
        column = f"rolling_vol_{window}"
        predictions[model_name] = pd.concat(
            [
                make_prediction_frame(
                    validation,
                    validation[column].astype(float).to_numpy() ** 2,
                    model=model_name,
                    split="validation",
                ),
                make_prediction_frame(
                    test,
                    test[column].astype(float).to_numpy() ** 2,
                    model=model_name,
                    split="test",
                ),
            ],
            ignore_index=True,
        )

    return predictions


def fit_garch_11(train: pd.DataFrame) -> tuple[Any, list[str]]:
    returns = train["log_return_pct"].astype(float)
    caught_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = arch_model(
            returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=True)
    caught_messages.extend(format_warning_messages(caught))
    return result, caught_messages


def format_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    messages = []
    for warning_msg in caught:
        messages.append(f"{warning_msg.category.__name__}: {warning_msg.message}")
    return messages


def extract_garch_params(result: Any, *, mean_default: float = 0.0) -> GarchParams:
    params = result.params

    def get_param(names: tuple[str, ...], default: float | None = None) -> float:
        for name in names:
            if name in params.index:
                return float(params[name])
        lowered = {str(index).lower(): index for index in params.index}
        for name in names:
            key = name.lower()
            if key in lowered:
                return float(params[lowered[key]])
        if default is not None:
            return default
        raise ValueError(f"Could not find required GARCH parameter among names: {names}")

    return GarchParams(
        mean=get_param(("mu", "Const", "constant"), mean_default),
        omega=get_param(("omega",)),
        alpha=get_param(("alpha[1]", "alpha1")),
        beta=get_param(("beta[1]", "beta1")),
    )


def garch_converged(result: Any) -> bool:
    if hasattr(result, "convergence_flag"):
        return int(result.convergence_flag) == 0
    optimization_result = getattr(result, "optimization_result", None)
    if optimization_result is not None and hasattr(optimization_result, "success"):
        return bool(optimization_result.success)
    return False


def forecast_garch_from_residuals(
    residuals: np.ndarray,
    *,
    initial_h: float,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    forecasts: list[float] = []
    current_h = float(np.clip(initial_h, EPSILON, None))
    for residual in residuals:
        next_h = omega + alpha * float(residual) ** 2 + beta * current_h
        if not np.isfinite(next_h):
            next_h = EPSILON
        next_h = float(np.clip(next_h, EPSILON, None))
        forecasts.append(next_h)
        current_h = next_h
    return np.asarray(forecasts, dtype=float)


def build_future_state_frame(splits: dict[str, pd.DataFrame], model_ready: pd.DataFrame) -> pd.DataFrame:
    """Return all post-training rows needed to update recursive forecast state.

    Some rows are intentionally excluded from validation/test evaluation because their
    target_date crosses a split boundary. Their observed returns are still known by
    later prediction origins and must update the fixed-parameter GARCH recursion.
    """

    train_last_date = splits["train"]["date"].max()
    future_state = model_ready.loc[model_ready["date"] > train_last_date].copy()
    if future_state.empty:
        raise ValueError("No model-ready rows remain after the training split for recursive forecasting.")

    if future_state.duplicated(["date", "target_date"]).any():
        duplicates = future_state.loc[
            future_state.duplicated(["date", "target_date"], keep=False),
            ["date", "target_date"],
        ].head(10)
        raise ValueError(
            "Model-ready data has duplicate date/target_date rows needed for recursive forecasting:\n"
            f"{duplicates.to_string(index=False)}"
        )

    future_keys = future_state[["date", "target_date"]]
    for split_name in ("validation", "test"):
        missing = splits[split_name][["date", "target_date"]].merge(
            future_keys,
            on=["date", "target_date"],
            how="left",
            indicator=True,
        )
        missing = missing.loc[missing["_merge"] == "left_only", ["date", "target_date"]]
        if not missing.empty:
            raise ValueError(
                f"{split_name} split contains rows missing from model-ready future state:\n"
                f"{missing.head(10).to_string(index=False)}"
            )

    return future_state.sort_values("date").reset_index(drop=True)


def align_future_values_to_split(
    future_values: pd.DataFrame,
    split_df: pd.DataFrame,
    value_column: str,
    split_name: str,
) -> np.ndarray:
    merged = split_df[["date", "target_date"]].merge(
        future_values[["date", "target_date", value_column]],
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if merged[value_column].isna().any():
        missing = merged.loc[merged[value_column].isna(), ["date", "target_date"]].head(10)
        raise ValueError(
            f"Could not align {value_column} forecasts to {split_name} split rows:\n"
            f"{missing.to_string(index=False)}"
        )
    return merged[value_column].to_numpy(dtype=float)


def forecast_garch_fixed_params(
    splits: dict[str, pd.DataFrame],
    result: Any,
    model_ready: pd.DataFrame,
) -> GarchForecasts:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    params = extract_garch_params(result)
    train_returns = train["log_return_pct"].astype(float).to_numpy()
    train_residuals = train_returns - params.mean
    train_cond_var = np.asarray(result.conditional_volatility, dtype=float) ** 2
    train_pred_var = forecast_next_from_current_state(train_residuals, train_cond_var, params)

    future_state = build_future_state_frame(splits, model_ready)
    future_residuals = future_state["log_return_pct"].astype(float).to_numpy() - params.mean
    future_pred_var = forecast_garch_from_residuals(
        future_residuals,
        initial_h=float(train_pred_var[-1]),
        omega=params.omega,
        alpha=params.alpha,
        beta=params.beta,
    )
    future_values = future_state[["date", "target_date"]].copy()
    future_values["pred_var"] = future_pred_var

    validation_pred_var = align_future_values_to_split(
        future_values,
        validation,
        "pred_var",
        "validation",
    )
    test_pred_var = align_future_values_to_split(future_values, test, "pred_var", "test")
    return GarchForecasts(
        train_pred_var=train_pred_var,
        validation_pred_var=validation_pred_var,
        test_pred_var=test_pred_var,
        params=params,
        converged=garch_converged(result),
        warnings=[],
    )


def forecast_next_from_current_state(
    residuals: np.ndarray,
    current_variances: np.ndarray,
    params: GarchParams,
) -> np.ndarray:
    forecasts = (
        params.omega + params.alpha * np.asarray(residuals, dtype=float) ** 2 + params.beta * current_variances
    )
    forecasts = np.where(np.isfinite(forecasts), forecasts, EPSILON)
    return np.clip(forecasts.astype(float), EPSILON, None)


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def save_garch_summary(result: Any, warnings_list: list[str], path: Path) -> None:
    params = extract_garch_params(result)
    lines = [
        "GARCH(1,1) summary",
        "",
        "Fixed-parameter forecasting: parameters are estimated on the training split only. "
        "Validation and test forecasts recursively update the conditional variance with every "
        "post-training observed return needed to reach each prediction origin, without refitting.",
        "",
        f"Converged: {garch_converged(result)}",
        f"mu: {params.mean:.12g}",
        f"omega: {params.omega:.12g}",
        f"alpha[1]: {params.alpha:.12g}",
        f"beta[1]: {params.beta:.12g}",
        "",
        "Fit warnings:",
        *(warnings_list or ["None"]),
        "",
        str(result.summary()),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def select_arima_order(train: pd.DataFrame) -> tuple[Any, tuple[int, int, int], pd.DataFrame, list[str]]:
    train_returns = pd.Series(train["log_return_pct"].astype(float).to_numpy(), name="log_return_pct")
    rows: list[dict[str, Any]] = []
    best_result: Any | None = None
    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf
    selected_warnings: list[str] = []

    for p, q in itertools.product(range(4), range(4)):
        order = (p, 0, q)
        row: dict[str, Any] = {
            "p": p,
            "d": 0,
            "q": q,
            "aic": np.nan,
            "bic": np.nan,
            "converged": pd.NA,
            "error": "",
            "warnings": "",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = ARIMA(train_returns, order=order).fit()
                row["aic"] = float(result.aic)
                row["bic"] = float(result.bic)
                row["converged"] = get_arima_converged(result)
                row["warnings"] = " | ".join(format_warning_messages(caught))
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_order = order
                    best_result = result
                    selected_warnings = format_warning_messages(caught)
            except Exception as exc:  # Keep failed candidates in the selection log.
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["warnings"] = " | ".join(format_warning_messages(caught))
        rows.append(row)

    selection = pd.DataFrame(rows)
    if best_result is None or best_order is None:
        raise RuntimeError(
            "All ARIMA(p,0,q) candidates failed on the training split. "
            f"See {METRICS_DIR / 'arima_order_selection.csv'} for candidate errors."
        )
    return best_result, best_order, selection, selected_warnings


def get_arima_converged(result: Any) -> bool | object:
    mle_retvals = getattr(result, "mle_retvals", None)
    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])
    return pd.NA


def save_arima_summary(
    result: Any,
    order: tuple[int, int, int],
    selected_warnings: list[str],
    path: Path,
) -> None:
    lines = [
        "ARIMA order selection summary",
        "",
        "Order selection grid: p in [0, 1, 2, 3], d = 0, q in [0, 1, 2, 3].",
        "Selection criterion: lowest training AIC among successfully fitted models.",
        "Validation and test data are not used for order selection.",
        "",
        f"Selected order: {order}",
        f"AIC: {float(result.aic):.12g}",
        f"BIC: {float(result.bic):.12g}",
        f"Converged: {get_arima_converged(result)}",
        "",
        "Selected fit warnings:",
        *(selected_warnings or ["None"]),
        "",
        str(result.summary()),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_arima_garch(
    splits: dict[str, pd.DataFrame],
    arima_result: Any,
    arima_order: tuple[int, int, int],
    model_ready: pd.DataFrame,
) -> tuple[Any, ArimaGarchForecasts]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    train_returns = train["log_return_pct"].astype(float).to_numpy()
    future_state = build_future_state_frame(splits, model_ready)
    future_returns = future_state["log_return_pct"].astype(float).to_numpy()

    train_mean = arima_fitted_values(arima_result, len(train), train_returns)
    future_mean, mean_method, mean_warnings = arima_future_one_step_means(
        arima_result,
        future_returns,
        train_length=len(train),
        train_returns=train_returns,
    )

    train_resid = train_returns - train_mean
    future_resid = future_returns - future_mean
    train_resid = replace_nonfinite_with_centered_returns(train_resid, train_returns)

    caught_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = arch_model(
            pd.Series(train_resid, name="arima_resid"),
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        garch_result = model.fit(disp="off", show_warning=True)
    caught_messages.extend(format_warning_messages(caught))

    garch_params = extract_garch_params(garch_result, mean_default=0.0)
    train_cond_var = np.asarray(garch_result.conditional_volatility, dtype=float) ** 2
    train_pred_var = forecast_next_from_current_state(train_resid, train_cond_var, garch_params)
    future_pred_var = forecast_garch_from_residuals(
        future_resid,
        initial_h=float(train_pred_var[-1]),
        omega=garch_params.omega,
        alpha=garch_params.alpha,
        beta=garch_params.beta,
    )
    future_values = future_state[["date", "target_date"]].copy()
    future_values["pred_var"] = future_pred_var
    future_values["arima_mean"] = future_mean
    future_values["arima_resid"] = future_resid

    validation_pred_var = align_future_values_to_split(
        future_values,
        validation,
        "pred_var",
        "validation",
    )
    test_pred_var = align_future_values_to_split(future_values, test, "pred_var", "test")
    validation_mean = align_future_values_to_split(
        future_values,
        validation,
        "arima_mean",
        "validation",
    )
    test_mean = align_future_values_to_split(future_values, test, "arima_mean", "test")
    validation_resid = align_future_values_to_split(
        future_values,
        validation,
        "arima_resid",
        "validation",
    )
    test_resid = align_future_values_to_split(future_values, test, "arima_resid", "test")

    all_warnings = mean_warnings + caught_messages
    forecasts = ArimaGarchForecasts(
        train_pred_var=train_pred_var,
        validation_pred_var=validation_pred_var,
        test_pred_var=test_pred_var,
        train_arima_mean=train_mean,
        validation_arima_mean=validation_mean,
        test_arima_mean=test_mean,
        train_resid_proxy=train_resid,
        validation_resid_proxy=validation_resid,
        test_resid_proxy=test_resid,
        garch_params=garch_params,
        garch_converged=garch_converged(garch_result),
        mean_method=mean_method,
        warnings=all_warnings,
    )
    save_arima_garch_summary(
        arima_order=arima_order,
        garch_result=garch_result,
        forecasts=forecasts,
        path=METRICS_DIR / "arima_garch_summary.txt",
    )
    return garch_result, forecasts


def arima_fitted_values(arima_result: Any, expected_length: int, train_returns: np.ndarray) -> np.ndarray:
    fitted = np.asarray(arima_result.fittedvalues, dtype=float)
    if fitted.shape[0] != expected_length or not np.isfinite(fitted).all():
        fallback = float(np.mean(train_returns))
        return np.repeat(fallback, expected_length)
    return fitted


def arima_future_one_step_means(
    arima_result: Any,
    future_returns: np.ndarray,
    *,
    train_length: int,
    train_returns: np.ndarray,
) -> tuple[np.ndarray, str, list[str]]:
    if len(future_returns) == 0:
        return np.asarray([], dtype=float), "no future observations", []

    try:
        future_index = pd.RangeIndex(start=train_length, stop=train_length + len(future_returns))
        future_series = pd.Series(future_returns, index=future_index, name="log_return_pct")
        extended_result = arima_result.append(future_series, refit=False)
        fitted = np.asarray(extended_result.fittedvalues, dtype=float)
        future_means = fitted[train_length:]
        if future_means.shape[0] != len(future_returns) or not np.isfinite(future_means).all():
            raise ValueError("statsmodels append returned invalid future fitted values.")
        method = "statsmodels ARIMAResults.append(..., refit=False) one-step fitted values"
        return future_means, method, []
    except Exception as exc:
        fallback = float(np.mean(train_returns))
        message = (
            "ARIMA future one-step mean prediction fallback used: "
            f"{type(exc).__name__}: {exc}. Replaced with training-sample mean return."
        )
        return np.repeat(fallback, len(future_returns)), "training mean fallback", [message]


def replace_nonfinite_with_centered_returns(residuals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float).copy()
    if np.isfinite(residuals).all():
        return residuals
    fallback_mean = float(np.mean(returns))
    fallback_resid = returns - fallback_mean
    bad_mask = ~np.isfinite(residuals)
    residuals[bad_mask] = fallback_resid[bad_mask]
    return residuals


def save_arima_garch_summary(
    *,
    arima_order: tuple[int, int, int],
    garch_result: Any,
    forecasts: ArimaGarchForecasts,
    path: Path,
) -> None:
    params = forecasts.garch_params
    lines = [
        "ARIMA-GARCH summary",
        "",
        "ARIMA-GARCH implementation:",
        "1. Fit ARIMA(p,0,q) on training log returns, with p and q selected by training AIC.",
        "2. Fit GARCH(1,1) on training ARIMA residuals only.",
        "3. Generate validation/test variance forecasts with fixed ARIMA and GARCH parameters. "
        "Every post-training observed return needed to reach each prediction origin updates the "
        "recursive state; parameters are not refit.",
        "",
        f"Selected ARIMA order: {arima_order}",
        f"ARIMA one-step mean method: {forecasts.mean_method}",
        f"GARCH converged: {forecasts.garch_converged}",
        f"omega: {params.omega:.12g}",
        f"alpha[1]: {params.alpha:.12g}",
        f"beta[1]: {params.beta:.12g}",
        "",
        "Warnings/fallbacks:",
        *(forecasts.warnings or ["None"]),
        "",
        str(garch_result.summary()),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_model_prediction_frames(
    splits: dict[str, pd.DataFrame],
    baseline_predictions: dict[str, pd.DataFrame],
    garch_forecasts: GarchForecasts,
    arima_garch_forecasts: ArimaGarchForecasts,
) -> dict[str, pd.DataFrame]:
    validation = splits["validation"]
    test = splits["test"]
    predictions = dict(baseline_predictions)
    predictions["GARCH(1,1)"] = pd.concat(
        [
            make_prediction_frame(
                validation,
                garch_forecasts.validation_pred_var,
                model="GARCH(1,1)",
                split="validation",
            ),
            make_prediction_frame(
                test,
                garch_forecasts.test_pred_var,
                model="GARCH(1,1)",
                split="test",
            ),
        ],
        ignore_index=True,
    )
    predictions["ARIMA-GARCH"] = pd.concat(
        [
            make_prediction_frame(
                validation,
                arima_garch_forecasts.validation_pred_var,
                model="ARIMA-GARCH",
                split="validation",
            ),
            make_prediction_frame(
                test,
                arima_garch_forecasts.test_pred_var,
                model="ARIMA-GARCH",
                split="test",
            ),
        ],
        ignore_index=True,
    )
    return predictions


def validate_prediction_frames(
    predictions: dict[str, pd.DataFrame],
    splits: dict[str, pd.DataFrame],
) -> None:
    for model, frame in predictions.items():
        if list(frame.columns) != PREDICTION_COLUMNS:
            raise ValueError(f"{model} prediction schema mismatch: {list(frame.columns)}")
        if set(frame["split"].unique()) != {"validation", "test"}:
            raise ValueError(f"{model} predictions must contain validation and test rows only.")
        if set(frame["model"].unique()) != {model}:
            raise ValueError(f"{model} prediction file contains inconsistent model names.")
        pred = pd.to_numeric(frame["pred_var"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError(f"{model} predictions contain non-finite or non-positive pred_var values.")

        for split_name in ("validation", "test"):
            expected = splits[split_name]
            observed = frame.loc[frame["split"] == split_name].reset_index(drop=True)
            if len(observed) != len(expected):
                raise ValueError(
                    f"{model} {split_name} prediction row count {len(observed)} "
                    f"does not match split row count {len(expected)}."
                )
            if not observed["date"].reset_index(drop=True).equals(expected["date"].reset_index(drop=True)):
                raise ValueError(f"{model} {split_name} prediction dates do not match split dates.")
            if not observed["target_date"].reset_index(drop=True).equals(
                expected["target_date"].reset_index(drop=True)
            ):
                raise ValueError(f"{model} {split_name} prediction target dates do not match split dates.")
            if not np.allclose(
                observed["actual_var"].astype(float).to_numpy(),
                expected["target_var_next"].astype(float).to_numpy(),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(f"{model} {split_name} actual_var does not equal target_var_next.")


def save_prediction_files(predictions: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model, filename in MODEL_FILES.items():
        path = PREDICTIONS_DIR / filename
        predictions[model].to_csv(path, index=False, encoding="utf-8-sig")
        paths[model] = path
    return paths


def evaluate_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_FILES:
        frame = predictions[model]
        for split_name in ("validation", "test"):
            split_frame = frame.loc[frame["split"] == split_name]
            metrics = evaluate_volatility_predictions(
                split_frame["actual_var"],
                split_frame["pred_var"],
                label=f"{model} {split_name}",
            )
            rows.append({"model": model, "split": split_name, **metrics})
    return pd.DataFrame(rows)[METRIC_COLUMNS]


def save_metrics(metrics: pd.DataFrame) -> dict[str, Path]:
    metrics_path = METRICS_DIR / "econometric_metrics.csv"
    test_only_path = METRICS_DIR / "econometric_metrics_test_only.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    test_only = metrics.loc[metrics["split"] == "test"].sort_values(
        ["qlike", "rmse"], ascending=[True, True]
    )
    test_only.to_csv(test_only_path, index=False, encoding="utf-8-sig")
    return {"metrics": metrics_path, "test_only": test_only_path}


def build_econometric_feature_bridge(
    splits: dict[str, pd.DataFrame],
    garch_forecasts: GarchForecasts,
    arima_garch_forecasts: ArimaGarchForecasts,
) -> pd.DataFrame:
    forecast_map = {
        "train": {
            "garch": garch_forecasts.train_pred_var,
            "arima_garch": arima_garch_forecasts.train_pred_var,
            "arima_mean": arima_garch_forecasts.train_arima_mean,
            "arima_resid": arima_garch_forecasts.train_resid_proxy,
        },
        "validation": {
            "garch": garch_forecasts.validation_pred_var,
            "arima_garch": arima_garch_forecasts.validation_pred_var,
            "arima_mean": arima_garch_forecasts.validation_arima_mean,
            "arima_resid": arima_garch_forecasts.validation_resid_proxy,
        },
        "test": {
            "garch": garch_forecasts.test_pred_var,
            "arima_garch": arima_garch_forecasts.test_pred_var,
            "arima_mean": arima_garch_forecasts.test_arima_mean,
            "arima_resid": arima_garch_forecasts.test_resid_proxy,
        },
    }

    frames: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        split = splits[split_name].copy()
        values = forecast_map[split_name]
        split["split"] = split_name
        split["garch_11_pred_var"] = np.clip(values["garch"], EPSILON, None)
        split["garch_11_pred_vol"] = np.sqrt(split["garch_11_pred_var"])
        split["arima_garch_pred_var"] = np.clip(values["arima_garch"], EPSILON, None)
        split["arima_garch_pred_vol"] = np.sqrt(split["arima_garch_pred_var"])
        split["arima_mean_pred"] = values["arima_mean"]
        split["arima_resid_proxy"] = values["arima_resid"]
        frames.append(split)

    bridge = pd.concat(frames, ignore_index=True)
    bridge_path = PROCESSED_DIR / "vnindex_with_econometric_features.csv"
    bridge.to_csv(bridge_path, index=False, encoding="utf-8-sig")
    return bridge


def validate_bridge(bridge: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    expected_rows = sum(len(split) for split in splits.values())
    required_columns = [
        "split",
        "garch_11_pred_var",
        "garch_11_pred_vol",
        "arima_garch_pred_var",
        "arima_garch_pred_vol",
    ]
    missing = [column for column in required_columns if column not in bridge.columns]
    if missing:
        raise ValueError(f"Econometric feature bridge is missing columns: {missing}")
    if len(bridge) != expected_rows:
        raise ValueError(f"Bridge row count {len(bridge)} does not match expected {expected_rows}.")
    for column in (
        "garch_11_pred_var",
        "garch_11_pred_vol",
        "arima_garch_pred_var",
        "arima_garch_pred_vol",
    ):
        values = pd.to_numeric(bridge[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Bridge column {column} contains non-finite or non-positive values.")


def format_params(params: GarchParams) -> str:
    return (
        f"mu={params.mean:.6g}, omega={params.omega:.6g}, "
        f"alpha[1]={params.alpha:.6g}, beta[1]={params.beta:.6g}"
    )


def print_metrics_summary(metrics: pd.DataFrame, split_name: str) -> None:
    summary = metrics.loc[metrics["split"] == split_name, ["model", "n_obs", "rmse", "mae", "qlike"]]
    print(f"\n{split_name.capitalize()} metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def validate_expected_outputs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected Stage 2 output files were not created: {missing}")
    empty = [path for path in paths if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"Expected Stage 2 output files are empty: {empty}")


def main() -> None:
    ensure_dirs()
    contract_path = write_experiment_contract()
    splits = load_splits()
    model_ready = load_model_ready()
    print(
        "Loaded splits: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )

    baseline_predictions = build_baseline_predictions(splits)

    garch_result, garch_warnings = fit_garch_11(splits["train"])
    save_pickle(garch_result, MODELS_DIR / "garch_11.pkl")
    save_garch_summary(garch_result, garch_warnings, METRICS_DIR / "garch_summary.txt")
    garch_forecasts = forecast_garch_fixed_params(splits, garch_result, model_ready)
    garch_forecasts.warnings.extend(garch_warnings)

    arima_result, arima_order, arima_selection, arima_warnings = select_arima_order(splits["train"])
    arima_selection.to_csv(METRICS_DIR / "arima_order_selection.csv", index=False, encoding="utf-8-sig")
    save_pickle(arima_result, MODELS_DIR / "arima_model.pkl")
    save_arima_summary(arima_result, arima_order, arima_warnings, METRICS_DIR / "arima_summary.txt")

    arima_garch_garch_result, arima_garch_forecasts = fit_arima_garch(
        splits,
        arima_result,
        arima_order,
        model_ready,
    )
    save_pickle(arima_garch_garch_result, MODELS_DIR / "arima_garch_garch.pkl")

    predictions = build_model_prediction_frames(
        splits,
        baseline_predictions,
        garch_forecasts,
        arima_garch_forecasts,
    )
    validate_prediction_frames(predictions, splits)
    prediction_paths = save_prediction_files(predictions)

    metrics = evaluate_predictions(predictions)
    metric_paths = save_metrics(metrics)

    bridge = build_econometric_feature_bridge(splits, garch_forecasts, arima_garch_forecasts)
    validate_bridge(bridge, splits)
    bridge_path = PROCESSED_DIR / "vnindex_with_econometric_features.csv"

    expected_paths = [
        contract_path,
        *prediction_paths.values(),
        metric_paths["metrics"],
        metric_paths["test_only"],
        METRICS_DIR / "arima_order_selection.csv",
        METRICS_DIR / "garch_summary.txt",
        METRICS_DIR / "arima_summary.txt",
        METRICS_DIR / "arima_garch_summary.txt",
        MODELS_DIR / "garch_11.pkl",
        MODELS_DIR / "arima_model.pkl",
        MODELS_DIR / "arima_garch_garch.pkl",
        bridge_path,
    ]
    validate_expected_outputs(expected_paths)

    warnings_and_fallbacks = [
        *(f"GARCH(1,1): {message}" for message in garch_warnings),
        *(f"ARIMA selected fit: {message}" for message in arima_warnings),
        *(f"ARIMA-GARCH: {message}" for message in arima_garch_forecasts.warnings),
    ]

    print(f"Selected ARIMA order: {arima_order}")
    print(f"GARCH(1,1) converged: {garch_forecasts.converged}")
    print(f"GARCH(1,1) parameters: {format_params(garch_forecasts.params)}")
    print(f"ARIMA-GARCH GARCH converged: {arima_garch_forecasts.garch_converged}")
    print(f"ARIMA-GARCH GARCH parameters: {format_params(arima_garch_forecasts.garch_params)}")

    print("\nSaved prediction files:")
    for path in prediction_paths.values():
        print(path)
    print("\nSaved metric files:")
    for path in [
        metric_paths["metrics"],
        metric_paths["test_only"],
        METRICS_DIR / "arima_order_selection.csv",
        METRICS_DIR / "garch_summary.txt",
        METRICS_DIR / "arima_summary.txt",
        METRICS_DIR / "arima_garch_summary.txt",
    ]:
        print(path)
    print(f"\nSaved model files:\n{MODELS_DIR / 'garch_11.pkl'}")
    print(MODELS_DIR / "arima_model.pkl")
    print(MODELS_DIR / "arima_garch_garch.pkl")
    print(f"\nSaved bridge file:\n{bridge_path}")

    print_metrics_summary(metrics, "validation")
    print_metrics_summary(metrics, "test")

    print("\nWarnings/fallbacks:")
    if warnings_and_fallbacks:
        for message in warnings_and_fallbacks:
            print(f"- {message}")
    else:
        print("None")

    print("\nStage 2 econometric modeling complete")


if __name__ == "__main__":
    main()
