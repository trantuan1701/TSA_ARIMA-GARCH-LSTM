#!/usr/bin/env python3
"""Validation-only neural calibration and forecast combination experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    IsotonicRegression = None

from advanced_experiment_utils import (
    ADVANCED_AUDIT_DIR,
    ADVANCED_PREDICTIONS_DIR,
    ADVANCED_TABLES_DIR,
    EPSILON,
    align_model_list,
    align_two_models,
    compute_metrics,
    ensure_advanced_dirs,
    load_all_predictions,
    load_original_predictions,
    loss_values,
    prediction_frame,
    rel_path,
    safe_write_csv,
    save_prediction_csv,
    slugify,
    underprediction_stats,
    write_markdown,
)


NEURAL_MODELS = [
    "LSTM",
    "ARIMA-GARCH-LSTM",
    "LSTM-QLIKE",
    "Hybrid-QLIKE",
    "LSTM-LogTarget",
    "LSTM-LogTarget-Small",
    "Hybrid-LogTarget",
    "Hybrid-LogTarget-Small",
]


@dataclass(frozen=True)
class ComboResult:
    model: str
    method: str
    pool: str
    input_models: list[str]
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    weights: dict[str, float]
    metadata: dict[str, Any]


def qlike_objective(actual: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), EPSILON, None)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.log(pred) + actual / pred))


def calibrate_scale(validation: pd.DataFrame) -> tuple[float, float]:
    actual = validation["actual_var"].to_numpy(dtype=float)
    pred = validation["pred_var"].to_numpy(dtype=float)
    grid = np.logspace(-2.0, 2.0, 401)
    losses = np.array([qlike_objective(actual, c * pred) for c in grid], dtype=float)
    best_index = int(np.argmin(losses))
    return float(grid[best_index]), float(losses[best_index])


def calibrate_affine(validation: pd.DataFrame, *, initial_scale: float) -> tuple[float, float, float, str]:
    actual = validation["actual_var"].to_numpy(dtype=float)
    pred = validation["pred_var"].to_numpy(dtype=float)
    if minimize is None:
        return 0.0, float(initial_scale), qlike_objective(actual, initial_scale * pred), "scipy_unavailable"

    def objective(params: np.ndarray) -> float:
        a, b = params
        return qlike_objective(actual, a + b * pred)

    result = minimize(
        objective,
        x0=np.array([0.0, max(initial_scale, EPSILON)], dtype=float),
        bounds=[(0.0, None), (0.0, None)],
        method="L-BFGS-B",
    )
    if not result.success:
        return 0.0, float(initial_scale), qlike_objective(actual, initial_scale * pred), (
            f"optimizer_failed: {result.message}"
        )
    a, b = result.x
    return float(a), float(b), float(result.fun), "success"


def apply_calibration(df: pd.DataFrame, *, method: str, params: dict[str, Any]) -> np.ndarray:
    pred = df["pred_var"].to_numpy(dtype=float)
    if method in {"scale_qlike", "scale_mean"}:
        return np.clip(float(params["c"]) * pred, EPSILON, None)
    if method == "affine_nonnegative":
        return np.clip(float(params["a"]) + float(params["b"]) * pred, EPSILON, None)
    if method == "isotonic":
        model = params["model"]
        return np.clip(model.predict(pred), EPSILON, None)
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_methods_for_model(validation: pd.DataFrame) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    c_best, scale_loss = calibrate_scale(validation)
    mean_pred = float(validation["pred_var"].mean())
    c_mean = float(validation["actual_var"].mean() / mean_pred) if mean_pred > 0 else 1.0
    a, b, affine_loss, affine_status = calibrate_affine(validation, initial_scale=c_best)
    methods: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "scale_qlike",
            {"c": c_best},
            {"validation_objective": scale_loss, "selection": "validation_QLIKE_grid"},
        ),
        (
            "scale_mean",
            {"c": c_mean},
            {"validation_objective": qlike_objective(validation["actual_var"], c_mean * validation["pred_var"]), "selection": "validation_mean_match"},
        ),
        (
            "affine_nonnegative",
            {"a": a, "b": b},
            {"validation_objective": affine_loss, "selection": "validation_QLIKE_optimization", "optimizer_status": affine_status},
        ),
    ]
    if IsotonicRegression is not None:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON)
        iso.fit(validation["pred_var"].to_numpy(dtype=float), validation["actual_var"].to_numpy(dtype=float))
        methods.append(
            (
                "isotonic",
                {"model": iso},
                {
                    "validation_objective": qlike_objective(
                        validation["actual_var"].to_numpy(dtype=float),
                        iso.predict(validation["pred_var"].to_numpy(dtype=float)),
                    ),
                    "selection": "validation_nonparametric_isotonic",
                    "overfit_caution": "isotonic is fitted on validation and is more flexible than scale/affine calibration",
                },
            )
        )
    return methods


def run_neural_calibration(*, force: bool = False) -> pd.DataFrame:
    ensure_advanced_dirs()
    originals = load_original_predictions(NEURAL_MODELS)
    rows_validation: list[dict[str, Any]] = []
    rows_test: list[dict[str, Any]] = []
    under_rows: list[dict[str, Any]] = []
    prediction_files: list[dict[str, str]] = []

    for base_model in NEURAL_MODELS:
        base = originals[originals["model"] == base_model].copy()
        if base.empty:
            continue
        validation = base[base["split"] == "validation"].copy()
        test = base[base["split"] == "test"].copy()
        if validation.empty or test.empty:
            continue
        methods = calibration_methods_for_model(validation)
        raw_test_stats = underprediction_stats(test)
        raw_val_metrics = compute_metrics(validation)
        raw_test_metrics = compute_metrics(test)
        rows_validation.append(
            {
                "base_model": base_model,
                "calibrated_model": base_model,
                "method": "raw",
                "parameters": "{}",
                **raw_val_metrics,
                "selected_by": "uncalibrated_original",
            }
        )
        rows_test.append(
            {
                "base_model": base_model,
                "calibrated_model": base_model,
                "method": "raw",
                "parameters": "{}",
                **raw_test_metrics,
                "selected_by": "uncalibrated_original",
            }
        )
        under_rows.append(
            {
                "base_model": base_model,
                "model": base_model,
                "method": "raw",
                **raw_test_stats,
                "test_QLIKE": raw_test_metrics["QLIKE"],
                "test_MAE": raw_test_metrics["MAE"],
            }
        )
        for method, params, metadata in methods:
            val_pred = apply_calibration(validation, method=method, params=params)
            test_pred = apply_calibration(test, method=method, params=params)
            model_name = f"Calibrated-{base_model}-{method}"
            val_frame = make_combo_frame(validation, model_name=model_name, split="validation", pred=val_pred)
            test_frame = make_combo_frame(test, model_name=model_name, split="test", pred=test_pred)
            combined = pd.concat([val_frame, test_frame], ignore_index=True)
            path = ADVANCED_PREDICTIONS_DIR / "calibrated_neural" / f"pred_{slugify(model_name)}.csv"
            save_prediction_csv(combined, path, force=force)
            serializable_params = {
                key: value for key, value in params.items() if key != "model"
            }
            serializable_params.update(metadata)
            val_metrics = compute_metrics(val_frame)
            test_metrics = compute_metrics(test_frame)
            rows_validation.append(
                {
                    "base_model": base_model,
                    "calibrated_model": model_name,
                    "method": method,
                    "parameters": json.dumps(serializable_params, sort_keys=True),
                    **val_metrics,
                    "selected_by": metadata["selection"],
                }
            )
            rows_test.append(
                {
                    "base_model": base_model,
                    "calibrated_model": model_name,
                    "method": method,
                    "parameters": json.dumps(serializable_params, sort_keys=True),
                    **test_metrics,
                    "selected_by": metadata["selection"],
                }
            )
            under_rows.append(
                {
                    "base_model": base_model,
                    "model": model_name,
                    "method": method,
                    **underprediction_stats(test_frame, p90=raw_test_stats["extreme_threshold_p90"]),
                    "test_QLIKE": test_metrics["QLIKE"],
                    "test_MAE": test_metrics["MAE"],
                }
            )
            prediction_files.append({"model": model_name, "prediction_file": rel_path(path)})

    val_df = pd.DataFrame(rows_validation)
    test_df = pd.DataFrame(rows_test)
    under_df = pd.DataFrame(under_rows)
    if not val_df.empty:
        val_df["selected_best_for_base"] = False
        calibrated = val_df[val_df["method"] != "raw"].copy()
        if not calibrated.empty:
            best_idx = calibrated.groupby("base_model")["QLIKE"].idxmin()
            val_df.loc[best_idx, "selected_best_for_base"] = True
    safe_write_csv(val_df, ADVANCED_TABLES_DIR / "table_neural_calibration_validation.csv", force=force)
    safe_write_csv(test_df, ADVANCED_TABLES_DIR / "table_neural_calibration_test.csv", force=force)
    safe_write_csv(under_df, ADVANCED_TABLES_DIR / "table_neural_calibration_underprediction.csv", force=force)
    if prediction_files:
        safe_write_csv(
            pd.DataFrame(prediction_files),
            ADVANCED_TABLES_DIR / "table_neural_calibration_prediction_files.csv",
            force=force,
        )
    write_neural_calibration_report(val_df, test_df, under_df, force=force)
    return val_df


def write_neural_calibration_report(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    under: pd.DataFrame,
    *,
    force: bool,
) -> None:
    lines = [
        "# Neural Calibration Report",
        "",
        "## Methods",
        "",
        "- Multiplicative QLIKE scale: `pred_calibrated = c * pred_var`, with `c` selected by validation QLIKE over `logspace(-2, 2, 401)`.",
        "- Mean-matching scale: `c = mean(actual_var_validation) / mean(pred_var_validation)`.",
        "- Nonnegative affine calibration: `max(epsilon, a + b * pred_var)`, with `a >= 0` and `b >= 0` fitted on validation.",
        "- Isotonic calibration is included only when scikit-learn is available and is flagged as more flexible.",
        "",
        "## Best Validation Calibrations",
        "",
    ]
    if validation.empty:
        lines.append("- No neural predictions were available for calibration.")
    else:
        calibrated = validation[validation["method"] != "raw"].copy()
        for base_model, group in calibrated.groupby("base_model"):
            best = group.sort_values(["QLIKE", "RMSE", "calibrated_model"]).iloc[0]
            raw = validation[(validation["base_model"] == base_model) & (validation["method"] == "raw")]
            raw_qlike = float(raw.iloc[0]["QLIKE"]) if not raw.empty else np.nan
            lines.append(
                f"- {base_model}: best `{best['method']}` validation QLIKE {float(best['QLIKE']):.6f}; "
                f"raw validation QLIKE {raw_qlike:.6f}."
            )
    lines.extend(["", "## Log-Target Scale Bias Check", ""])
    log_under = under[under["base_model"].astype(str).str.contains("LogTarget", case=False, regex=False)].copy()
    if log_under.empty:
        lines.append("- Log-target models were not available.")
    else:
        for base_model, group in log_under.groupby("base_model"):
            raw = group[group["method"] == "raw"].iloc[0]
            best = group[group["method"] != "raw"].sort_values(["test_QLIKE", "method"]).iloc[0]
            lines.append(
                f"- {base_model}: raw test pred/actual ratio {float(raw['pred_actual_ratio']):.3f}, "
                f"spike underprediction {float(raw['spike_underprediction_rate']):.3f}; "
                f"best calibrated test QLIKE {float(best['test_QLIKE']):.6f} "
                f"with ratio {float(best['pred_actual_ratio']):.3f}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Calibration is fitted only on validation data and then applied unchanged to test predictions.",
            "- A QLIKE improvement after calibration supports scale bias as one mechanism; persistent spike underprediction after calibration points to shape/timing errors rather than scale alone.",
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_neural_calibration_validation.csv`",
            "- `outputs/advanced/tables/table_neural_calibration_test.csv`",
            "- `outputs/advanced/tables/table_neural_calibration_underprediction.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "neural_calibration_report.md", lines, force=force)


def validation_metrics_by_model(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions[predictions["split"] == "validation"].groupby("model"):
        metrics = compute_metrics(group)
        rows.append({"model": model, **metrics})
    return pd.DataFrame(rows).sort_values(["QLIKE", "RMSE", "model"]).reset_index(drop=True)


def test_metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = compute_metrics(frame)
    return {
        "n_obs": metrics["n_obs"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "QLIKE": metrics["QLIKE"],
        "mean_actual_var": metrics["mean_actual_var"],
        "mean_pred_var": metrics["mean_pred_var"],
    }


def make_combo_frame(
    wide: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": wide["date"],
            "target_date": wide["target_date"],
            "actual_var": wide["actual_var"],
            "pred_var": np.clip(pred, EPSILON, None),
            "model": model_name,
            "split": split,
        }
    )


def pair_combo(
    predictions: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    label: str,
) -> ComboResult:
    val = align_two_models(predictions, model_a=model_a, model_b=model_b, split="validation")
    test = align_two_models(predictions, model_a=model_a, model_b=model_b, split="test")
    weights = np.linspace(0.0, 1.0, 101)
    losses = [
        qlike_objective(val["actual_var"].to_numpy(dtype=float), w * val["pred_a"] + (1.0 - w) * val["pred_b"])
        for w in weights
    ]
    best_w = float(weights[int(np.argmin(losses))])
    model_name = f"ComboPair-{label}"
    val_pred = best_w * val["pred_a"].to_numpy(dtype=float) + (1.0 - best_w) * val["pred_b"].to_numpy(dtype=float)
    test_pred = best_w * test["pred_a"].to_numpy(dtype=float) + (1.0 - best_w) * test["pred_b"].to_numpy(dtype=float)
    return ComboResult(
        model=model_name,
        method="two_model_convex_grid",
        pool=label,
        input_models=[model_a, model_b],
        validation_frame=make_combo_frame(val, model_name=model_name, split="validation", pred=val_pred),
        test_frame=make_combo_frame(test, model_name=model_name, split="test", pred=test_pred),
        weights={model_a: best_w, model_b: 1.0 - best_w},
        metadata={"validation_grid_points": 101, "validation_objective": float(np.min(losses))},
    )


def optimize_convex_weights(actual: np.ndarray, pred_matrix: np.ndarray, *, seed: int = 20260525) -> tuple[np.ndarray, str]:
    n_models = pred_matrix.shape[1]
    if n_models == 1:
        return np.ones(1), "single_model"
    if minimize is not None:
        def objective(weights: np.ndarray) -> float:
            return qlike_objective(actual, pred_matrix @ weights)

        result = minimize(
            objective,
            x0=np.repeat(1.0 / n_models, n_models),
            bounds=[(0.0, 1.0)] * n_models,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and np.isfinite(result.fun):
            weights = np.clip(result.x, 0.0, 1.0)
            total = float(weights.sum())
            if total > 0:
                return weights / total, "scipy_slsqp"
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(n_models), size=5000)
    losses = np.array([qlike_objective(actual, pred_matrix @ weights) for weights in samples])
    return samples[int(np.argmin(losses))], "dirichlet_random_fallback_5000"


def multi_combo(
    predictions: pd.DataFrame,
    *,
    models: list[str],
    pool_name: str,
    method: str,
) -> ComboResult:
    val = align_model_list(predictions, models=models, split="validation")
    test = align_model_list(predictions, models=models, split="test")
    val_matrix = val[models].to_numpy(dtype=float)
    test_matrix = test[models].to_numpy(dtype=float)
    if method == "stacking":
        weights_arr, optimizer = optimize_convex_weights(val["actual_var"].to_numpy(dtype=float), val_matrix)
        weights = {model: float(weight) for model, weight in zip(models, weights_arr)}
        val_pred = val_matrix @ weights_arr
        test_pred = test_matrix @ weights_arr
        model_name = f"ComboStack-{pool_name}"
        metadata = {"optimizer": optimizer}
    elif method == "mean":
        weights = {model: 1.0 / len(models) for model in models}
        val_pred = np.mean(val_matrix, axis=1)
        test_pred = np.mean(test_matrix, axis=1)
        model_name = f"ComboMean-{pool_name}"
        metadata = {"rule": "equal_weight_mean"}
    elif method == "median":
        weights = {model: np.nan for model in models}
        val_pred = np.median(val_matrix, axis=1)
        test_pred = np.median(test_matrix, axis=1)
        model_name = f"ComboMedian-{pool_name}"
        metadata = {"rule": "cross_model_median"}
    else:
        raise ValueError(f"Unknown combination method: {method}")
    return ComboResult(
        model=model_name,
        method=f"multi_model_{method}",
        pool=pool_name,
        input_models=models,
        validation_frame=make_combo_frame(val, model_name=model_name, split="validation", pred=val_pred),
        test_frame=make_combo_frame(test, model_name=model_name, split="test", pred=test_pred),
        weights=weights,
        metadata=metadata,
    )


def choose_best_available(metrics: pd.DataFrame, candidates: list[str]) -> str | None:
    frame = metrics[metrics["model"].isin(candidates)].copy()
    if frame.empty:
        return None
    return str(frame.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]["model"])


def build_combination_specs(predictions: pd.DataFrame) -> tuple[list[tuple[str, str, str]], dict[str, list[str]], dict[str, str | None]]:
    metrics = validation_metrics_by_model(predictions)
    available = set(metrics["model"])
    econ_candidates = [
        model
        for model in available
        if model in {"GARCH(1,1)", "ARIMA-GARCH"}
        or model.startswith("AdvGARCH-")
        or model.startswith("Refit-")
    ]
    best_econ = choose_best_available(metrics, econ_candidates)
    best_garch_family = choose_best_available(metrics, [m for m in available if m.startswith("AdvGARCH-")])
    best_refit = choose_best_available(metrics, [m for m in available if m.startswith("Refit-")])
    calibrated_candidates = [m for m in available if m.startswith("Calibrated-")]
    best_calibrated = choose_best_available(metrics, calibrated_candidates)
    hybrid_calibrated = choose_best_available(metrics, [m for m in calibrated_candidates if "Hybrid-QLIKE" in m])

    pairs = [
        ("GARCH(1,1)", "ARIMA-GARCH", "GARCH11_ARIMAGARCH"),
        ("GARCH(1,1)", "Hybrid-QLIKE", "GARCH11_HybridQLIKE"),
    ]
    if best_garch_family is not None:
        pairs.append((best_garch_family, "Hybrid-QLIKE", "BestGARCHFamily_HybridQLIKE"))
        if best_calibrated is not None:
            pairs.append((best_garch_family, best_calibrated, "BestGARCHFamily_BestCalibratedNeural"))
    if best_refit is not None:
        pairs.append((best_refit, "Hybrid-QLIKE", "BestRefitGARCH_HybridQLIKE"))
        if best_calibrated is not None:
            pairs.append((best_refit, best_calibrated, "BestRefitGARCH_BestCalibratedNeural"))
    pairs = [(a, b, label) for a, b, label in pairs if a in available and b in available]

    pool_a = [m for m in ["GARCH(1,1)", "ARIMA-GARCH", best_garch_family, best_refit] if m and m in available]
    pool_b = [m for m in [best_econ, "LSTM-QLIKE", "Hybrid-QLIKE", "ARIMA-GARCH-LSTM"] if m and m in available]
    pool_c = [m for m in [best_econ, best_calibrated, hybrid_calibrated] if m and m in available]
    best_qlike = float(metrics["QLIKE"].min())
    stable = metrics[metrics["QLIKE"] <= 1.25 * best_qlike]["model"].tolist()
    stable = [
        model
        for model in stable
        if not ("LogTarget" in model and not model.startswith("Calibrated-"))
    ]
    pools = {
        "PoolA_Econometric": sorted(set(pool_a)),
        "PoolB_GARCH_Neural": sorted(set(pool_b)),
        "PoolC_GARCH_CalibratedNeural": sorted(set(pool_c)),
        "PoolD_AllStable": sorted(set(stable)),
    }
    context = {
        "best_econometric": best_econ,
        "best_garch_family": best_garch_family,
        "best_refit": best_refit,
        "best_calibrated_neural": best_calibrated,
        "hybrid_qlike_calibrated": hybrid_calibrated,
    }
    return pairs, pools, context


def save_combo_result(result: ComboResult, *, force: bool) -> None:
    combined = pd.concat([result.validation_frame, result.test_frame], ignore_index=True)
    path = ADVANCED_PREDICTIONS_DIR / "forecast_combinations" / f"pred_{slugify(result.model)}.csv"
    save_prediction_csv(combined, path, force=force)


def run_forecast_combinations(*, force: bool = False) -> pd.DataFrame:
    ensure_advanced_dirs()
    predictions = load_all_predictions(include_advanced=True)
    pairs, pools, context = build_combination_specs(predictions)
    results: list[ComboResult] = []

    for model_a, model_b, label in pairs:
        try:
            results.append(pair_combo(predictions, model_a=model_a, model_b=model_b, label=label))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping pair combination {label}: {type(exc).__name__}: {exc}")

    for pool_name, models in pools.items():
        if len(models) < 2:
            continue
        for method in ["stacking", "mean", "median"]:
            try:
                results.append(multi_combo(predictions, models=models, pool_name=pool_name, method=method))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {method} for {pool_name}: {type(exc).__name__}: {exc}")

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for result in results:
        save_combo_result(result, force=force)
        val_metrics = test_metrics_for_frame(result.validation_frame)
        test_metrics = test_metrics_for_frame(result.test_frame)
        row_base = {
            "model": result.model,
            "method": result.method,
            "pool": result.pool,
            "input_models": json.dumps(result.input_models),
            "metadata": json.dumps(result.metadata, sort_keys=True),
            "selected_by": "validation_QLIKE",
        }
        validation_rows.append({**row_base, **val_metrics})
        test_rows.append({**row_base, **test_metrics})
        for input_model, weight in result.weights.items():
            weight_rows.append(
                {
                    "model": result.model,
                    "method": result.method,
                    "pool": result.pool,
                    "input_model": input_model,
                    "weight": weight,
                }
            )

    validation_df = pd.DataFrame(validation_rows).sort_values(["QLIKE", "RMSE", "model"]) if validation_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows).sort_values(["QLIKE", "RMSE", "model"]) if test_rows else pd.DataFrame()
    weights_df = pd.DataFrame(weight_rows)
    safe_write_csv(validation_df, ADVANCED_TABLES_DIR / "table_forecast_combination_validation.csv", force=force)
    safe_write_csv(test_df, ADVANCED_TABLES_DIR / "table_forecast_combination_test.csv", force=force)
    safe_write_csv(weights_df, ADVANCED_TABLES_DIR / "table_forecast_combination_weights.csv", force=force)
    write_forecast_combination_report(validation_df, test_df, weights_df, context, force=force)
    return validation_df


def write_forecast_combination_report(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    weights: pd.DataFrame,
    context: dict[str, str | None],
    *,
    force: bool,
) -> None:
    lines = [
        "# Forecast Combination Report",
        "",
        "## Validation Context",
        "",
        f"- Best econometric model by validation QLIKE: `{context.get('best_econometric')}`.",
        f"- Best selected GARCH-family model: `{context.get('best_garch_family')}`.",
        f"- Best validation-selected refit model: `{context.get('best_refit')}`.",
        f"- Best calibrated neural model: `{context.get('best_calibrated_neural')}`.",
        "",
        "## Best Validation Combinations",
        "",
    ]
    if validation.empty:
        lines.append("- No forecast combinations were available.")
    else:
        for _, row in validation.head(10).iterrows():
            lines.append(
                f"- {row['model']}: validation QLIKE {float(row['QLIKE']):.6f}, "
                f"method `{row['method']}`, pool `{row['pool']}`."
            )
    lines.extend(["", "## Test Results For Validation-Fitted Combinations", ""])
    if test.empty:
        lines.append("- No test combination results were available.")
    else:
        for _, row in test.head(10).iterrows():
            lines.append(
                f"- {row['model']}: test QLIKE {float(row['QLIKE']):.6f}, "
                f"RMSE {float(row['RMSE']):.6f}, MAE {float(row['MAE']):.6f}."
            )
    lines.extend(
        [
            "",
            "## Weight Fitting",
            "",
            "- Pair weights and stacking weights are fitted on validation QLIKE only.",
            "- Test data are never used to choose pair weights, stacking weights, pools, or calibrated inputs.",
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_forecast_combination_validation.csv`",
            "- `outputs/advanced/tables/table_forecast_combination_test.csv`",
            "- `outputs/advanced/tables/table_forecast_combination_weights.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "forecast_combination_report.md", lines, force=force)


def run_calibration_and_combinations(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = run_neural_calibration(force=force)
    combinations = run_forecast_combinations(force=force)
    return calibration, combinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing advanced outputs.")
    args = parser.parse_args()
    run_calibration_and_combinations(force=args.force)


if __name__ == "__main__":
    main()
