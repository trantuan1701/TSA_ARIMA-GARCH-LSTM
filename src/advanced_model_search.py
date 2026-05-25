#!/usr/bin/env python3
"""Advanced validation-based GARCH-family model search and refit protocols."""

from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("The advanced GARCH search requires the 'arch' package.") from exc

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover
    raise ImportError("The advanced ARIMA-GARCH search requires statsmodels.") from exc

from advanced_experiment_utils import (
    ADVANCED_AUDIT_DIR,
    ADVANCED_PREDICTIONS_DIR,
    ADVANCED_TABLES_DIR,
    EPSILON,
    combine_splits_with_gaps,
    compute_metrics,
    ensure_advanced_dirs,
    load_model_ready,
    load_original_predictions,
    load_splits,
    prediction_frame,
    rel_path,
    safe_write_csv,
    save_failure_rows,
    save_prediction_csv,
    slugify,
    underprediction_stats,
    write_markdown,
)


DISTRIBUTIONS = ["normal", "t", "skewt", "ged"]
MAXITER = 400


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_type: str
    mean_label: str
    arch_mean: str
    arch_lags: int
    arima_p: int | None
    arima_q: int | None
    volatility_label: str
    arch_vol: str
    p: int
    o: int
    q: int
    distribution: str
    volatility_group: str
    is_asymmetric: bool
    is_heavy_tail: bool

    @property
    def uses_arima_mean(self) -> bool:
        return self.candidate_type == "two_step_arima"


def arch_converged(result: Any) -> bool:
    if hasattr(result, "convergence_flag"):
        return int(result.convergence_flag) == 0
    opt = getattr(result, "optimization_result", None)
    if opt is not None and hasattr(opt, "success"):
        return bool(opt.success)
    return False


def warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [f"{item.category.__name__}: {item.message}" for item in caught]


def arima_converged(result: Any) -> bool | object:
    vals = getattr(result, "mle_retvals", None)
    if isinstance(vals, dict) and "converged" in vals:
        return bool(vals["converged"])
    return pd.NA


def build_volatility_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for q in [1, 2, 3, 5]:
        specs.append(
            {
                "volatility_label": f"ARCH({q})",
                "arch_vol": "ARCH",
                "p": q,
                "o": 0,
                "q": 0,
                "volatility_group": "ARCH",
                "is_asymmetric": False,
            }
        )
    for p, q in itertools.product([1, 2, 3], [1, 2, 3]):
        specs.append(
            {
                "volatility_label": f"GARCH({p},{q})",
                "arch_vol": "GARCH",
                "p": p,
                "o": 0,
                "q": q,
                "volatility_group": "GARCH",
                "is_asymmetric": False,
            }
        )
    for p, q in itertools.product([1, 2], [1, 2]):
        specs.append(
            {
                "volatility_label": f"GJR-GARCH({p},1,{q})",
                "arch_vol": "GARCH",
                "p": p,
                "o": 1,
                "q": q,
                "volatility_group": "GJR-GARCH",
                "is_asymmetric": True,
            }
        )
    for p, o, q in itertools.product([1, 2], [0, 1], [1, 2]):
        specs.append(
            {
                "volatility_label": f"EGARCH({p},{o},{q})",
                "arch_vol": "EGARCH",
                "p": p,
                "o": o,
                "q": q,
                "volatility_group": "EGARCH",
                "is_asymmetric": bool(o),
            }
        )
    for p, o, q in itertools.product([1, 2], [0, 1], [1, 2]):
        specs.append(
            {
                "volatility_label": f"APARCH({p},{o},{q})",
                "arch_vol": "APARCH",
                "p": p,
                "o": o,
                "q": q,
                "volatility_group": "APARCH",
                "is_asymmetric": bool(o),
            }
        )
    for p, q in [(1, 1), (1, 0), (0, 1)]:
        specs.append(
            {
                "volatility_label": f"FIGARCH({p},d,{q})",
                "arch_vol": "FIGARCH",
                "p": p,
                "o": 0,
                "q": q,
                "volatility_group": "FIGARCH",
                "is_asymmetric": False,
            }
        )
    return specs


def build_candidate_grid() -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    volatility_specs = build_volatility_specs()
    direct_means = [
        ("Zero", "Zero", 0),
        ("Constant", "Constant", 0),
        *[(f"AR({lag})", "AR", lag) for lag in [1, 2, 3, 5]],
    ]
    for mean_label, arch_mean, arch_lags in direct_means:
        for vol_spec, dist in itertools.product(volatility_specs, DISTRIBUTIONS):
            candidate_id = "_".join(
                [
                    "direct",
                    slugify(mean_label),
                    slugify(vol_spec["volatility_label"]),
                    slugify(dist),
                ]
            )
            candidates.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    candidate_type="direct_arch",
                    mean_label=mean_label,
                    arch_mean=arch_mean,
                    arch_lags=arch_lags,
                    arima_p=None,
                    arima_q=None,
                    distribution=dist,
                    is_heavy_tail=dist != "normal",
                    **vol_spec,
                )
            )

    for p, q in itertools.product(range(4), range(4)):
        for vol_spec, dist in itertools.product(volatility_specs, DISTRIBUTIONS):
            candidate_id = "_".join(
                [
                    "arima",
                    f"{p}_0_{q}",
                    slugify(vol_spec["volatility_label"]),
                    slugify(dist),
                ]
            )
            candidates.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    candidate_type="two_step_arima",
                    mean_label=f"ARIMA({p},0,{q})",
                    arch_mean="Zero",
                    arch_lags=0,
                    arima_p=p,
                    arima_q=q,
                    distribution=dist,
                    is_heavy_tail=dist != "normal",
                    **vol_spec,
                )
            )
    return candidates


def fit_arch_fixed_forecast(
    *,
    spec: CandidateSpec,
    fit_y: pd.Series,
    context_y: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
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
        result = model.fit(disp="off", show_warning=False, options={"maxiter": MAXITER})
    metadata["warnings"] = " | ".join(warning_messages(caught))
    metadata["aic"] = float(getattr(result, "aic", np.nan))
    metadata["bic"] = float(getattr(result, "bic", np.nan))
    metadata["converged"] = arch_converged(result)
    metadata["loglikelihood"] = float(getattr(result, "loglikelihood", np.nan))
    metadata["n_params"] = int(len(getattr(result, "params", [])))
    if not metadata["converged"]:
        raise RuntimeError("convergence failure")

    full_model = arch_model(
        context_y,
        mean=spec.arch_mean,
        lags=spec.arch_lags if spec.arch_mean == "AR" else 0,
        vol=spec.arch_vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.distribution,
        rescale=False,
    )
    fixed = full_model.fix(result.params)
    forecast = fixed.forecast(horizon=1, start=0, reindex=True).variance["h.1"].to_numpy(dtype=float)
    if not np.isfinite(forecast).all():
        raise FloatingPointError("non-finite forecast")
    if (forecast <= 0).any():
        raise FloatingPointError("non-positive forecast")
    return forecast, metadata


def fit_arima_residual_forecast(
    *,
    spec: CandidateSpec,
    fit_y: pd.Series,
    context_y: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.arima_p is None or spec.arima_q is None:
        raise ValueError("ARIMA candidate is missing p/q metadata.")
    order = (int(spec.arima_p), 0, int(spec.arima_q))
    metadata: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arima_result = ARIMA(fit_y.to_numpy(dtype=float), order=order).fit()
    metadata["arima_warnings"] = " | ".join(warning_messages(caught))
    metadata["arima_aic"] = float(getattr(arima_result, "aic", np.nan))
    metadata["arima_bic"] = float(getattr(arima_result, "bic", np.nan))
    metadata["arima_converged"] = arima_converged(arima_result)
    if metadata["arima_converged"] is False:
        raise RuntimeError("ARIMA convergence failure")

    applied = arima_result.apply(context_y.to_numpy(dtype=float), refit=False)
    arima_mean = np.asarray(applied.predict(start=0, end=len(context_y) - 1), dtype=float)
    if not np.isfinite(arima_mean).all():
        raise FloatingPointError("non-finite ARIMA mean forecast")

    context_resid = context_y.to_numpy(dtype=float) - arima_mean
    fit_resid = context_resid[: len(fit_y)]
    if not np.isfinite(fit_resid).all() or not np.isfinite(context_resid).all():
        raise FloatingPointError("non-finite ARIMA residuals")
    zero_mean_spec = CandidateSpec(
        candidate_id=spec.candidate_id,
        candidate_type=spec.candidate_type,
        mean_label=spec.mean_label,
        arch_mean="Zero",
        arch_lags=0,
        arima_p=spec.arima_p,
        arima_q=spec.arima_q,
        volatility_label=spec.volatility_label,
        arch_vol=spec.arch_vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        distribution=spec.distribution,
        volatility_group=spec.volatility_group,
        is_asymmetric=spec.is_asymmetric,
        is_heavy_tail=spec.is_heavy_tail,
    )
    forecast, arch_metadata = fit_arch_fixed_forecast(
        spec=zero_mean_spec,
        fit_y=pd.Series(fit_resid, name="arima_resid"),
        context_y=pd.Series(context_resid, name="arima_resid"),
    )
    metadata.update({f"garch_{key}": value for key, value in arch_metadata.items()})
    return forecast, metadata


def precompute_arima_cache(
    *,
    train: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Fit each ARIMA(p,0,q) mean once for the fixed-parameter search grid."""

    cache: dict[tuple[int, int], dict[str, Any]] = {}
    fit_y = train["log_return_pct"].astype(float).to_numpy()
    context_y = full_df["log_return_pct"].astype(float).to_numpy()
    for p, q in itertools.product(range(4), range(4)):
        order = (p, 0, q)
        key = (p, q)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                arima_result = ARIMA(fit_y, order=order).fit()
            if arima_converged(arima_result) is False:
                raise RuntimeError("ARIMA convergence failure")
            applied = arima_result.apply(context_y, refit=False)
            arima_mean = np.asarray(applied.predict(start=0, end=len(context_y) - 1), dtype=float)
            if not np.isfinite(arima_mean).all():
                raise FloatingPointError("non-finite ARIMA mean forecast")
            context_resid = context_y - arima_mean
            fit_resid = context_resid[: len(fit_y)]
            if not np.isfinite(context_resid).all() or not np.isfinite(fit_resid).all():
                raise FloatingPointError("non-finite ARIMA residuals")
            cache[key] = {
                "status": "success",
                "fit_resid": fit_resid,
                "context_resid": context_resid,
                "metadata": {
                    "arima_aic": float(getattr(arima_result, "aic", np.nan)),
                    "arima_bic": float(getattr(arima_result, "bic", np.nan)),
                    "arima_converged": arima_converged(arima_result),
                    "arima_warnings": " | ".join(warning_messages(caught)),
                },
            }
        except Exception as exc:  # noqa: BLE001
            cache[key] = {
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
                "failure_reason": failure_reason(exc),
            }
    return cache


def forecast_on_context(
    *,
    spec: CandidateSpec,
    fit_df: pd.DataFrame,
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_y = pd.Series(fit_df["log_return_pct"].astype(float).to_numpy(), name="returns")
    context_y = pd.Series(context_df["log_return_pct"].astype(float).to_numpy(), name="returns")
    if spec.uses_arima_mean:
        forecast, metadata = fit_arima_residual_forecast(spec=spec, fit_y=fit_y, context_y=context_y)
    else:
        forecast, metadata = fit_arch_fixed_forecast(spec=spec, fit_y=fit_y, context_y=context_y)

    forecast_df = context_df[["date", "target_date"]].copy()
    forecast_df["pred_var"] = forecast
    aligned = target_df[["date", "target_date"]].merge(
        forecast_df,
        on=["date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if aligned["pred_var"].isna().any():
        missing = aligned.loc[aligned["pred_var"].isna(), ["date", "target_date"]].head(10)
        raise ValueError(f"Forecast alignment failed:\n{missing.to_string(index=False)}")
    pred = aligned["pred_var"].to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise FloatingPointError("non-finite aligned forecast")
    if (pred <= 0).any():
        raise FloatingPointError("non-positive aligned forecast")
    metadata["fit_rows"] = int(len(fit_df))
    metadata["context_rows"] = int(len(context_df))
    return pred, metadata


def failure_reason(exc: BaseException) -> str:
    text = str(exc).lower()
    if "convergence" in text:
        return "convergence failure"
    if "non-positive" in text:
        return "non-positive forecast"
    if "non-finite" in text or "nan" in text or "inf" in text:
        return "numerical error"
    if "not supported" in text or "unsupported" in text:
        return "unsupported specification"
    if "inequality constraints incompatible" in text or "invalid" in text:
        return "invalid parameters"
    return "numerical error"


def candidate_metadata(spec: CandidateSpec) -> dict[str, Any]:
    return asdict(spec)


def evaluate_candidate(
    *,
    spec: CandidateSpec,
    splits: dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    arima_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    target_df = pd.concat([validation, test], ignore_index=True)
    if spec.uses_arima_mean and arima_cache is not None:
        key = (int(spec.arima_p), int(spec.arima_q))  # type: ignore[arg-type]
        cached = arima_cache.get(key)
        if cached is None or cached.get("status") != "success":
            raise RuntimeError(
                cached.get("error", "ARIMA cache unavailable") if cached else "ARIMA cache unavailable"
            )
        zero_spec = CandidateSpec(
            candidate_id=spec.candidate_id,
            candidate_type=spec.candidate_type,
            mean_label=spec.mean_label,
            arch_mean="Zero",
            arch_lags=0,
            arima_p=spec.arima_p,
            arima_q=spec.arima_q,
            volatility_label=spec.volatility_label,
            arch_vol=spec.arch_vol,
            p=spec.p,
            o=spec.o,
            q=spec.q,
            distribution=spec.distribution,
            volatility_group=spec.volatility_group,
            is_asymmetric=spec.is_asymmetric,
            is_heavy_tail=spec.is_heavy_tail,
        )
        forecast_full, arch_metadata = fit_arch_fixed_forecast(
            spec=zero_spec,
            fit_y=pd.Series(cached["fit_resid"], name="arima_resid"),
            context_y=pd.Series(cached["context_resid"], name="arima_resid"),
        )
        forecast_df = full_df[["date", "target_date"]].copy()
        forecast_df["pred_var"] = forecast_full
        aligned = target_df[["date", "target_date"]].merge(
            forecast_df, on=["date", "target_date"], how="left", validate="one_to_one"
        )
        if aligned["pred_var"].isna().any():
            raise ValueError("Cached ARIMA-GARCH forecast alignment failed.")
        forecast = aligned["pred_var"].to_numpy(dtype=float)
        metadata = {**cached["metadata"], **{f"garch_{key}": value for key, value in arch_metadata.items()}}
    else:
        forecast, metadata = forecast_on_context(
            spec=spec,
            fit_df=train,
            context_df=full_df,
            target_df=target_df,
        )
    val_pred = forecast[: len(validation)]
    test_pred = forecast[len(validation) :]
    val_frame = prediction_frame(validation, val_pred, model=spec.candidate_id, split="validation")
    test_frame = prediction_frame(test, test_pred, model=spec.candidate_id, split="test")
    val_metrics = compute_metrics(val_frame)
    test_metrics = compute_metrics(test_frame)
    row = {
        **candidate_metadata(spec),
        "status": "success",
        "forecast_policy": "fixed_train_parameters_recursive_state_updates",
        "validation_n_obs": val_metrics["n_obs"],
        "validation_RMSE": val_metrics["RMSE"],
        "validation_MAE": val_metrics["MAE"],
        "validation_QLIKE": val_metrics["QLIKE"],
        "test_n_obs": test_metrics["n_obs"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_QLIKE": test_metrics["QLIKE"],
        **metadata,
    }
    predictions = pd.concat([val_frame, test_frame], ignore_index=True)
    return row, predictions, metadata


def selected_role_rows(all_candidates: pd.DataFrame) -> pd.DataFrame:
    successful = all_candidates[all_candidates["status"] == "success"].copy()
    if successful.empty:
        return pd.DataFrame()
    successful["validation_QLIKE"] = pd.to_numeric(successful["validation_QLIKE"], errors="coerce")
    successful = successful[np.isfinite(successful["validation_QLIKE"])]

    selections: list[tuple[str, pd.DataFrame]] = [
        ("best_fixed_by_validation_qlike", successful),
        (
            "best_symmetric_garch_by_validation_qlike",
            successful[
                (successful["candidate_type"] == "direct_arch")
                & (~successful["is_asymmetric"].astype(bool))
                & (successful["volatility_group"].isin(["ARCH", "GARCH", "FIGARCH"]))
            ],
        ),
        (
            "best_asymmetric_garch_by_validation_qlike",
            successful[
                (successful["candidate_type"] == "direct_arch")
                & (successful["is_asymmetric"].astype(bool))
            ],
        ),
        (
            "best_heavy_tail_by_validation_qlike",
            successful[successful["is_heavy_tail"].astype(bool)],
        ),
        (
            "best_two_step_arima_garch_by_validation_qlike",
            successful[successful["candidate_type"] == "two_step_arima"],
        ),
    ]
    model_names = {
        "best_fixed_by_validation_qlike": "AdvGARCH-BestQLIKE",
        "best_symmetric_garch_by_validation_qlike": "AdvGARCH-BestSymmetric",
        "best_asymmetric_garch_by_validation_qlike": "AdvGARCH-BestAsymmetric",
        "best_heavy_tail_by_validation_qlike": "AdvGARCH-BestHeavyTail",
        "best_two_step_arima_garch_by_validation_qlike": "AdvGARCH-BestARIMA",
    }
    rows = []
    for role, frame in selections:
        if frame.empty:
            continue
        best = frame.sort_values(["validation_QLIKE", "validation_RMSE", "candidate_id"]).iloc[0].to_dict()
        best["selection_role"] = role
        best["model"] = model_names[role]
        best["selected_by"] = "validation_QLIKE"
        rows.append(best)
    return pd.DataFrame(rows)


def save_selected_predictions(
    *,
    selected: pd.DataFrame,
    predictions_by_candidate: dict[str, pd.DataFrame],
    force: bool,
) -> pd.DataFrame:
    rows = []
    pred_dir = ADVANCED_PREDICTIONS_DIR / "garch_family"
    for _, row in selected.iterrows():
        candidate_id = str(row["candidate_id"])
        model_name = str(row["model"])
        pred = predictions_by_candidate[candidate_id].copy()
        pred["model"] = model_name
        path = pred_dir / f"pred_{slugify(model_name)}.csv"
        save_prediction_csv(pred, path, force=force)
        rows.append({"selection_role": row["selection_role"], "model": model_name, "prediction_file": rel_path(path)})
    return pd.DataFrame(rows)


def add_original_garch_benchmark(test_results: pd.DataFrame) -> pd.DataFrame:
    original = load_original_predictions(["GARCH(1,1)"])
    if original.empty:
        return test_results
    rows = []
    for split in ["validation", "test"]:
        df = original[original["split"] == split].copy()
        metrics = compute_metrics(df)
        rows.append(
            {
                "selection_role": "original_benchmark",
                "model": "GARCH(1,1)",
                "candidate_id": "original_garch_11",
                "candidate_type": "original",
                "mean_label": "Constant",
                "volatility_label": "GARCH(1,1)",
                "distribution": "normal",
                "split": split,
                "n_obs": metrics["n_obs"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "QLIKE": metrics["QLIKE"],
                "selected_by": "original_baseline",
            }
        )
    return pd.concat([test_results, pd.DataFrame(rows)], ignore_index=True)


def selected_results_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        for split in ["validation", "test"]:
            prefix = f"{split}_"
            rows.append(
                {
                    "selection_role": row["selection_role"],
                    "model": row["model"],
                    "candidate_id": row["candidate_id"],
                    "candidate_type": row["candidate_type"],
                    "mean_label": row["mean_label"],
                    "volatility_label": row["volatility_label"],
                    "volatility_group": row["volatility_group"],
                    "distribution": row["distribution"],
                    "split": split,
                    "n_obs": row[f"{prefix}n_obs"],
                    "RMSE": row[f"{prefix}RMSE"],
                    "MAE": row[f"{prefix}MAE"],
                    "QLIKE": row[f"{prefix}QLIKE"],
                    "selected_by": row["selected_by"],
                }
            )
    return add_original_garch_benchmark(pd.DataFrame(rows))


def parse_candidate_from_row(row: pd.Series | dict[str, Any], *, candidate_id: str | None = None) -> CandidateSpec:
    data = dict(row)
    def int_or_none(key: str) -> int | None:
        value = data.get(key)
        if pd.isna(value):
            return None
        return int(value)

    return CandidateSpec(
        candidate_id=candidate_id or str(data["candidate_id"]),
        candidate_type=str(data["candidate_type"]),
        mean_label=str(data["mean_label"]),
        arch_mean=str(data["arch_mean"]),
        arch_lags=int(data.get("arch_lags", 0) if not pd.isna(data.get("arch_lags", 0)) else 0),
        arima_p=int_or_none("arima_p"),
        arima_q=int_or_none("arima_q"),
        volatility_label=str(data["volatility_label"]),
        arch_vol=str(data["arch_vol"]),
        p=int(data["p"]),
        o=int(data["o"]),
        q=int(data["q"]),
        distribution=str(data["distribution"]),
        volatility_group=str(data["volatility_group"]),
        is_asymmetric=bool(data["is_asymmetric"]),
        is_heavy_tail=bool(data["is_heavy_tail"]),
    )


def original_garch_spec() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="original_garch_11_spec",
        candidate_type="direct_arch",
        mean_label="Constant",
        arch_mean="Constant",
        arch_lags=0,
        arima_p=None,
        arima_q=None,
        volatility_label="GARCH(1,1)",
        arch_vol="GARCH",
        p=1,
        o=0,
        q=1,
        distribution="normal",
        volatility_group="GARCH",
        is_asymmetric=False,
        is_heavy_tail=False,
    )


def original_arima_garch_spec() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="original_arima_garch_spec",
        candidate_type="two_step_arima",
        mean_label="ARIMA(0,0,3)",
        arch_mean="Zero",
        arch_lags=0,
        arima_p=0,
        arima_q=3,
        volatility_label="GARCH(1,1)",
        arch_vol="GARCH",
        p=1,
        o=0,
        q=1,
        distribution="normal",
        volatility_group="GARCH",
        is_asymmetric=False,
        is_heavy_tail=False,
    )


def period_starts(df: pd.DataFrame, *, frequency: str) -> list[pd.Timestamp]:
    dates = df["date"].sort_values().reset_index(drop=True)
    if frequency == "monthly":
        groups = dates.groupby([dates.dt.year, dates.dt.month])
    elif frequency == "quarterly":
        groups = dates.groupby([dates.dt.year, dates.dt.quarter])
    else:
        raise ValueError(f"Unknown refit frequency: {frequency}")
    return [pd.Timestamp(values.iloc[0]) for _, values in groups]


def segmented_refit_forecast(
    *,
    spec: CandidateSpec,
    full_df: pd.DataFrame,
    target_df: pd.DataFrame,
    frequency: str,
    window: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    starts = period_starts(target_df, frequency=frequency)
    all_parts = []
    metadata_rows = []
    for index, start_date in enumerate(starts):
        end_date = starts[index + 1] if index + 1 < len(starts) else pd.Timestamp.max
        segment = target_df[(target_df["date"] >= start_date) & (target_df["date"] < end_date)].copy()
        if segment.empty:
            continue
        fit_df = full_df[full_df["date"] < start_date].copy()
        if window is not None:
            fit_df = fit_df.tail(window).copy()
        if len(fit_df) < 50:
            raise ValueError(f"Insufficient fit rows ({len(fit_df)}) before {start_date:%Y-%m-%d}.")
        context_df = pd.concat([fit_df, segment], ignore_index=True)
        pred, metadata = forecast_on_context(
            spec=spec,
            fit_df=fit_df,
            context_df=context_df,
            target_df=segment,
        )
        part = segment[["date", "target_date"]].copy()
        part["pred_var"] = pred
        all_parts.append(part)
        metadata_rows.append(
            {
                "segment_start": start_date.strftime("%Y-%m-%d"),
                "segment_end_exclusive": end_date.strftime("%Y-%m-%d")
                if end_date is not pd.Timestamp.max
                else "end",
                "fit_rows": int(len(fit_df)),
                "segment_rows": int(len(segment)),
                "metadata": json.dumps(metadata, sort_keys=True),
            }
        )
    if not all_parts:
        raise ValueError("No refit forecast segments were produced.")
    pred_df = pd.concat(all_parts, ignore_index=True)
    aligned = target_df[["date", "target_date"]].merge(
        pred_df, on=["date", "target_date"], how="left", validate="one_to_one"
    )
    if aligned["pred_var"].isna().any():
        raise ValueError("Segmented refit forecast did not cover every target row.")
    return aligned["pred_var"].to_numpy(dtype=float), metadata_rows


def fixed_or_window_refit_forecast(
    *,
    spec: CandidateSpec,
    splits: dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    split: str,
    protocol: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_df = splits[split]
    train = splits["train"]
    first_target_date = target_df["date"].min()

    if protocol == "fixed_train_only":
        fit_df = train.copy()
        context_df = full_df.copy()
        pred, metadata = forecast_on_context(spec=spec, fit_df=fit_df, context_df=context_df, target_df=target_df)
        metadata["test_fit_policy"] = "train_only"
        return pred, metadata

    if protocol == "refit_at_validation_start":
        if split == "validation":
            fit_df = train.copy()
            policy = "train_only_before_validation"
        else:
            fit_df = full_df[full_df["date"] < first_target_date].copy()
            policy = "refit_train_val_before_test"
        context_df = full_df[full_df["date"] <= target_df["date"].max()].copy()
        pred, metadata = forecast_on_context(spec=spec, fit_df=fit_df, context_df=context_df, target_df=target_df)
        metadata["test_fit_policy"] = policy
        return pred, metadata

    if protocol in {"expanding_monthly", "expanding_quarterly"}:
        frequency = protocol.replace("expanding_", "")
        pred, segment_meta = segmented_refit_forecast(
            spec=spec,
            full_df=full_df,
            target_df=target_df,
            frequency=frequency,
            window=None,
        )
        return pred, {"segment_count": len(segment_meta), "segment_metadata": segment_meta}

    if protocol.startswith("rolling_"):
        parts = protocol.split("_")
        if len(parts) != 3:
            raise ValueError(f"Bad rolling protocol name: {protocol}")
        window = int(parts[1])
        frequency = parts[2]
        pred, segment_meta = segmented_refit_forecast(
            spec=spec,
            full_df=full_df,
            target_df=target_df,
            frequency=frequency,
            window=window,
        )
        return pred, {"segment_count": len(segment_meta), "segment_metadata": segment_meta}

    raise ValueError(f"Unknown refit protocol: {protocol}")


def run_refit_protocols(
    *,
    selected: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocols = [
        "fixed_train_only",
        "refit_at_validation_start",
        "expanding_monthly",
        "expanding_quarterly",
        "rolling_1000_monthly",
        "rolling_1500_monthly",
        "rolling_2000_monthly",
        "rolling_1000_quarterly",
        "rolling_1500_quarterly",
        "rolling_2000_quarterly",
    ]
    model_specs: list[tuple[str, str, CandidateSpec]] = [
        ("OriginalGARCH", "Original GARCH(1,1)", original_garch_spec()),
        ("OriginalARIMAGARCH", "Original ARIMA-GARCH", original_arima_garch_spec()),
    ]
    if not selected.empty:
        best_fixed = selected[selected["selection_role"] == "best_fixed_by_validation_qlike"]
        if not best_fixed.empty:
            model_specs.append(
                (
                    "BestGARCHFamily",
                    "Best selected GARCH-family",
                    parse_candidate_from_row(best_fixed.iloc[0], candidate_id="best_garch_family_refit_spec"),
                )
            )
        best_asym = selected[selected["selection_role"] == "best_asymmetric_garch_by_validation_qlike"]
        if not best_asym.empty:
            model_specs.append(
                (
                    "BestAsymmetric",
                    "Best asymmetric GARCH-family",
                    parse_candidate_from_row(best_asym.iloc[0], candidate_id="best_asymmetric_refit_spec"),
                )
            )

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    for model_key, model_description, spec in model_specs:
        print(f"Running refit protocols for {model_key}...")
        validation_predictions: dict[str, pd.DataFrame] = {}
        for protocol in protocols:
            row_base = {
                "base_model_key": model_key,
                "base_model_description": model_description,
                "protocol": protocol,
                "candidate_type": spec.candidate_type,
                "mean_label": spec.mean_label,
                "volatility_label": spec.volatility_label,
                "distribution": spec.distribution,
                "selected_by": "validation_QLIKE",
            }
            try:
                pred, metadata = fixed_or_window_refit_forecast(
                    spec=spec,
                    splits=splits,
                    full_df=full_df,
                    split="validation",
                    protocol=protocol,
                )
                frame = prediction_frame(
                    splits["validation"],
                    pred,
                    model=f"Refit-{model_key}-{protocol}",
                    split="validation",
                )
                metrics = compute_metrics(frame)
                validation_predictions[protocol] = frame
                validation_rows.append(
                    {
                        **row_base,
                        "status": "success",
                        "failure_reason": "",
                        "n_obs": metrics["n_obs"],
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "QLIKE": metrics["QLIKE"],
                        "metadata": json.dumps(metadata, sort_keys=True),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - failures are part of the experiment audit.
                validation_rows.append(
                    {
                        **row_base,
                        "status": "failure",
                        "failure_reason": failure_reason(exc),
                        "n_obs": 0,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "QLIKE": np.nan,
                        "metadata": f"{type(exc).__name__}: {exc}",
                    }
                )

        val_model_rows = pd.DataFrame(
            [row for row in validation_rows if row["base_model_key"] == model_key and row["status"] == "success"]
        )
        if val_model_rows.empty:
            continue
        best_protocol = str(val_model_rows.sort_values(["QLIKE", "RMSE", "protocol"]).iloc[0]["protocol"])
        for protocol in protocols:
            selected_by_validation = protocol == best_protocol
            row_base = {
                "base_model_key": model_key,
                "base_model_description": model_description,
                "validation_selected_protocol": best_protocol,
                "protocol": protocol,
                "candidate_type": spec.candidate_type,
                "mean_label": spec.mean_label,
                "volatility_label": spec.volatility_label,
                "distribution": spec.distribution,
                "selected_by_validation": selected_by_validation,
            }
            try:
                pred, metadata = fixed_or_window_refit_forecast(
                    spec=spec,
                    splits=splits,
                    full_df=full_df,
                    split="test",
                    protocol=protocol,
                )
                model_name = f"Refit-{model_key}-{protocol}"
                if protocol == "refit_at_validation_start":
                    model_name = f"Refit-{model_key}-refit_train_val_before_test"
                frame = prediction_frame(splits["test"], pred, model=model_name, split="test")
                metrics = compute_metrics(frame)
                test_rows.append(
                    {
                        **row_base,
                        "test_model": model_name,
                        "status": "success",
                        "failure_reason": "",
                        "n_obs": metrics["n_obs"],
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "QLIKE": metrics["QLIKE"],
                        "metadata": json.dumps(metadata, sort_keys=True),
                    }
                )
                if selected_by_validation and protocol in validation_predictions:
                    val_frame = validation_predictions[protocol].copy()
                    val_frame["model"] = model_name
                    combined = pd.concat([val_frame, frame], ignore_index=True)
                    path = ADVANCED_PREDICTIONS_DIR / "refit_protocols" / f"pred_{slugify(model_name)}.csv"
                    save_prediction_csv(combined, path, force=force)
                    prediction_records.append(
                        {
                            "base_model_key": model_key,
                            "model": model_name,
                            "protocol": protocol,
                            "prediction_file": rel_path(path),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                test_rows.append(
                    {
                        **row_base,
                        "test_model": f"Refit-{model_key}-{protocol}",
                        "status": "failure",
                        "failure_reason": failure_reason(exc),
                        "n_obs": 0,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "QLIKE": np.nan,
                        "metadata": f"{type(exc).__name__}: {exc}",
                    }
                )

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)
    if prediction_records:
        safe_write_csv(
            pd.DataFrame(prediction_records),
            ADVANCED_TABLES_DIR / "table_refit_protocol_selected_prediction_files.csv",
            force=force,
        )
    safe_write_csv(validation_df, ADVANCED_TABLES_DIR / "table_refit_protocol_validation.csv", force=force)
    safe_write_csv(test_df, ADVANCED_TABLES_DIR / "table_refit_protocol_test.csv", force=force)
    write_refit_report(validation_df, test_df, force=force)
    return validation_df, test_df


def write_refit_report(validation_df: pd.DataFrame, test_df: pd.DataFrame, *, force: bool) -> None:
    successful_val = validation_df[validation_df["status"] == "success"].copy()
    successful_test = test_df[test_df["status"] == "success"].copy()
    lines = [
        "# Refit Protocol Report",
        "",
        "## Protocols",
        "",
        "- Fixed train-only: parameters are estimated on the training split and post-training state is updated recursively.",
        "- Refit at validation start: validation uses the training split; the analogous test policy is `refit_train_val_before_test`.",
        "- Expanding monthly/quarterly: refit on all observations available before the first trading day of each period.",
        "- Rolling monthly/quarterly: refit on the latest 1000, 1500, or 2000 observations before each refit date.",
        "",
        "## Validation Selection",
        "",
    ]
    if successful_val.empty:
        lines.append("- No refit protocol completed successfully.")
    else:
        for model_key, group in successful_val.groupby("base_model_key"):
            best = group.sort_values(["QLIKE", "RMSE", "protocol"]).iloc[0]
            fixed = group[group["protocol"] == "fixed_train_only"]
            fixed_qlike = float(fixed.iloc[0]["QLIKE"]) if not fixed.empty else np.nan
            delta = float(best["QLIKE"]) - fixed_qlike if np.isfinite(fixed_qlike) else np.nan
            lines.append(
                f"- {model_key}: best validation protocol `{best['protocol']}` with QLIKE "
                f"{float(best['QLIKE']):.6f}; delta vs fixed train-only = {delta:.6f}."
            )
    lines.extend(["", "## Test Evaluation", ""])
    selected_test = successful_test[successful_test["selected_by_validation"].astype(bool)].copy()
    if selected_test.empty:
        lines.append("- No validation-selected refit protocol produced test forecasts.")
    else:
        for _, row in selected_test.sort_values(["QLIKE", "base_model_key"]).iterrows():
            lines.append(
                f"- {row['base_model_key']}: selected `{row['protocol']}`; test QLIKE "
                f"{float(row['QLIKE']):.6f}, RMSE {float(row['RMSE']):.6f}, MAE {float(row['MAE']):.6f}."
            )
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_refit_protocol_validation.csv`",
            "- `outputs/advanced/tables/table_refit_protocol_test.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "refit_protocol_report.md", lines, force=force)


def write_model_search_report(
    *,
    all_candidates: pd.DataFrame,
    failures: pd.DataFrame,
    selected: pd.DataFrame,
    selected_results: pd.DataFrame,
    force: bool,
) -> None:
    attempted = len(all_candidates) + len(failures)
    successful = int((all_candidates["status"] == "success").sum()) if not all_candidates.empty else 0
    failure_counts = failures["failure_reason"].value_counts().to_dict() if not failures.empty else {}
    failure_lines = [f"- {reason}: {count}" for reason, count in failure_counts.items()] or ["- None."]
    best_lines = []
    for _, row in selected.iterrows():
        best_lines.append(
            "- {role}: `{model}` from `{candidate}` ({mean}, {vol}, {dist}), "
            "validation QLIKE={val:.6f}, test QLIKE={test:.6f}.".format(
                role=row["selection_role"],
                model=row["model"],
                candidate=row["candidate_id"],
                mean=row["mean_label"],
                vol=row["volatility_label"],
                dist=row["distribution"],
                val=float(row["validation_QLIKE"]),
                test=float(row["test_QLIKE"]),
            )
        )

    test = selected_results[selected_results["split"] == "test"].copy()
    garch = test[test["model"] == "GARCH(1,1)"]
    selected_only = test[test["model"] != "GARCH(1,1)"]
    improvement_lines = []
    if not garch.empty and not selected_only.empty:
        garch_qlike = float(garch.iloc[0]["QLIKE"])
        for _, row in selected_only.sort_values("QLIKE").iterrows():
            delta = float(row["QLIKE"]) - garch_qlike
            improvement_lines.append(
                f"- {row['model']}: test QLIKE {float(row['QLIKE']):.6f}; "
                f"delta vs original GARCH(1,1) = {delta:.6f}."
            )
    else:
        improvement_lines.append("- Original GARCH(1,1) benchmark was not available for comparison.")

    lines = [
        "# GARCH-Family Model Search Report",
        "",
        "## Candidate Grid",
        "",
        "- Direct `arch` mean models: Zero, Constant, and AR lags 1, 2, 3, and 5.",
        "- Two-step mean models: ARIMA(p,0,q), p and q in {0,1,2,3}; residuals are passed to GARCH-family volatility models.",
        "- Volatility models: ARCH, GARCH, GJR-GARCH, EGARCH, APARCH, and FIGARCH specifications requested for this stage.",
        "- Distributions: Normal, Student-t, skewed Student-t, and GED.",
        "- Selection uses validation QLIKE only; test QLIKE is reported after selection.",
        "",
        "## Attempt Summary",
        "",
        f"- Attempted candidates: {attempted}.",
        f"- Successful fits: {successful}.",
        f"- Failed fits: {len(failures)}.",
        "",
        "## Failures By Reason",
        "",
        *failure_lines,
        "",
        "## Selected Models",
        "",
        *(best_lines or ["- No model was selected."]),
        "",
        "## Test QLIKE Against Original GARCH(1,1)",
        "",
        *improvement_lines,
        "",
        "## Interpretation",
        "",
        "- Improvements below roughly 0.01 QLIKE should be treated as economically small unless supported by the advanced statistical tests.",
        "- Positive variance clipping is used only inside metric computations and prediction-schema validation; non-positive raw model forecasts are recorded as failures.",
        "- Forecasts are fixed-parameter unless a row is explicitly part of the refit protocol experiment.",
        "",
        "## Output Tables",
        "",
        "- `outputs/advanced/tables/table_garch_family_all_candidates.csv`",
        "- `outputs/advanced/tables/table_garch_family_fit_failures.csv`",
        "- `outputs/advanced/tables/table_garch_family_validation_ranking.csv`",
        "- `outputs/advanced/tables/table_garch_family_selected_models.csv`",
        "- `outputs/advanced/tables/table_garch_family_test_results.csv`",
    ]
    write_markdown(ADVANCED_AUDIT_DIR / "garch_family_model_search_report.md", lines, force=force)


def run_model_search(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_advanced_dirs()
    splits = load_splits()
    full_df = combine_splits_with_gaps(splits, load_model_ready())
    candidates = build_candidate_grid()
    print(f"Attempting {len(candidates)} GARCH-family candidates...")
    arima_cache = precompute_arima_cache(train=splits["train"], full_df=full_df)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    predictions_by_candidate: dict[str, pd.DataFrame] = {}
    started = time.time()

    for index, spec in enumerate(candidates, start=1):
        if index == 1 or index % 100 == 0:
            elapsed = time.time() - started
            print(f"  candidate {index}/{len(candidates)} elapsed={elapsed:.1f}s")
        try:
            row, predictions, _metadata = evaluate_candidate(
                spec=spec,
                splits=splits,
                full_df=full_df,
                arima_cache=arima_cache,
            )
            rows.append(row)
            predictions_by_candidate[spec.candidate_id] = predictions
        except Exception as exc:  # noqa: BLE001 - every failed model is audited and skipped.
            failures.append(
                {
                    **candidate_metadata(spec),
                    "stage": "fit_or_forecast",
                    "failure_reason": failure_reason(exc),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "metadata": json.dumps(candidate_metadata(spec), sort_keys=True),
                }
            )

    all_candidates = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    if all_candidates.empty:
        safe_write_csv(all_candidates, ADVANCED_TABLES_DIR / "table_garch_family_all_candidates.csv", force=force)
        save_failure_rows(failures, ADVANCED_TABLES_DIR / "table_garch_family_fit_failures.csv", force=force)
        raise RuntimeError("All advanced GARCH-family candidates failed.")

    ranking = all_candidates.sort_values(["validation_QLIKE", "validation_RMSE", "candidate_id"]).reset_index(drop=True)
    ranking["validation_rank_QLIKE"] = np.arange(1, len(ranking) + 1)
    selected = selected_role_rows(all_candidates)
    selected_prediction_files = save_selected_predictions(
        selected=selected,
        predictions_by_candidate=predictions_by_candidate,
        force=force,
    )
    if not selected.empty and not selected_prediction_files.empty:
        selected = selected.merge(selected_prediction_files, on=["selection_role", "model"], how="left")
    selected_results = selected_results_table(selected)

    safe_write_csv(all_candidates, ADVANCED_TABLES_DIR / "table_garch_family_all_candidates.csv", force=force)
    save_failure_rows(failures, ADVANCED_TABLES_DIR / "table_garch_family_fit_failures.csv", force=force)
    safe_write_csv(ranking, ADVANCED_TABLES_DIR / "table_garch_family_validation_ranking.csv", force=force)
    safe_write_csv(selected, ADVANCED_TABLES_DIR / "table_garch_family_selected_models.csv", force=force)
    safe_write_csv(selected_results, ADVANCED_TABLES_DIR / "table_garch_family_test_results.csv", force=force)
    write_model_search_report(
        all_candidates=all_candidates,
        failures=failures_df,
        selected=selected,
        selected_results=selected_results,
        force=force,
    )
    run_refit_protocols(selected=selected, splits=splits, full_df=full_df, force=force)
    return all_candidates, failures_df, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing advanced outputs.")
    args = parser.parse_args()
    run_model_search(force=args.force)


if __name__ == "__main__":
    main()
