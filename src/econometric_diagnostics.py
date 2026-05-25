#!/usr/bin/env python3
"""Read-only econometric diagnostics for VN-Index volatility artifacts."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

from empirical_rigor_utils import AUDIT_DIR, PROJECT_ROOT, TABLES_DIR, rel_path, write_markdown


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODEL_READY_PATH = PROCESSED_DIR / "vnindex_model_ready.csv"
LJUNG_BOX_LAGS = [10, 20]
ARCH_LAGS = 10

GARCH_ARTIFACTS = [
    {
        "model": "GARCH(1,1)",
        "artifact": MODELS_DIR / "garch_11.pkl",
        "summary": METRICS_DIR / "garch_summary.txt",
    },
    {
        "model": "ARIMA-GARCH",
        "artifact": MODELS_DIR / "arima_garch_garch.pkl",
        "summary": METRICS_DIR / "arima_garch_summary.txt",
    },
]

STATIONARITY_COLUMNS = [
    "section",
    "test",
    "series",
    "transformation",
    "n_obs",
    "lags",
    "statistic",
    "p_value",
    "used_lag",
    "critical_value_1pct",
    "critical_value_5pct",
    "critical_value_10pct",
]

ARCH_COLUMNS = [
    "section",
    "test",
    "series",
    "n_obs",
    "lags",
    "lm_stat",
    "lm_p_value",
    "f_stat",
    "f_p_value",
]

GARCH_COLUMNS = [
    "model",
    "artifact_file",
    "summary_file",
    "availability",
    "n_obs",
    "loglikelihood",
    "aic",
    "bic",
    "convergence_status",
    "convergence_flag",
    "mu",
    "mu_p_value",
    "omega",
    "omega_p_value",
    "alpha_1",
    "alpha_1_p_value",
    "beta_1",
    "beta_1_p_value",
    "alpha_beta_persistence",
    "reason",
]


def ensure_dirs() -> None:
    for path in (TABLES_DIR, AUDIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_model_ready() -> pd.DataFrame:
    if not MODEL_READY_PATH.exists():
        raise FileNotFoundError(f"Required data file is missing: {rel_path(MODEL_READY_PATH)}")
    df = pd.read_csv(MODEL_READY_PATH, encoding="utf-8-sig")
    required = {"date", "log_return_pct", "squared_return"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rel_path(MODEL_READY_PATH)} is missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"{rel_path(MODEL_READY_PATH)} contains unparseable date values.")
    for column in ("log_return_pct", "squared_return"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def finite_series(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series.to_numpy(dtype=float))]
    return series.astype(float).reset_index(drop=True)


def effective_lags(n_obs: int, requested_lags: list[int]) -> list[int]:
    return [lag for lag in requested_lags if 0 < lag < n_obs]


def append_adf_row(rows: list[dict[str, Any]], series: pd.Series, series_name: str) -> None:
    result = adfuller(series.to_numpy(dtype=float), autolag="AIC")
    critical_values = result[4]
    rows.append(
        {
            "section": "stationarity_dependence",
            "test": "ADF",
            "series": series_name,
            "transformation": "level",
            "n_obs": int(result[3]),
            "lags": "auto_aic",
            "statistic": float(result[0]),
            "p_value": float(result[1]),
            "used_lag": int(result[2]),
            "critical_value_1pct": float(critical_values["1%"]),
            "critical_value_5pct": float(critical_values["5%"]),
            "critical_value_10pct": float(critical_values["10%"]),
        }
    )


def append_ljung_box_rows(
    rows: list[dict[str, Any]],
    series: pd.Series,
    *,
    section: str,
    series_name: str,
    transformation: str,
) -> None:
    lags = effective_lags(len(series), LJUNG_BOX_LAGS)
    if not lags:
        rows.append(
            {
                "section": section,
                "test": "Ljung-Box",
                "series": series_name,
                "transformation": transformation,
                "n_obs": len(series),
                "lags": "not_available",
                "statistic": "not_available",
                "p_value": "not_available",
                "used_lag": "not_available",
                "critical_value_1pct": "not_available",
                "critical_value_5pct": "not_available",
                "critical_value_10pct": "not_available",
            }
        )
        return

    lb = acorr_ljungbox(series.to_numpy(dtype=float), lags=lags, return_df=True)
    for lag, row in lb.iterrows():
        rows.append(
            {
                "section": section,
                "test": "Ljung-Box",
                "series": series_name,
                "transformation": transformation,
                "n_obs": int(len(series)),
                "lags": int(lag),
                "statistic": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
                "used_lag": int(lag),
                "critical_value_1pct": "not_available",
                "critical_value_5pct": "not_available",
                "critical_value_10pct": "not_available",
            }
        )


def append_arch_lm_row(
    rows: list[dict[str, Any]],
    series: pd.Series,
    *,
    section: str,
    series_name: str,
) -> None:
    max_lag = max(1, min(ARCH_LAGS, len(series) - 2))
    demeaned = series.to_numpy(dtype=float) - float(series.mean())
    lm_stat, lm_p_value, f_stat, f_p_value = het_arch(demeaned, nlags=max_lag, ddof=0)
    rows.append(
        {
            "section": section,
            "test": "ARCH-LM",
            "series": series_name,
            "n_obs": int(len(series)),
            "lags": int(max_lag),
            "lm_stat": float(lm_stat),
            "lm_p_value": float(lm_p_value),
            "f_stat": float(f_stat),
            "f_p_value": float(f_p_value),
        }
    )


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def convergence_status(result: Any) -> tuple[str, str]:
    if hasattr(result, "convergence_flag"):
        flag = int(result.convergence_flag)
        return ("converged" if flag == 0 else "not_converged", str(flag))
    optimization_result = getattr(result, "optimization_result", None)
    if optimization_result is not None and hasattr(optimization_result, "success"):
        return ("converged" if bool(optimization_result.success) else "not_converged", "not_available")
    return "not_available", "not_available"


def param_column_name(param_name: str) -> str:
    normalized = param_name.strip().lower()
    normalized = normalized.replace("[", "_").replace("]", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized


def add_param_columns(row: dict[str, Any], result: Any) -> None:
    params = getattr(result, "params", None)
    pvalues = getattr(result, "pvalues", None)
    if params is None:
        return

    for raw_name, value in params.items():
        name = param_column_name(str(raw_name))
        if name in {"const", "constant"}:
            name = "mu"
        if name in {"alpha1", "alpha_1"}:
            name = "alpha_1"
        if name in {"beta1", "beta_1"}:
            name = "beta_1"
        if name in row:
            row[name] = float(value)
            if pvalues is not None and raw_name in pvalues.index:
                row[f"{name}_p_value"] = float(pvalues[raw_name])

    alpha_values = [
        float(value)
        for raw_name, value in params.items()
        if param_column_name(str(raw_name)).startswith("alpha")
    ]
    beta_values = [
        float(value)
        for raw_name, value in params.items()
        if param_column_name(str(raw_name)).startswith("beta")
    ]
    if alpha_values or beta_values:
        row["alpha_beta_persistence"] = float(sum(alpha_values) + sum(beta_values))


def garch_diagnostic_row(model: str, artifact_path: Path, summary_path: Path, result: Any | None, reason: str) -> dict[str, Any]:
    row = {column: "not_available" for column in GARCH_COLUMNS}
    row.update(
        {
            "model": model,
            "artifact_file": rel_path(artifact_path),
            "summary_file": rel_path(summary_path) if summary_path.exists() else "not_available",
            "availability": "available" if result is not None else "not_available",
            "reason": reason,
        }
    )
    if result is None:
        return row

    status, flag = convergence_status(result)
    row.update(
        {
            "n_obs": int(getattr(result, "nobs", "not_available")),
            "loglikelihood": float(getattr(result, "loglikelihood", np.nan)),
            "aic": float(getattr(result, "aic", np.nan)),
            "bic": float(getattr(result, "bic", np.nan)),
            "convergence_status": status,
            "convergence_flag": flag,
        }
    )
    add_param_columns(row, result)
    return row


def get_standardized_residuals(result: Any) -> tuple[pd.Series | None, str]:
    if hasattr(result, "std_resid"):
        residuals = finite_series(getattr(result, "std_resid"))
        if not residuals.empty:
            return residuals, "std_resid attribute"

    if hasattr(result, "resid") and hasattr(result, "conditional_volatility"):
        resid = pd.to_numeric(pd.Series(getattr(result, "resid")), errors="coerce").to_numpy(dtype=float)
        cond_vol = pd.to_numeric(
            pd.Series(getattr(result, "conditional_volatility")), errors="coerce"
        ).to_numpy(dtype=float)
        mask = np.isfinite(resid) & np.isfinite(cond_vol) & (cond_vol > 0)
        if mask.any():
            return pd.Series(resid[mask] / cond_vol[mask]), "resid / conditional_volatility"

    return None, "not_available"


def dataframe_with_not_available(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    return df.replace([np.inf, -np.inf], np.nan).fillna("not_available")


def build_report(
    *,
    stationarity_rows: list[dict[str, Any]],
    arch_rows: list[dict[str, Any]],
    garch_rows: list[dict[str, Any]],
    input_files: list[Path],
    unavailable: list[str],
    residual_sources: list[str],
) -> list[str]:
    adf_row = next(row for row in stationarity_rows if row["test"] == "ADF")
    garch_summary = [
        f"- {row['model']}: persistence={row['alpha_beta_persistence']}, "
        f"convergence={row['convergence_status']}"
        for row in garch_rows
    ]
    unavailable_lines = [f"- {item}" for item in unavailable] if unavailable else ["None."]

    lines = [
        "# Econometric Diagnostics Report",
        "",
        "## Computed",
        "",
        "- ADF stationarity test on full-sample percentage log returns.",
        "- Ljung-Box tests on percentage log returns and squared returns.",
        "- ARCH-LM test on demeaned percentage log returns.",
        "- GARCH parameter, p-value, persistence, and convergence diagnostics from existing pickles.",
        "- Residual Ljung-Box and ARCH-LM diagnostics from stored standardized residuals where available.",
        "",
        "## Key Results",
        "",
        f"- ADF on returns: statistic={adf_row['statistic']:.6f}, p-value={adf_row['p_value']:.6g}.",
        *garch_summary,
        "",
        "## Input Files Used",
        "",
        *[f"- `{rel_path(path)}`" for path in sorted(set(input_files))],
        "",
        "## Assumptions",
        "",
        "- Stationarity and dependence tests use `data/processed/vnindex_model_ready.csv`.",
        f"- Ljung-Box tests are reported at lags {LJUNG_BOX_LAGS}.",
        f"- ARCH-LM tests use {ARCH_LAGS} lags when enough observations are available.",
        "- ARCH-LM inputs are demeaned before testing for conditional heteroskedasticity.",
        "- GARCH residual diagnostics use training-sample standardized residuals stored in fitted artifacts.",
        "- The ARIMA-GARCH GARCH leg is a zero-mean residual model, so `mu` and `mu_p_value` "
        "are marked `not_available` for that row.",
        "- Ljung-Box rows report p-values directly; ADF-only critical-value columns are marked "
        "`not_available` for Ljung-Box rows.",
        "",
        "## Residual Sources",
        "",
        *(residual_sources or ["- not_available"]),
        "",
        "## Unavailable Diagnostics",
        "",
        *unavailable_lines,
        "",
        "## Output Tables",
        "",
        "- `outputs/tables/table_stationarity_tests.csv`",
        "- `outputs/tables/table_arch_lm_tests.csv`",
        "- `outputs/tables/table_garch_diagnostics.csv`",
    ]
    return lines


def main() -> None:
    ensure_dirs()
    model_ready = load_model_ready()
    returns = finite_series(model_ready["log_return_pct"])
    squared_returns = finite_series(model_ready["squared_return"])

    stationarity_rows: list[dict[str, Any]] = []
    arch_rows: list[dict[str, Any]] = []
    garch_rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    residual_sources: list[str] = []
    input_files: list[Path] = [MODEL_READY_PATH]

    append_adf_row(stationarity_rows, returns, "log_return_pct")
    append_ljung_box_rows(
        stationarity_rows,
        returns,
        section="stationarity_dependence",
        series_name="log_return_pct",
        transformation="level",
    )
    append_ljung_box_rows(
        stationarity_rows,
        squared_returns,
        section="stationarity_dependence",
        series_name="squared_return",
        transformation="level",
    )
    append_arch_lm_row(
        arch_rows,
        returns,
        section="stationarity_dependence",
        series_name="log_return_pct",
    )

    for spec in GARCH_ARTIFACTS:
        model = spec["model"]
        artifact_path = spec["artifact"]
        summary_path = spec["summary"]
        if summary_path.exists():
            input_files.append(summary_path)

        result = None
        reason = ""
        if artifact_path.exists():
            input_files.append(artifact_path)
            try:
                result = load_pickle(artifact_path)
            except Exception as exc:  # pragma: no cover - defensive artifact handling.
                reason = f"Could not load pickle: {type(exc).__name__}: {exc}"
                unavailable.append(f"{model} fitted artifact could not be loaded from `{rel_path(artifact_path)}`.")
        else:
            reason = f"Missing fitted artifact: {rel_path(artifact_path)}"
            unavailable.append(f"{model} fitted artifact is missing, so GARCH diagnostics are not available.")

        garch_rows.append(garch_diagnostic_row(model, artifact_path, summary_path, result, reason))
        if result is None:
            unavailable.append(f"{model} standardized residual diagnostics are not available.")
            continue

        residuals, residual_source = get_standardized_residuals(result)
        if residuals is None:
            unavailable.append(f"{model} standardized residuals could not be loaded or reconstructed.")
            continue

        residual_sources.append(f"- {model}: {residual_source}, n={len(residuals)}.")
        append_ljung_box_rows(
            stationarity_rows,
            residuals,
            section="residual_diagnostics",
            series_name=f"{model} standardized residuals",
            transformation="level",
        )
        append_ljung_box_rows(
            stationarity_rows,
            residuals**2,
            section="residual_diagnostics",
            series_name=f"{model} standardized residuals",
            transformation="squared",
        )
        append_arch_lm_row(
            arch_rows,
            residuals,
            section="residual_diagnostics",
            series_name=f"{model} standardized residuals",
        )

    stationarity_path = TABLES_DIR / "table_stationarity_tests.csv"
    arch_path = TABLES_DIR / "table_arch_lm_tests.csv"
    garch_path = TABLES_DIR / "table_garch_diagnostics.csv"
    report_path = AUDIT_DIR / "econometric_diagnostics_report.md"

    dataframe_with_not_available(stationarity_rows, STATIONARITY_COLUMNS).to_csv(
        stationarity_path, index=False
    )
    dataframe_with_not_available(arch_rows, ARCH_COLUMNS).to_csv(arch_path, index=False)
    dataframe_with_not_available(garch_rows, GARCH_COLUMNS).to_csv(garch_path, index=False)
    write_markdown(
        report_path,
        build_report(
            stationarity_rows=stationarity_rows,
            arch_rows=arch_rows,
            garch_rows=garch_rows,
            input_files=input_files,
            unavailable=unavailable,
            residual_sources=residual_sources,
        ),
    )

    print(f"Wrote {rel_path(stationarity_path)}")
    print(f"Wrote {rel_path(arch_path)}")
    print(f"Wrote {rel_path(garch_path)}")
    print(f"Wrote {rel_path(report_path)}")


if __name__ == "__main__":
    main()
