#!/usr/bin/env python3
"""Advanced statistical comparison, MCS approximation, and encompassing tests."""

from __future__ import annotations

import argparse
import json
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

from advanced_experiment_utils import (
    ADVANCED_AUDIT_DIR,
    ADVANCED_TABLES_DIR,
    LOSS_TYPES,
    align_two_models,
    common_window,
    compute_metrics,
    ensure_advanced_dirs,
    holm_adjust,
    load_all_predictions,
    loss_values,
    safe_write_csv,
    write_markdown,
)


DM_COLUMNS = [
    "model",
    "benchmark",
    "loss_type",
    "n",
    "mean_loss_model",
    "mean_loss_benchmark",
    "mean_diff",
    "dm_stat",
    "p_value",
    "winner",
    "hac_lag",
]


def normal_p_value(statistic: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(statistic))))


def newey_west_long_run_variance(diff: np.ndarray) -> tuple[float, int]:
    diff = np.asarray(diff, dtype=float)
    n_obs = len(diff)
    if n_obs <= 1:
        return 0.0, 0
    centered = diff - float(np.mean(diff))
    max_lag = min(int(np.floor(n_obs ** (1.0 / 3.0))), n_obs - 1)
    gamma_0 = float(np.dot(centered, centered) / n_obs)
    lrv = gamma_0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * cov
    if lrv < 0 and abs(lrv) < 1e-14:
        lrv = 0.0
    return float(lrv), int(max_lag)


def dm_statistic(diff: np.ndarray) -> tuple[float, float, int]:
    diff = np.asarray(diff, dtype=float)
    mean_diff = float(np.mean(diff))
    lrv, lag = newey_west_long_run_variance(diff)
    if lrv <= 0:
        if abs(mean_diff) <= 1e-14:
            return 0.0, 1.0, lag
        return np.nan, np.nan, lag
    stat = mean_diff / np.sqrt(lrv / len(diff))
    return float(stat), normal_p_value(float(stat)), lag


def winner_label(model: str, benchmark: str, mean_diff: float, p_value: float) -> str:
    if not np.isfinite(p_value) or p_value >= 0.10:
        return "No significant difference"
    return model if mean_diff < 0 else benchmark


def compare_to_benchmark(
    common_long: pd.DataFrame,
    *,
    model: str,
    benchmark: str,
    loss_type: str,
) -> dict[str, Any]:
    model_df = common_long.loc[
        common_long["model"] == model, ["date", "target_date", "actual_var", "pred_var"]
    ].rename(columns={"pred_var": "pred_model"})
    benchmark_df = common_long.loc[
        common_long["model"] == benchmark, ["date", "target_date", "pred_var"]
    ].rename(columns={"pred_var": "pred_benchmark"})
    merged = model_df.merge(benchmark_df, on=["date", "target_date"], how="inner", validate="one_to_one")
    loss_model = loss_values(merged["actual_var"], merged["pred_model"], loss_type)
    loss_benchmark = loss_values(merged["actual_var"], merged["pred_benchmark"], loss_type)
    diff = loss_model - loss_benchmark
    stat, p_value, lag = dm_statistic(diff)
    mean_diff = float(np.mean(diff))
    return {
        "model": model,
        "benchmark": benchmark,
        "loss_type": loss_type,
        "n": int(len(diff)),
        "mean_loss_model": float(np.mean(loss_model)),
        "mean_loss_benchmark": float(np.mean(loss_benchmark)),
        "mean_diff": mean_diff,
        "dm_stat": stat,
        "p_value": p_value,
        "winner": winner_label(model, benchmark, mean_diff, p_value),
        "hac_lag": lag,
    }


def wilcoxon_compare(
    common_long: pd.DataFrame,
    *,
    model: str,
    benchmark: str,
    loss_type: str,
) -> dict[str, Any]:
    model_df = common_long.loc[
        common_long["model"] == model, ["date", "target_date", "actual_var", "pred_var"]
    ].rename(columns={"pred_var": "pred_model"})
    benchmark_df = common_long.loc[
        common_long["model"] == benchmark, ["date", "target_date", "pred_var"]
    ].rename(columns={"pred_var": "pred_benchmark"})
    merged = model_df.merge(benchmark_df, on=["date", "target_date"], how="inner", validate="one_to_one")
    diff = loss_values(merged["actual_var"], merged["pred_model"], loss_type) - loss_values(
        merged["actual_var"], merged["pred_benchmark"], loss_type
    )
    if wilcoxon is None:
        stat, p_value, status = np.nan, np.nan, "scipy_unavailable"
    elif np.allclose(diff, 0.0):
        stat, p_value, status = 0.0, 1.0, "all_differences_zero"
    else:
        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
        stat, p_value, status = float(result.statistic), float(result.pvalue), "success"
    return {
        "model": model,
        "benchmark": benchmark,
        "loss_type": loss_type,
        "n": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "status": status,
    }


def validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions[predictions["split"] == "validation"].groupby("model"):
        rows.append({"model": model, **compute_metrics(group)})
    return pd.DataFrame(rows).sort_values(["QLIKE", "RMSE", "model"]).reset_index(drop=True)


def selected_benchmarks(predictions: pd.DataFrame) -> list[str]:
    benchmarks = ["GARCH(1,1)"]
    selected_path = ADVANCED_TABLES_DIR / "table_garch_family_selected_models.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        row = selected[selected["selection_role"] == "best_fixed_by_validation_qlike"]
        if not row.empty:
            benchmarks.append(str(row.iloc[0]["model"]))
    metrics = validation_metrics(predictions)
    if not metrics.empty:
        benchmarks.append(str(metrics.iloc[0]["model"]))
    seen = set()
    ordered = []
    available = set(predictions["model"].unique())
    for model in benchmarks:
        if model in available and model not in seen:
            ordered.append(model)
            seen.add(model)
    return ordered


def run_dm_and_wilcoxon(*, force: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    ensure_advanced_dirs()
    predictions = load_all_predictions(include_advanced=True)
    common = common_window(predictions, split="test")
    benchmarks = selected_benchmarks(predictions)
    dm_tables: dict[str, pd.DataFrame] = {}
    wilcoxon_rows: list[dict[str, Any]] = []

    for loss_type in LOSS_TYPES:
        rows = []
        for benchmark in benchmarks:
            for model in common.models:
                if model == benchmark:
                    continue
                rows.append(compare_to_benchmark(common.long, model=model, benchmark=benchmark, loss_type=loss_type))
                wilcoxon_rows.append(
                    wilcoxon_compare(common.long, model=model, benchmark=benchmark, loss_type=loss_type)
                )
        table = pd.DataFrame(rows, columns=DM_COLUMNS)
        dm_tables[loss_type] = table
        suffix = {
            "qlike": "qlike",
            "squared_error": "squared_error",
            "absolute_error": "absolute_error",
        }[loss_type]
        safe_write_csv(table, ADVANCED_TABLES_DIR / f"table_advanced_dm_tests_{suffix}.csv", force=force)

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    safe_write_csv(wilcoxon_df, ADVANCED_TABLES_DIR / "table_advanced_wilcoxon_tests.csv", force=force)
    holm_df = build_holm_table(dm_tables, wilcoxon_df)
    safe_write_csv(holm_df, ADVANCED_TABLES_DIR / "table_advanced_holm_corrected_tests.csv", force=force)
    mcs_df = model_confidence_set_table(common.wide, force=force)
    write_hypothesis_report(dm_tables, wilcoxon_df, holm_df, mcs_df, benchmarks, force=force)
    return dm_tables, holm_df, mcs_df


def build_holm_table(dm_tables: dict[str, pd.DataFrame], wilcoxon_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loss_type, table in dm_tables.items():
        for _, row in table.iterrows():
            rows.append(
                {
                    "test_type": "DM",
                    "loss_type": loss_type,
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "raw_p_value": row["p_value"],
                    "mean_diff": row["mean_diff"],
                }
            )
    for _, row in wilcoxon_df.iterrows():
        rows.append(
            {
                "test_type": "Wilcoxon",
                "loss_type": row["loss_type"],
                "model": row["model"],
                "benchmark": row["benchmark"],
                "raw_p_value": row["p_value"],
                "mean_diff": row["mean_diff"],
            }
        )
    holm = pd.DataFrame(rows)
    if holm.empty:
        return holm
    holm["holm_adjusted_p_value"] = np.nan
    for (test_type, loss_type), index in holm.groupby(["test_type", "loss_type"]).groups.items():
        adjusted = holm_adjust(holm.loc[index, "raw_p_value"].to_numpy(dtype=float))
        holm.loc[index, "holm_adjusted_p_value"] = adjusted
        holm.loc[index, "holm_group"] = f"{test_type}_{loss_type}"
    holm["significant_10pct_holm"] = holm["holm_adjusted_p_value"] < 0.10
    holm["significant_5pct_holm"] = holm["holm_adjusted_p_value"] < 0.05
    return holm.sort_values(["loss_type", "test_type", "holm_adjusted_p_value", "model"]).reset_index(drop=True)


def moving_block_bootstrap_indices(n_obs: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(1, n_obs - block_length + 1), size=int(np.ceil(n_obs / block_length)))
    indices = np.concatenate([np.arange(start, min(start + block_length, n_obs)) for start in starts])
    if len(indices) < n_obs:
        extra = rng.integers(0, n_obs, size=n_obs - len(indices))
        indices = np.concatenate([indices, extra])
    return indices[:n_obs]


def mcs_elimination(
    loss_matrix: pd.DataFrame,
    *,
    alpha: float,
    bootstrap_replications: int = 300,
    seed: int = 20260525,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + int(alpha * 1000))
    active = list(loss_matrix.columns)
    n_obs = len(loss_matrix)
    block_length = max(5, int(round(np.sqrt(n_obs))))
    elimination: list[dict[str, Any]] = []
    retained: list[str] = active.copy()
    step = 0
    while len(active) > 1:
        step += 1
        sub = loss_matrix[active].to_numpy(dtype=float)
        means = sub.mean(axis=0)
        best_idx = int(np.argmin(means))
        worst_idx = int(np.argmax(means))
        scale = float(np.nanmean(np.std(sub, axis=0, ddof=1)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        obs_stat = float(np.sqrt(n_obs) * (means[worst_idx] - means[best_idx]) / scale)
        centered = sub - means
        boot_stats = []
        for _ in range(bootstrap_replications):
            idx = moving_block_bootstrap_indices(n_obs, block_length=block_length, rng=rng)
            boot_means = centered[idx].mean(axis=0)
            boot_stats.append(float(np.sqrt(n_obs) * (boot_means.max() - boot_means.min()) / scale))
        p_value = float(np.mean(np.asarray(boot_stats) >= obs_stat))
        worst_model = active[worst_idx]
        if p_value < alpha:
            elimination.append(
                {
                    "model": worst_model,
                    "alpha": alpha,
                    "elimination_step": step,
                    "mcs_p_value": p_value,
                    "mean_qlike_at_elimination": float(means[worst_idx]),
                    "status": "eliminated",
                }
            )
            active.pop(worst_idx)
            retained = active.copy()
        else:
            break
    for model in retained:
        elimination.append(
            {
                "model": model,
                "alpha": alpha,
                "elimination_step": pd.NA,
                "mcs_p_value": pd.NA,
                "mean_qlike_at_elimination": float(loss_matrix[model].mean()),
                "status": "retained",
            }
        )
    return {
        "rows": elimination,
        "retained": retained,
        "block_length": block_length,
        "bootstrap_replications": bootstrap_replications,
        "method": "documented_approximation_block_bootstrap_range_statistic",
    }


def model_confidence_set_table(wide: pd.DataFrame, *, force: bool) -> pd.DataFrame:
    models = [column for column in wide.columns if column not in {"date", "target_date", "actual_var"}]
    losses = {
        model: loss_values(wide["actual_var"], wide[model], "qlike")
        for model in models
    }
    loss_matrix = pd.DataFrame(losses)
    mean_losses = loss_matrix.mean().sort_values()
    results_90 = mcs_elimination(loss_matrix, alpha=0.10)
    results_95 = mcs_elimination(loss_matrix, alpha=0.05)
    retained_90 = set(results_90["retained"])
    retained_95 = set(results_95["retained"])
    rows = []
    elimination_90 = pd.DataFrame(results_90["rows"])
    elimination_95 = pd.DataFrame(results_95["rows"])
    for rank, (model, mean_loss) in enumerate(mean_losses.items(), start=1):
        row90 = elimination_90[elimination_90["model"] == model].head(1)
        row95 = elimination_95[elimination_95["model"] == model].head(1)
        rows.append(
            {
                "model": model,
                "mean_qlike_loss": float(mean_loss),
                "mean_loss_rank": rank,
                "included_in_MCS_90": model in retained_90,
                "included_in_MCS_95": model in retained_95,
                "elimination_step_90": row90.iloc[0]["elimination_step"] if not row90.empty else pd.NA,
                "mcs_p_value_90": row90.iloc[0]["mcs_p_value"] if not row90.empty else pd.NA,
                "elimination_step_95": row95.iloc[0]["elimination_step"] if not row95.empty else pd.NA,
                "mcs_p_value_95": row95.iloc[0]["mcs_p_value"] if not row95.empty else pd.NA,
                "method": results_90["method"],
                "bootstrap_replications": results_90["bootstrap_replications"],
                "block_length": results_90["block_length"],
            }
        )
    table = pd.DataFrame(rows)
    safe_write_csv(table, ADVANCED_TABLES_DIR / "table_model_confidence_set_qlike.csv", force=force)
    return table


def write_hypothesis_report(
    dm_tables: dict[str, pd.DataFrame],
    wilcoxon_df: pd.DataFrame,
    holm_df: pd.DataFrame,
    mcs_df: pd.DataFrame,
    benchmarks: list[str],
    *,
    force: bool,
) -> None:
    qlike = dm_tables.get("qlike", pd.DataFrame()).copy()
    significant = qlike[(qlike["benchmark"] == "GARCH(1,1)") & (qlike["p_value"] < 0.10) & (qlike["mean_diff"] < 0)]
    holm_sig = holm_df[
        (holm_df["test_type"] == "DM")
        & (holm_df["loss_type"] == "qlike")
        & (holm_df["benchmark"] == "GARCH(1,1)")
        & (holm_df["holm_adjusted_p_value"] < 0.10)
        & (holm_df["mean_diff"] < 0)
    ]
    lines = [
        "# Advanced Hypothesis Tests Report",
        "",
        "## Benchmarks",
        "",
        *[f"- `{benchmark}`" for benchmark in benchmarks],
        "",
        "## Loss Difference Convention",
        "",
        "- `d_t = loss_model_t - loss_benchmark_t`.",
        "- Positive mean differences mean the model is worse than the benchmark.",
        "- Negative mean differences mean the model is better than the benchmark.",
        "",
        "## QLIKE DM Results Against Original GARCH(1,1)",
        "",
    ]
    if significant.empty:
        lines.append("- No model significantly beats original GARCH(1,1) by QLIKE DM at the 10% level.")
    else:
        for _, row in significant.sort_values("p_value").iterrows():
            lines.append(
                f"- {row['model']}: mean_diff {float(row['mean_diff']):.6f}, p={float(row['p_value']):.6g}."
            )
    lines.extend(["", "## Holm Correction", ""])
    if holm_sig.empty:
        lines.append("- No QLIKE DM improvement over GARCH(1,1) survives Holm correction at the 10% level.")
    else:
        for _, row in holm_sig.sort_values("holm_adjusted_p_value").iterrows():
            lines.append(
                f"- {row['model']}: Holm-adjusted p={float(row['holm_adjusted_p_value']):.6g}."
            )
    lines.extend(["", "## Model Confidence Set", ""])
    if mcs_df.empty:
        lines.append("- MCS was not available.")
    else:
        retained_90 = mcs_df[mcs_df["included_in_MCS_90"]]["model"].tolist()
        retained_95 = mcs_df[mcs_df["included_in_MCS_95"]]["model"].tolist()
        lines.append("- MCS method: documented approximation using a moving-block bootstrap range statistic.")
        lines.append(f"- Models retained in 90% MCS: {', '.join(retained_90) if retained_90 else 'None'}.")
        lines.append(f"- Models retained in 95% MCS: {', '.join(retained_95) if retained_95 else 'None'}.")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_advanced_dm_tests_qlike.csv`",
            "- `outputs/advanced/tables/table_advanced_dm_tests_squared_error.csv`",
            "- `outputs/advanced/tables/table_advanced_dm_tests_absolute_error.csv`",
            "- `outputs/advanced/tables/table_advanced_wilcoxon_tests.csv`",
            "- `outputs/advanced/tables/table_advanced_holm_corrected_tests.csv`",
            "- `outputs/advanced/tables/table_model_confidence_set_qlike.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "advanced_hypothesis_tests_report.md", lines, force=force)


def choose_encompassing_candidates(predictions: pd.DataFrame) -> list[str]:
    candidates = ["ARIMA-GARCH", "LSTM-QLIKE", "Hybrid-QLIKE"]
    cal_path = ADVANCED_TABLES_DIR / "table_neural_calibration_validation.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        cal = cal[cal["method"] != "raw"].copy()
        if not cal.empty:
            candidates.append(str(cal.sort_values(["QLIKE", "RMSE", "calibrated_model"]).iloc[0]["calibrated_model"]))
    combo_path = ADVANCED_TABLES_DIR / "table_forecast_combination_validation.csv"
    if combo_path.exists():
        combo = pd.read_csv(combo_path)
        if not combo.empty:
            candidates.append(str(combo.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]["model"]))
    available = set(predictions["model"].unique())
    seen = set()
    out = []
    for model in candidates:
        if model in available and model not in seen and model != "GARCH(1,1)":
            out.append(model)
            seen.add(model)
    return out


def mincer_zarnowitz_row(predictions: pd.DataFrame, *, candidate: str, split: str) -> dict[str, Any]:
    aligned = align_two_models(predictions, model_a="GARCH(1,1)", model_b=candidate, split=split)
    y = aligned["actual_var"].to_numpy(dtype=float)
    x = aligned[["pred_a", "pred_b"]].to_numpy(dtype=float)
    if sm is None:
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        return {
            "split": split,
            "candidate": candidate,
            "n": int(len(aligned)),
            "intercept": float(beta[0]),
            "coef_garch": float(beta[1]),
            "coef_candidate": float(beta[2]),
            "se_garch": np.nan,
            "se_candidate": np.nan,
            "p_value_candidate": np.nan,
            "r_squared": np.nan,
            "covariance": "not_available_statsmodels_missing",
        }
    model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HC1")
    return {
        "split": split,
        "candidate": candidate,
        "n": int(len(aligned)),
        "intercept": float(model.params[0]),
        "coef_garch": float(model.params[1]),
        "coef_candidate": float(model.params[2]),
        "se_garch": float(model.bse[1]),
        "se_candidate": float(model.bse[2]),
        "p_value_candidate": float(model.pvalues[2]),
        "r_squared": float(model.rsquared),
        "covariance": "HC1",
    }


def incremental_dm_row(predictions: pd.DataFrame, *, candidate: str) -> dict[str, Any]:
    validation = align_two_models(predictions, model_a="GARCH(1,1)", model_b=candidate, split="validation")
    test = align_two_models(predictions, model_a="GARCH(1,1)", model_b=candidate, split="test")
    weights = np.linspace(0.0, 1.0, 101)
    losses = [
        qlike_objective(validation["actual_var"], w * validation["pred_a"] + (1.0 - w) * validation["pred_b"])
        for w in weights
    ]
    best_w = float(weights[int(np.argmin(losses))])
    combo_pred = best_w * test["pred_a"].to_numpy(dtype=float) + (1.0 - best_w) * test["pred_b"].to_numpy(dtype=float)
    loss_combo = loss_values(test["actual_var"], combo_pred, "qlike")
    loss_garch = loss_values(test["actual_var"], test["pred_a"], "qlike")
    diff = loss_combo - loss_garch
    stat, p_value, lag = dm_statistic(diff)
    return {
        "candidate": candidate,
        "benchmark": "GARCH(1,1)",
        "combination": f"GARCH(1,1)+{candidate}",
        "validation_weight_on_garch": best_w,
        "validation_weight_on_candidate": 1.0 - best_w,
        "n_test": int(len(diff)),
        "mean_loss_combo": float(np.mean(loss_combo)),
        "mean_loss_garch": float(np.mean(loss_garch)),
        "mean_diff_combo_minus_garch": float(np.mean(diff)),
        "dm_stat": stat,
        "p_value": p_value,
        "hac_lag": lag,
    }


def qlike_objective(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    pred_arr = np.clip(pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=float), 1e-8, None)
    actual_arr = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    return float(np.mean(np.log(pred_arr) + actual_arr / pred_arr))


def run_encompassing_tests(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_advanced_dirs()
    predictions = load_all_predictions(include_advanced=True)
    candidates = choose_encompassing_candidates(predictions)
    encompassing_rows = []
    incremental_rows = []
    for candidate in candidates:
        for split in ["validation", "test"]:
            try:
                encompassing_rows.append(mincer_zarnowitz_row(predictions, candidate=candidate, split=split))
            except Exception as exc:  # noqa: BLE001
                encompassing_rows.append(
                    {
                        "split": split,
                        "candidate": candidate,
                        "n": 0,
                        "intercept": np.nan,
                        "coef_garch": np.nan,
                        "coef_candidate": np.nan,
                        "se_garch": np.nan,
                        "se_candidate": np.nan,
                        "p_value_candidate": np.nan,
                        "r_squared": np.nan,
                        "covariance": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        try:
            incremental_rows.append(incremental_dm_row(predictions, candidate=candidate))
        except Exception as exc:  # noqa: BLE001
            incremental_rows.append(
                {
                    "candidate": candidate,
                    "benchmark": "GARCH(1,1)",
                    "combination": f"GARCH(1,1)+{candidate}",
                    "validation_weight_on_garch": np.nan,
                    "validation_weight_on_candidate": np.nan,
                    "n_test": 0,
                    "mean_loss_combo": np.nan,
                    "mean_loss_garch": np.nan,
                    "mean_diff_combo_minus_garch": np.nan,
                    "dm_stat": np.nan,
                    "p_value": np.nan,
                    "hac_lag": np.nan,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )
    encompassing = pd.DataFrame(encompassing_rows)
    incremental = pd.DataFrame(incremental_rows)
    safe_write_csv(encompassing, ADVANCED_TABLES_DIR / "table_forecast_encompassing.csv", force=force)
    safe_write_csv(incremental, ADVANCED_TABLES_DIR / "table_incremental_information_dm.csv", force=force)
    write_encompassing_report(encompassing, incremental, force=force)
    return encompassing, incremental


def write_encompassing_report(encompassing: pd.DataFrame, incremental: pd.DataFrame, *, force: bool) -> None:
    lines = [
        "# Forecast Encompassing Report",
        "",
        "## Regression Specification",
        "",
        "- `actual_var_t = a + b1 * pred_garch_t + b2 * pred_candidate_t + error_t`.",
        "- Robust HC1 standard errors are used when statsmodels is available.",
        "- Because squared return is a noisy volatility proxy, coefficient tests are interpreted cautiously.",
        "",
        "## Candidate Coefficients",
        "",
    ]
    if encompassing.empty:
        lines.append("- No encompassing regressions were available.")
    else:
        test_rows = encompassing[encompassing["split"] == "test"].copy()
        for _, row in test_rows.iterrows():
            p_text = "not_available" if pd.isna(row["p_value_candidate"]) else f"{float(row['p_value_candidate']):.6g}"
            lines.append(
                f"- {row['candidate']}: test candidate coefficient {float(row['coef_candidate']):.6f}, "
                f"p={p_text}, R2={float(row['r_squared']):.4f}."
            )
    lines.extend(["", "## Incremental QLIKE DM Tests", ""])
    if incremental.empty:
        lines.append("- No incremental-information DM tests were available.")
    else:
        for _, row in incremental.iterrows():
            p_text = "not_available" if pd.isna(row["p_value"]) else f"{float(row['p_value']):.6g}"
            lines.append(
                f"- {row['candidate']}: combo minus GARCH mean QLIKE diff "
                f"{float(row['mean_diff_combo_minus_garch']):.6f}, p={p_text}, "
                f"candidate weight {float(row['validation_weight_on_candidate']):.2f}."
            )
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_forecast_encompassing.csv`",
            "- `outputs/advanced/tables/table_incremental_information_dm.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "forecast_encompassing_report.md", lines, force=force)


def run_advanced_hypothesis_layer(*, force: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    run_encompassing_tests(force=force)
    return run_dm_and_wilcoxon(force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing advanced outputs.")
    args = parser.parse_args()
    run_advanced_hypothesis_layer(force=args.force)


if __name__ == "__main__":
    main()
