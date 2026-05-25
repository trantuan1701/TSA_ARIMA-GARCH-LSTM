#!/usr/bin/env python3
"""Diebold-Mariano forecast comparison tests on raw prediction files."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from empirical_rigor_utils import (
    AUDIT_DIR,
    BENCHMARK_MODEL,
    PROJECT_ROOT,
    TABLES_DIR,
    build_common_test_window,
    ensure_output_dirs,
    load_all_predictions,
    loss_values,
    rel_path,
    significance_stars,
    write_markdown,
)


LOSS_OUTPUTS = {
    "qlike": TABLES_DIR / "table_dm_tests_qlike.csv",
    "squared_error": TABLES_DIR / "table_dm_tests_squared_error.csv",
    "absolute_error": TABLES_DIR / "table_dm_tests_absolute_error.csv",
}

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
    "significance",
]


def normal_p_value(statistic: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(statistic))))


def newey_west_long_run_variance(diff: np.ndarray) -> tuple[float, int]:
    """Bartlett-kernel long-run variance for the loss differential mean."""

    n_obs = len(diff)
    centered = diff - float(np.mean(diff))
    if n_obs <= 1:
        return 0.0, 0

    max_lag = min(int(np.floor(n_obs ** (1.0 / 3.0))), n_obs - 1)
    gamma_0 = float(np.dot(centered, centered) / n_obs)
    long_run_variance = gamma_0
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance

    if long_run_variance < 0 and abs(long_run_variance) < 1e-14:
        long_run_variance = 0.0
    return long_run_variance, max_lag


def dm_statistic(loss_diff: np.ndarray) -> tuple[float, float, int]:
    n_obs = len(loss_diff)
    mean_diff = float(np.mean(loss_diff))
    long_run_variance, hac_lag = newey_west_long_run_variance(loss_diff)
    if long_run_variance <= 0:
        if abs(mean_diff) <= 1e-14:
            return 0.0, 1.0, hac_lag
        return np.nan, np.nan, hac_lag

    statistic = mean_diff / np.sqrt(long_run_variance / n_obs)
    return float(statistic), normal_p_value(float(statistic)), hac_lag


def winner_label(model: str, benchmark: str, mean_diff: float, p_value: float) -> str:
    if not np.isfinite(p_value) or p_value >= 0.10:
        return "No significant difference"
    if mean_diff > 0:
        return benchmark
    if mean_diff < 0:
        return model
    return "Tie"


def compare_model_to_benchmark(
    common_long: pd.DataFrame,
    *,
    model: str,
    benchmark: str,
    loss_type: str,
) -> dict[str, float | int | str]:
    model_df = common_long.loc[
        common_long["model"] == model, ["date", "target_date", "actual_var", "pred_var"]
    ].rename(columns={"pred_var": "pred_var_model"})
    benchmark_df = common_long.loc[
        common_long["model"] == benchmark, ["date", "target_date", "pred_var"]
    ].rename(columns={"pred_var": "pred_var_benchmark"})
    merged = model_df.merge(benchmark_df, on=["date", "target_date"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"No common rows between {model} and {benchmark}.")

    loss_model = loss_values(merged["actual_var"], merged["pred_var_model"], loss_type)
    loss_benchmark = loss_values(merged["actual_var"], merged["pred_var_benchmark"], loss_type)
    diff = loss_model - loss_benchmark
    stat, p_value, _hac_lag = dm_statistic(diff)
    mean_diff = float(np.mean(diff))
    return {
        "model": model,
        "benchmark": benchmark,
        "loss_type": loss_type,
        "n": int(len(merged)),
        "mean_loss_model": float(np.mean(loss_model)),
        "mean_loss_benchmark": float(np.mean(loss_benchmark)),
        "mean_diff": mean_diff,
        "dm_stat": stat,
        "p_value": p_value,
        "winner": winner_label(model, benchmark, mean_diff, p_value),
        "significance": significance_stars(p_value),
    }


def save_empty_dm_tables(reason: str) -> None:
    empty = pd.DataFrame(columns=DM_COLUMNS)
    for path in LOSS_OUTPUTS.values():
        empty.to_csv(path, index=False)
    write_markdown(
        AUDIT_DIR / "statistical_tests_report.md",
        [
            "# Statistical Tests Report",
            "",
            "## Status",
            "",
            f"Diebold-Mariano tests are not_available: {reason}",
        ],
    )


def table_from_rows(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=DM_COLUMNS)
    return df.replace([np.inf, -np.inf], np.nan).fillna("not_available")


def build_report(
    *,
    common_window_rows: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    benchmark: str,
    compared_models: list[str],
    skipped: dict[str, str],
    source_files: list[str],
    dm_tables: dict[str, pd.DataFrame],
) -> list[str]:
    skipped_lines = [f"- {model}: {reason}" for model, reason in skipped.items()] or ["None."]
    model_lines = [f"- {model}" for model in compared_models] or ["None."]

    qlike = dm_tables["qlike"].copy()
    qlike["p_value_numeric"] = pd.to_numeric(qlike["p_value"], errors="coerce")
    significant = qlike[qlike["p_value_numeric"] < 0.10].sort_values("mean_diff")
    if significant.empty:
        significant_lines = ["- No QLIKE DM comparisons are significant at the 10% level."]
    else:
        significant_lines = [
            "- {model}: mean_diff={mean_diff:.6f}, p={p_value:.6g}, winner={winner}".format(
                model=row["model"],
                mean_diff=float(row["mean_diff"]),
                p_value=float(row["p_value_numeric"]),
                winner=row["winner"],
            )
            for _, row in significant.iterrows()
        ]

    lines = [
        "# Statistical Tests Report",
        "",
        "## Common Test Window",
        "",
        f"- Size: {common_window_rows} observations.",
        f"- Target-date range: {target_start:%Y-%m-%d} to {target_end:%Y-%m-%d}.",
        "- Constructed as the exact intersection of `(date, target_date)` keys across all test predictions.",
        "",
        "## Benchmark",
        "",
        f"- Benchmark model: {benchmark}.",
        "",
        "## Loss Difference Convention",
        "",
        f"- `d_t = loss_model_t - loss_{benchmark}_t`.",
        f"- Positive `mean_diff` means the compared model is worse than {benchmark}.",
        f"- Negative `mean_diff` means the compared model is better than {benchmark}.",
        "- DM statistics use a Bartlett/Newey-West long-run variance estimate with lag floor(n^(1/3)).",
        "",
        "## Compared Models",
        "",
        *model_lines,
        "",
        "## Skipped Models",
        "",
        *skipped_lines,
        "",
        "## QLIKE Significant Comparisons",
        "",
        *significant_lines,
        "",
        "## Input Files Used",
        "",
        *[f"- `{path}`" for path in source_files],
        "",
        "## Output Tables",
        "",
        "- `outputs/tables/table_dm_tests_qlike.csv`",
        "- `outputs/tables/table_dm_tests_squared_error.csv`",
        "- `outputs/tables/table_dm_tests_absolute_error.csv`",
    ]
    return lines


def main() -> None:
    ensure_output_dirs()
    predictions, paths = load_all_predictions(PROJECT_ROOT)
    common = build_common_test_window(predictions)
    source_files = [rel_path(path) for path in paths]

    if BENCHMARK_MODEL not in common.models:
        reason = f"benchmark model {BENCHMARK_MODEL!r} is absent from common test predictions"
        save_empty_dm_tables(reason)
        print(f"DM tests not_available: {reason}")
        return

    compared_models = [model for model in common.models if model != BENCHMARK_MODEL]
    skipped: dict[str, str] = {}
    dm_tables: dict[str, pd.DataFrame] = {}

    for loss_type, path in LOSS_OUTPUTS.items():
        rows = [
            compare_model_to_benchmark(
                common.long,
                model=model,
                benchmark=BENCHMARK_MODEL,
                loss_type=loss_type,
            )
            for model in compared_models
        ]
        table = table_from_rows(rows)
        table.to_csv(path, index=False)
        dm_tables[loss_type] = table
        print(f"Wrote {rel_path(path)}")

    report_path = AUDIT_DIR / "statistical_tests_report.md"
    write_markdown(
        report_path,
        build_report(
            common_window_rows=len(common.common_keys),
            target_start=common.common_keys["target_date"].min(),
            target_end=common.common_keys["target_date"].max(),
            benchmark=BENCHMARK_MODEL,
            compared_models=compared_models,
            skipped=skipped,
            source_files=source_files,
            dm_tables=dm_tables,
        ),
    )
    print(f"Wrote {rel_path(report_path)}")


if __name__ == "__main__":
    main()
