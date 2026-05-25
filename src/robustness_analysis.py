#!/usr/bin/env python3
"""Robustness and underprediction analysis on common test predictions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from empirical_rigor_utils import (
    AUDIT_DIR,
    BENCHMARK_MODEL,
    FIGURES_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    build_common_test_window,
    compute_metrics,
    ensure_output_dirs,
    load_all_predictions,
    loss_values,
    rel_path,
    write_markdown,
)


YEARLY_PATH = TABLES_DIR / "table_yearly_results.csv"
REGIME_PATH = TABLES_DIR / "table_regime_results.csv"
UNDERPREDICTION_PATH = TABLES_DIR / "table_underprediction_analysis.csv"
LOSS_DIFF_FIGURE_PATH = FIGURES_DIR / "fig_loss_difference_vs_garch.png"
REGIME_QLIKE_FIGURE_PATH = FIGURES_DIR / "fig_regime_qlike.png"
REPORT_PATH = AUDIT_DIR / "robustness_report.md"

REGIME_ORDER = ["calm", "high-volatility", "extreme-volatility"]


def metric_row(df: pd.DataFrame, *, model: str, label_column: str, label_value: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            label_column: label_value,
            "model": model,
            "n_obs": 0,
            "RMSE": "not_available",
            "MAE": "not_available",
            "QLIKE": "not_available",
            "mean_actual_var": "not_available",
            "mean_pred_var": "not_available",
        }
    metrics = compute_metrics(df)
    return {
        label_column: label_value,
        "model": model,
        "n_obs": metrics["n_obs"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "QLIKE": metrics["QLIKE"],
        "mean_actual_var": metrics["mean_actual_var"],
        "mean_pred_var": metrics["mean_pred_var"],
    }


def yearly_results(common_long: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for period in ["2023", "2024", "2025", "full_common_test"]:
        for model in models:
            model_df = common_long[common_long["model"] == model].copy()
            if period != "full_common_test":
                model_df = model_df[model_df["target_date"].dt.year == int(period)]
            rows.append(metric_row(model_df, model=model, label_column="period", label_value=period))
    return pd.DataFrame(rows)


def regime_masks(wide: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, float]]:
    actual = wide["actual_var"].astype(float)
    thresholds = {
        "median": float(actual.median()),
        "p75": float(actual.quantile(0.75)),
        "p90": float(actual.quantile(0.90)),
    }
    masks = {
        "calm": actual <= thresholds["median"],
        "high-volatility": actual > thresholds["p75"],
        "extreme-volatility": actual > thresholds["p90"],
    }
    return masks, thresholds


def regime_results(
    common_long: pd.DataFrame,
    wide: pd.DataFrame,
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    masks, thresholds = regime_masks(wide)
    rule_by_regime = {
        "calm": f"actual_var <= median ({thresholds['median']:.12g})",
        "high-volatility": f"actual_var > 75th percentile ({thresholds['p75']:.12g})",
        "extreme-volatility": f"actual_var > 90th percentile ({thresholds['p90']:.12g})",
    }
    key_sets = {
        regime: set(
            wide.loc[mask, ["date", "target_date"]].itertuples(index=False, name=None)
        )
        for regime, mask in masks.items()
    }

    rows: list[dict[str, float | int | str]] = []
    for regime in REGIME_ORDER:
        keys = pd.DataFrame(sorted(key_sets[regime]), columns=["date", "target_date"])
        for model in models:
            model_df = common_long[common_long["model"] == model]
            if keys.empty:
                subset = model_df.iloc[0:0].copy()
            else:
                subset = model_df.merge(keys, on=["date", "target_date"], how="inner")
            row = metric_row(subset, model=model, label_column="regime", label_value=regime)
            row["threshold_rule"] = rule_by_regime[regime]
            rows.append(row)
    return pd.DataFrame(rows), thresholds


def qlike_or_not_available(df: pd.DataFrame) -> float | str:
    if df.empty:
        return "not_available"
    return float(np.mean(loss_values(df["actual_var"], df["pred_var"], "qlike")))


def underprediction_analysis(
    common_long: pd.DataFrame,
    models: list[str],
    *,
    p75: float,
    p90: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in models:
        model_df = common_long[common_long["model"] == model].copy()
        metrics = compute_metrics(model_df)
        high_df = model_df[model_df["actual_var"] > p75]
        extreme_df = model_df[model_df["actual_var"] > p90]
        if extreme_df.empty:
            spike_rate: float | str = "not_available"
        else:
            spike_rate = float((extreme_df["pred_var"] < extreme_df["actual_var"]).mean())

        mean_actual = float(metrics["mean_actual_var"])
        mean_pred = float(metrics["mean_pred_var"])
        ratio: float | str = mean_pred / mean_actual if mean_actual > 0 else "not_available"
        rows.append(
            {
                "model": model,
                "n_obs": metrics["n_obs"],
                "mean_actual_var": mean_actual,
                "mean_pred_var": mean_pred,
                "pred_actual_ratio": ratio,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "QLIKE": metrics["QLIKE"],
                "high_vol_QLIKE": qlike_or_not_available(high_df),
                "extreme_vol_QLIKE": qlike_or_not_available(extreme_df),
                "spike_underprediction_rate": spike_rate,
            }
        )
    return pd.DataFrame(rows)


def plot_loss_difference_vs_garch(common_long: pd.DataFrame, models: list[str], path) -> None:
    if BENCHMARK_MODEL not in models:
        return

    benchmark_df = common_long.loc[
        common_long["model"] == BENCHMARK_MODEL, ["date", "target_date", "actual_var", "pred_var"]
    ].rename(columns={"pred_var": "pred_var_benchmark"})
    rows = []
    for model in models:
        if model == BENCHMARK_MODEL:
            continue
        model_df = common_long.loc[
            common_long["model"] == model, ["date", "target_date", "actual_var", "pred_var"]
        ].rename(columns={"pred_var": "pred_var_model"})
        merged = model_df.merge(
            benchmark_df[["date", "target_date", "pred_var_benchmark"]],
            on=["date", "target_date"],
            how="inner",
            validate="one_to_one",
        )
        diff = loss_values(merged["actual_var"], merged["pred_var_model"], "qlike") - loss_values(
            merged["actual_var"], merged["pred_var_benchmark"], "qlike"
        )
        rows.append({"model": model, "mean_qlike_diff": float(np.mean(diff))})

    plot_df = pd.DataFrame(rows).sort_values("mean_qlike_diff", ascending=True)
    colors = ["#2E7D59" if value < 0 else "#B94A48" for value in plot_df["mean_qlike_diff"]]
    fig_height = max(5.5, 0.36 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["mean_qlike_diff"], color=colors)
    ax.axvline(0.0, color="#222222", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel(f"Mean QLIKE loss difference vs {BENCHMARK_MODEL}")
    ax.set_title("Common Test QLIKE Loss Difference")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_regime_qlike(regime_df: pd.DataFrame, yearly_df: pd.DataFrame, path) -> None:
    full = yearly_df[yearly_df["period"] == "full_common_test"].copy()
    full["QLIKE_numeric"] = pd.to_numeric(full["QLIKE"], errors="coerce")
    models = full.sort_values(["QLIKE_numeric", "model"])["model"].tolist()
    plot_df = regime_df.copy()
    plot_df["QLIKE_numeric"] = pd.to_numeric(plot_df["QLIKE"], errors="coerce")

    x = np.arange(len(models))
    width = 0.24
    colors = {
        "calm": "#4C8C6B",
        "high-volatility": "#3B6EA8",
        "extreme-volatility": "#C44E52",
    }
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for index, regime in enumerate(REGIME_ORDER):
        values = (
            plot_df[plot_df["regime"] == regime]
            .set_index("model")
            .reindex(models)["QLIKE_numeric"]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (index - 1) * width, values, width=width, label=regime, color=colors[regime])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("QLIKE")
    ax.set_title("QLIKE by Volatility Regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def yearly_split_lines(wide: pd.DataFrame) -> list[str]:
    lines = ["| Period | Observations |", "|---|---:|"]
    for year in [2023, 2024, 2025]:
        count = int((wide["target_date"].dt.year == year).sum())
        lines.append(f"| {year} | {count} |")
    lines.append(f"| full_common_test | {len(wide)} |")
    return lines


def key_findings_lines(
    yearly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    under_df: pd.DataFrame,
) -> list[str]:
    full = yearly_df[yearly_df["period"] == "full_common_test"].copy()
    full["QLIKE_numeric"] = pd.to_numeric(full["QLIKE"], errors="coerce")
    full = full.sort_values(["QLIKE_numeric", "RMSE", "model"]).reset_index(drop=True)
    best = full.iloc[0]
    garch_rank = int(full.index[full["model"] == BENCHMARK_MODEL][0] + 1) if BENCHMARK_MODEL in set(full["model"]) else -1

    extreme = regime_df[regime_df["regime"] == "extreme-volatility"].copy()
    extreme["QLIKE_numeric"] = pd.to_numeric(extreme["QLIKE"], errors="coerce")
    best_extreme = extreme.sort_values(["QLIKE_numeric", "model"]).iloc[0]

    logtarget = under_df[under_df["model"].str.contains("LogTarget", case=False, regex=False)].copy()
    logtarget_lines: list[str]
    if logtarget.empty:
        logtarget_lines = ["- Log-target neural variants are not present in the common window."]
    else:
        logtarget["spike_rate_numeric"] = pd.to_numeric(
            logtarget["spike_underprediction_rate"], errors="coerce"
        )
        avg_rate = float(logtarget["spike_rate_numeric"].mean())
        worst = logtarget.sort_values("spike_rate_numeric", ascending=False).iloc[0]
        logtarget_lines = [
            "- Log-target neural variants have an average extreme-spike underprediction "
            f"rate of {avg_rate:.3f}; the highest is {worst['model']} at "
            f"{float(worst['spike_rate_numeric']):.3f}."
        ]

    return [
        f"- Best full-window QLIKE: {best['model']} ({float(best['QLIKE_numeric']):.6f}).",
        f"- {BENCHMARK_MODEL} full-window QLIKE rank: {garch_rank} of {len(full)}.",
        "- Best extreme-volatility QLIKE: "
        f"{best_extreme['model']} ({float(best_extreme['QLIKE_numeric']):.6f}).",
        *logtarget_lines,
    ]


def build_report(
    *,
    common_rows: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    thresholds: dict[str, float],
    wide: pd.DataFrame,
    yearly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    under_df: pd.DataFrame,
    source_files: list[str],
) -> list[str]:
    lines = [
        "# Robustness Analysis Report",
        "",
        "## Common Window Construction",
        "",
        "- Loaded all existing prediction CSV files under `outputs/predictions/`.",
        "- Validated the required prediction schema: `date,target_date,actual_var,pred_var,model,split`.",
        "- Used the exact intersection of test-split `(date, target_date)` keys across all models.",
        f"- Common test window: {common_rows} observations, target dates {target_start:%Y-%m-%d} to {target_end:%Y-%m-%d}.",
        "",
        "## Yearly Split Sizes",
        "",
        *yearly_split_lines(wide),
        "",
        "## Volatility Regime Thresholds",
        "",
        f"- calm: actual_var <= median = {thresholds['median']:.12g}.",
        f"- high-volatility: actual_var > 75th percentile = {thresholds['p75']:.12g}.",
        f"- extreme-volatility: actual_var > 90th percentile = {thresholds['p90']:.12g}.",
        "- High-volatility and extreme-volatility regimes intentionally overlap.",
        "",
        "## Underprediction Definition",
        "",
        "- `spike_underprediction_rate` is the proportion of extreme-volatility observations "
        "where `pred_var < actual_var`.",
        "",
        "## Key Findings",
        "",
        *key_findings_lines(yearly_df, regime_df, under_df),
        "",
        "## Input Files Used",
        "",
        *[f"- `{path}`" for path in source_files],
        "",
        "## Output Files",
        "",
        "- `outputs/tables/table_yearly_results.csv`",
        "- `outputs/tables/table_regime_results.csv`",
        "- `outputs/tables/table_underprediction_analysis.csv`",
        "- `outputs/figures/fig_loss_difference_vs_garch.png`",
        "- `outputs/figures/fig_regime_qlike.png`",
    ]
    return lines


def main() -> None:
    ensure_output_dirs()
    predictions, paths = load_all_predictions(PROJECT_ROOT)
    common = build_common_test_window(predictions)
    source_files = [rel_path(path) for path in paths]

    yearly_df = yearly_results(common.long, common.models)
    regime_df, thresholds = regime_results(common.long, common.wide, common.models)
    under_df = underprediction_analysis(
        common.long,
        common.models,
        p75=thresholds["p75"],
        p90=thresholds["p90"],
    )

    yearly_df.to_csv(YEARLY_PATH, index=False)
    regime_df.to_csv(REGIME_PATH, index=False)
    under_df.to_csv(UNDERPREDICTION_PATH, index=False)
    plot_loss_difference_vs_garch(common.long, common.models, LOSS_DIFF_FIGURE_PATH)
    plot_regime_qlike(regime_df, yearly_df, REGIME_QLIKE_FIGURE_PATH)

    write_markdown(
        REPORT_PATH,
        build_report(
            common_rows=len(common.common_keys),
            target_start=common.common_keys["target_date"].min(),
            target_end=common.common_keys["target_date"].max(),
            thresholds=thresholds,
            wide=common.wide,
            yearly_df=yearly_df,
            regime_df=regime_df,
            under_df=under_df,
            source_files=source_files,
        ),
    )

    print(f"Wrote {rel_path(YEARLY_PATH)}")
    print(f"Wrote {rel_path(REGIME_PATH)}")
    print(f"Wrote {rel_path(UNDERPREDICTION_PATH)}")
    print(f"Wrote {rel_path(LOSS_DIFF_FIGURE_PATH)}")
    print(f"Wrote {rel_path(REGIME_QLIKE_FIGURE_PATH)}")
    print(f"Wrote {rel_path(REPORT_PATH)}")


if __name__ == "__main__":
    main()
