#!/usr/bin/env python3
"""Run and summarize the advanced VN-Index volatility experiment layer."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advanced_experiment_utils import (
    ADVANCED_AUDIT_DIR,
    ADVANCED_TABLES_DIR,
    PROJECT_ROOT,
    compute_metrics,
    ensure_advanced_dirs,
    load_all_predictions,
    safe_write_csv,
    underprediction_stats,
    write_markdown,
)
from advanced_forecast_combinations import run_calibration_and_combinations
from advanced_hypothesis_tests import run_advanced_hypothesis_layer
from advanced_model_search import run_model_search
from advanced_var_backtesting import run_var_backtesting


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_output_hashes() -> pd.DataFrame:
    outputs_dir = PROJECT_ROOT / "outputs"
    rows = []
    if not outputs_dir.exists():
        return pd.DataFrame(columns=["path", "size_bytes", "sha256"])
    for path in sorted(outputs_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            path.relative_to(outputs_dir / "advanced")
            continue
        except ValueError:
            pass
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_hash(path),
            }
        )
    return pd.DataFrame(rows)


def compare_hashes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="path", how="outer", suffixes=("_before", "_after"), indicator=True)
    merged["status"] = "unchanged"
    merged.loc[merged["_merge"] == "left_only", "status"] = "deleted"
    merged.loc[merged["_merge"] == "right_only", "status"] = "created"
    changed = (
        (merged["_merge"] == "both")
        & (
            merged["sha256_before"].astype(str).ne(merged["sha256_after"].astype(str))
            | merged["size_bytes_before"].astype(str).ne(merged["size_bytes_after"].astype(str))
        )
    )
    merged.loc[changed, "status"] = "modified"
    return merged.sort_values(["status", "path"]).reset_index(drop=True)


def model_family(model: str) -> str:
    if model.startswith("AdvGARCH-"):
        return "advanced_garch_family"
    if model.startswith("Refit-"):
        return "refit_garch"
    if model.startswith("Calibrated-"):
        return "calibrated_neural"
    if model.startswith("Combo"):
        return "forecast_combination"
    if "GARCH" in model and "LSTM" not in model:
        return "original_econometric"
    if "LSTM" in model or "Hybrid" in model:
        return "original_neural_or_hybrid"
    if "RollingVol" in model or model == "HistoricalMean":
        return "baseline"
    return "other"


def selected_by_label(model: str) -> str:
    if model.startswith("AdvGARCH-"):
        return "validation_QLIKE_garch_family"
    if model.startswith("Refit-"):
        return "validation_QLIKE_refit_protocol"
    if model.startswith("Calibrated-"):
        return "validation_calibration"
    if model.startswith("Combo"):
        return "validation_QLIKE_forecast_combination"
    return "original_existing_model"


def metrics_by_model_split(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), group in predictions.groupby(["model", "split"]):
        rows.append({"model": model, "split": split, **compute_metrics(group)})
    return pd.DataFrame(rows)


def final_leaderboard(*, force: bool = False) -> pd.DataFrame:
    predictions = load_all_predictions(include_advanced=True)
    metrics = metrics_by_model_split(predictions)
    validation = metrics[metrics["split"] == "validation"].set_index("model")
    test = metrics[metrics["split"] == "test"].set_index("model")
    mcs_path = ADVANCED_TABLES_DIR / "table_model_confidence_set_qlike.csv"
    mcs = pd.read_csv(mcs_path).set_index("model") if mcs_path.exists() else pd.DataFrame()

    rows = []
    for model in sorted(set(predictions["model"])):
        val = validation.loc[model] if model in validation.index else pd.Series(dtype=float)
        tst = test.loc[model] if model in test.index else pd.Series(dtype=float)
        test_df = predictions[(predictions["model"] == model) & (predictions["split"] == "test")]
        under = underprediction_stats(test_df)
        rows.append(
            {
                "model": model,
                "family": model_family(model),
                "selected_by": selected_by_label(model),
                "uses_test_for_selection": False,
                "validation_RMSE": val.get("RMSE", np.nan),
                "validation_MAE": val.get("MAE", np.nan),
                "validation_QLIKE": val.get("QLIKE", np.nan),
                "test_RMSE": tst.get("RMSE", np.nan),
                "test_MAE": tst.get("MAE", np.nan),
                "test_QLIKE": tst.get("QLIKE", np.nan),
                "pred_actual_ratio": under["pred_actual_ratio"],
                "spike_underprediction_rate": under["spike_underprediction_rate"],
                "included_in_MCS_90": bool(mcs.loc[model, "included_in_MCS_90"])
                if not mcs.empty and model in mcs.index
                else pd.NA,
                "included_in_MCS_95": bool(mcs.loc[model, "included_in_MCS_95"])
                if not mcs.empty and model in mcs.index
                else pd.NA,
                "notes": "",
            }
        )
    leaderboard = pd.DataFrame(rows)
    for metric, rank_col in [
        ("test_QLIKE", "test_rank_QLIKE"),
        ("test_RMSE", "test_rank_RMSE"),
        ("test_MAE", "test_rank_MAE"),
    ]:
        leaderboard[rank_col] = leaderboard[metric].rank(method="min", ascending=True)
    ordered_cols = [
        "model",
        "family",
        "selected_by",
        "uses_test_for_selection",
        "validation_RMSE",
        "validation_MAE",
        "validation_QLIKE",
        "test_RMSE",
        "test_MAE",
        "test_QLIKE",
        "test_rank_QLIKE",
        "test_rank_RMSE",
        "test_rank_MAE",
        "pred_actual_ratio",
        "spike_underprediction_rate",
        "included_in_MCS_90",
        "included_in_MCS_95",
        "notes",
    ]
    leaderboard = leaderboard[ordered_cols].sort_values(["test_QLIKE", "test_RMSE", "model"]).reset_index(drop=True)
    safe_write_csv(leaderboard, ADVANCED_TABLES_DIR / "table_advanced_final_leaderboard.csv", force=force)
    return leaderboard


def dm_lookup(model: str, benchmark: str = "GARCH(1,1)") -> dict[str, Any]:
    path = ADVANCED_TABLES_DIR / "table_advanced_dm_tests_qlike.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    row = table[(table["model"] == model) & (table["benchmark"] == benchmark)]
    return row.iloc[0].to_dict() if not row.empty else {}


def holm_lookup(model: str, benchmark: str = "GARCH(1,1)") -> dict[str, Any]:
    path = ADVANCED_TABLES_DIR / "table_advanced_holm_corrected_tests.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    row = table[
        (table["model"] == model)
        & (table["benchmark"] == benchmark)
        & (table["test_type"] == "DM")
        & (table["loss_type"] == "qlike")
    ]
    return row.iloc[0].to_dict() if not row.empty else {}


def leaderboard_row(leaderboard: pd.DataFrame, model: str) -> pd.Series:
    row = leaderboard[leaderboard["model"] == model]
    return row.iloc[0] if not row.empty else pd.Series(dtype=object)


def best_model_with_prefix(leaderboard: pd.DataFrame, prefix: str) -> str | None:
    frame = leaderboard[leaderboard["model"].astype(str).str.startswith(prefix)].copy()
    if frame.empty:
        return None
    return str(frame.sort_values(["validation_QLIKE", "test_QLIKE", "model"]).iloc[0]["model"])


def hypothesis_summary(leaderboard: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    rows = []

    def add(hypothesis: str, status: str, evidence: str, caveats: str) -> None:
        rows.append(
            {
                "hypothesis": hypothesis,
                "status": status,
                "key_evidence": evidence,
                "caveats": caveats,
            }
        )

    garch = leaderboard_row(leaderboard, "GARCH(1,1)")
    arima = leaderboard_row(leaderboard, "ARIMA-GARCH")
    dm_arima = dm_lookup("ARIMA-GARCH")
    h1_status = "not_supported"
    if not arima.empty and not garch.empty and float(arima["validation_QLIKE"]) < float(garch["validation_QLIKE"]):
        h1_status = "partially_supported"
    if dm_arima and dm_arima.get("p_value", 1.0) < 0.10 and dm_arima.get("mean_diff", 1.0) < 0:
        h1_status = "supported"
    add(
        "H1: Mean dynamics add value beyond GARCH.",
        h1_status,
        f"ARIMA-GARCH validation/test QLIKE {arima.get('validation_QLIKE', np.nan):.6f}/{arima.get('test_QLIKE', np.nan):.6f}; "
        f"GARCH {garch.get('validation_QLIKE', np.nan):.6f}/{garch.get('test_QLIKE', np.nan):.6f}; "
        f"DM p={dm_arima.get('p_value', np.nan)}.",
        "ARIMA mean value is assessed through noisy squared-return volatility proxies and QLIKE loss.",
    )

    best_adv = leaderboard_row(leaderboard, "AdvGARCH-BestQLIKE")
    dm_adv = dm_lookup("AdvGARCH-BestQLIKE")
    holm_adv = holm_lookup("AdvGARCH-BestQLIKE")
    h2_status = "not_supported"
    if not best_adv.empty and float(best_adv["test_QLIKE"]) < float(garch["test_QLIKE"]):
        h2_status = "partially_supported"
    if dm_adv and dm_adv.get("p_value", 1.0) < 0.10 and dm_adv.get("mean_diff", 1.0) < 0:
        h2_status = "supported"
    add(
        "H2: Heavy-tailed/asymmetric GARCH variants improve forecasts.",
        h2_status,
        f"Best advanced GARCH validation/test QLIKE {best_adv.get('validation_QLIKE', np.nan):.6f}/"
        f"{best_adv.get('test_QLIKE', np.nan):.6f}; DM p={dm_adv.get('p_value', np.nan)}; "
        f"Holm p={holm_adv.get('holm_adjusted_p_value', np.nan)}.",
        "The selected best GARCH-family model may be symmetric or asymmetric; see selected-model table for specification.",
    )

    best_refit_model = best_model_with_prefix(leaderboard, "Refit-")
    best_refit = leaderboard_row(leaderboard, best_refit_model) if best_refit_model else pd.Series(dtype=object)
    dm_refit = dm_lookup(best_refit_model) if best_refit_model else {}
    h3_status = "inconclusive"
    if best_refit_model:
        h3_status = "partially_supported" if float(best_refit["test_QLIKE"]) < float(garch["test_QLIKE"]) else "not_supported"
        if dm_refit and dm_refit.get("p_value", 1.0) < 0.10 and dm_refit.get("mean_diff", 1.0) < 0:
            h3_status = "supported"
    add(
        "H3: Rolling/expanding refits improve forecasts.",
        h3_status,
        f"Best validation-selected refit model {best_refit_model}; validation/test QLIKE "
        f"{best_refit.get('validation_QLIKE', np.nan):.6f}/{best_refit.get('test_QLIKE', np.nan):.6f}; "
        f"DM p={dm_refit.get('p_value', np.nan)}.",
        "Protocol selection uses validation QLIKE; test table also reports non-selected protocol outcomes for audit only.",
    )

    best_combo_model = best_model_with_prefix(leaderboard, "Combo")
    best_combo = leaderboard_row(leaderboard, best_combo_model) if best_combo_model else pd.Series(dtype=object)
    best_econ = leaderboard[leaderboard["family"].isin(["original_econometric", "advanced_garch_family", "refit_garch"])]
    econ_name = str(best_econ.sort_values("validation_QLIKE").iloc[0]["model"]) if not best_econ.empty else "not_available"
    econ_row = leaderboard_row(leaderboard, econ_name)
    dm_combo = dm_lookup(best_combo_model) if best_combo_model else {}
    h4_status = "inconclusive"
    if best_combo_model and not econ_row.empty:
        h4_status = (
            "partially_supported"
            if float(best_combo["test_QLIKE"]) < float(econ_row["test_QLIKE"])
            else "not_supported"
        )
        if dm_combo and dm_combo.get("p_value", 1.0) < 0.10 and dm_combo.get("mean_diff", 1.0) < 0:
            h4_status = "supported"
    add(
        "H4: Hybrid/neural forecasts add incremental information.",
        h4_status,
        f"Best combo {best_combo_model} validation/test QLIKE {best_combo.get('validation_QLIKE', np.nan):.6f}/"
        f"{best_combo.get('test_QLIKE', np.nan):.6f}; best validation econometric {econ_name} test QLIKE "
        f"{econ_row.get('test_QLIKE', np.nan):.6f}; combo DM p={dm_combo.get('p_value', np.nan)}.",
        "Combination gains can be small and are tested again in encompassing and incremental-information tables.",
    )

    cal_path = ADVANCED_TABLES_DIR / "table_neural_calibration_underprediction.csv"
    h5_status = "inconclusive"
    h5_evidence = "Calibration table unavailable."
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        log_rows = cal[cal["base_model"].astype(str).str.contains("LogTarget", case=False, regex=False)].copy()
        raw = log_rows[log_rows["method"] == "raw"]
        calibrated = log_rows[log_rows["method"] != "raw"]
        if not raw.empty and not calibrated.empty:
            best_cal = calibrated.sort_values("test_QLIKE").groupby("base_model").head(1)
            raw_mean_qlike = float(raw["test_QLIKE"].mean())
            cal_mean_qlike = float(best_cal["test_QLIKE"].mean())
            raw_ratio = float(raw["pred_actual_ratio"].mean())
            cal_ratio = float(best_cal["pred_actual_ratio"].mean())
            h5_status = "partially_supported" if cal_mean_qlike < raw_mean_qlike and cal_ratio > raw_ratio else "not_supported"
            h5_evidence = (
                f"Average raw log-target test QLIKE {raw_mean_qlike:.6f}, best calibrated {cal_mean_qlike:.6f}; "
                f"pred/actual ratio {raw_ratio:.3f} -> {cal_ratio:.3f}."
            )
    add(
        "H5: Log-target neural models mainly suffer from scale bias.",
        h5_status,
        h5_evidence,
        "If calibrated QLIKE remains materially worse than GARCH, the issue is not just scale bias.",
    )

    var_path = ADVANCED_TABLES_DIR / "table_var_backtesting_5pct.csv"
    h6_status = "inconclusive"
    h6_evidence = "VaR table unavailable."
    if var_path.exists():
        var5 = pd.read_csv(var_path)
        if not var5.empty:
            closest = var5.assign(distance=(var5["violation_rate"] - 0.05).abs()).sort_values("distance").iloc[0]
            best_qlike = leaderboard.sort_values("test_QLIKE").iloc[0]
            h6_status = "partially_supported" if closest["model"] == best_qlike["model"] else "inconclusive"
            h6_evidence = (
                f"Closest 5% VaR coverage: {closest['model']} violation rate {float(closest['violation_rate']):.3f}; "
                f"best test QLIKE model: {best_qlike['model']}."
            )
    add(
        "H6: Best QLIKE models improve risk management.",
        h6_status,
        h6_evidence,
        "VaR uses Normal quantiles for all models, so distribution-specific tail parameters are not credited here.",
    )

    table = pd.DataFrame(rows)
    safe_write_csv(table, ADVANCED_TABLES_DIR / "table_advanced_hypothesis_summary.csv", force=force)
    return table


def write_summary_report(
    leaderboard: pd.DataFrame,
    hypotheses: pd.DataFrame,
    integrity: pd.DataFrame,
    *,
    force: bool,
) -> None:
    best = leaderboard.sort_values("test_QLIKE").iloc[0] if not leaderboard.empty else None
    modified_original = integrity[integrity["status"].isin(["modified", "created", "deleted"])]
    lines = [
        "# Advanced Experiment Summary",
        "",
        "## Top Test QLIKE Models",
        "",
    ]
    for _, row in leaderboard.head(10).iterrows():
        lines.append(
            f"- {row['model']}: test QLIKE {float(row['test_QLIKE']):.6f}, "
            f"validation QLIKE {float(row['validation_QLIKE']):.6f}, family `{row['family']}`."
        )
    lines.extend(["", "## Best Model", ""])
    if best is not None:
        lines.append(
            f"- Best test QLIKE: `{best['model']}` ({float(best['test_QLIKE']):.6f})."
        )
    lines.extend(["", "## Hypothesis Summary", ""])
    for _, row in hypotheses.iterrows():
        lines.append(f"- {row['hypothesis']} Status: `{row['status']}`. {row['key_evidence']}")
    lines.extend(["", "## Original Output Integrity", ""])
    if modified_original.empty:
        lines.append("- No existing output files outside `outputs/advanced/` changed during `run_advanced_experiments.py`.")
    else:
        for _, row in modified_original.iterrows():
            lines.append(f"- {row['status']}: `{row['path']}`")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_advanced_final_leaderboard.csv`",
            "- `outputs/advanced/tables/table_advanced_hypothesis_summary.csv`",
            "- `outputs/advanced/tables/table_original_output_integrity.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "advanced_experiment_summary.md", lines, force=force)


def run_all(*, force: bool = False) -> dict[str, Any]:
    ensure_advanced_dirs()
    before = original_output_hashes()
    print("Running advanced GARCH-family model search and refit protocols...")
    all_candidates, failures, selected = run_model_search(force=force)
    print("Running neural calibration and forecast combinations...")
    run_calibration_and_combinations(force=force)
    print("Running encompassing, DM, Wilcoxon, Holm, and MCS tests...")
    run_advanced_hypothesis_layer(force=force)
    print("Running VaR backtesting...")
    run_var_backtesting(force=force)
    print("Writing final advanced leaderboard and hypothesis summary...")
    leaderboard = final_leaderboard(force=force)
    hypotheses = hypothesis_summary(leaderboard, force=force)
    after = original_output_hashes()
    integrity = compare_hashes(before, after)
    safe_write_csv(integrity, ADVANCED_TABLES_DIR / "table_original_output_integrity.csv", force=force)
    write_summary_report(leaderboard, hypotheses, integrity, force=force)
    return {
        "all_candidates": all_candidates,
        "failures": failures,
        "selected": selected,
        "leaderboard": leaderboard,
        "hypotheses": hypotheses,
        "integrity": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing advanced outputs.")
    args = parser.parse_args()
    run_all(force=args.force)


if __name__ == "__main__":
    main()
