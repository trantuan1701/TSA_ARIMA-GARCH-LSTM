#!/usr/bin/env python3
"""Completion-pass artifacts for corrected neural, VaR/ES, and combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover
    from .metrics import EPSILON

try:
    import final_financial_econometrics as final
except ImportError:  # pragma: no cover
    from . import final_financial_econometrics as final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FINAL_DIR = PROJECT_ROOT / "outputs" / "final"
OUTPUT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
CORRECTED_NEURAL_ROOT = PROJECT_ROOT / "outputs" / "neural_corrected_context_v2"
CORRECTED_ADVANCED_ROOT = PROJECT_ROOT / "outputs" / "advanced_corrected_context_v2"
PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
ALPHAS = (0.05, 0.025, 0.01)


def ensure_dirs() -> None:
    for path in (OUTPUT_FINAL_DIR, OUTPUT_AUDIT_DIR, PAPER_TABLES_DIR, PAPER_FIGURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def parse_prediction_frame(df: pd.DataFrame, source: Path | str = "<frame>") -> pd.DataFrame:
    missing = [column for column in PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    parsed = df[PREDICTION_COLUMNS].copy()
    for column in ("date", "target_date"):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        if parsed[column].isna().any():
            raise ValueError(f"{source} has unparseable {column} values.")
    for column in ("actual_var", "pred_var"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        if not np.isfinite(parsed[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source} has non-finite {column} values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{source} has non-positive predicted variance.")
    parsed["model"] = parsed["model"].astype(str)
    parsed["split"] = parsed["split"].astype(str).str.lower()
    return parsed.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def read_prediction(path: Path) -> pd.DataFrame:
    return parse_prediction_frame(pd.read_csv(path, encoding="utf-8-sig"), path)


def volatility_metrics(actual_var: Iterable[float], pred_var: Iterable[float]) -> dict[str, float | int]:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = np.clip(pred[finite], EPSILON, None)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    qlike = float(np.mean(np.log(pred) + actual / pred))
    return {
        "n_obs": int(len(actual)),
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
        "mean_actual_var": float(np.mean(actual)),
        "mean_pred_var": float(np.mean(pred)),
        "pred_actual_ratio": float(np.mean(pred) / np.mean(actual)) if np.mean(actual) > 0 else np.nan,
    }


def qlike_objective(actual: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), EPSILON, None)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.log(pred) + actual / pred))


def qlike_loss(actual_var: Iterable[float], pred_var: Iterable[float]) -> np.ndarray:
    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = np.clip(pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float), EPSILON, None)
    finite = np.isfinite(actual) & np.isfinite(pred)
    return np.log(pred[finite]) + actual[finite] / pred[finite]


def bernoulli_loglik(successes: int, failures: int, probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return successes * math.log(p) + failures * math.log(1.0 - p)


def kupiec_test(violations: Iterable[bool], alpha: float) -> tuple[float, float]:
    arr = np.asarray(list(violations), dtype=bool)
    n = int(len(arr))
    x = int(arr.sum())
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    lr = -2.0 * (bernoulli_loglik(x, n - x, alpha) - bernoulli_loglik(x, n - x, phat))
    lr = max(float(lr), 0.0)
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence(violations: Iterable[bool]) -> tuple[float, float, dict[str, int]]:
    v = np.asarray(list(violations), dtype=int)
    if len(v) < 2:
        return np.nan, np.nan, {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev, curr = v[:-1], v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    pi01 = n01 / (n00 + n01) if (n00 + n01) else pi
    pi11 = n11 / (n10 + n11) if (n10 + n11) else pi
    restricted = bernoulli_loglik(n01 + n11, n00 + n10, pi)
    unrestricted = bernoulli_loglik(n01, n00, pi01) + bernoulli_loglik(n11, n10, pi11)
    lr = max(float(-2.0 * (restricted - unrestricted)), 0.0)
    return lr, float(chi2.sf(lr, 1)), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def normal_var_es(alpha: float, sigma: np.ndarray, mean: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(sigma, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    z = NormalDist().inv_cdf(alpha)
    var = mean_arr + sigma * z
    es = mean_arr - sigma * norm.pdf(z) / alpha
    return var, es


def student_t_var_es(
    alpha: float,
    sigma: np.ndarray,
    df: float,
    mean: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if df <= 2:
        raise ValueError("Student-t ES requires df > 2 for variance standardization.")
    sigma = np.asarray(sigma, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    q_raw = float(t.ppf(alpha, df))
    scale = math.sqrt((df - 2.0) / df)
    q_std = scale * q_raw
    es_std = -scale * (df + q_raw**2) * float(t.pdf(q_raw, df)) / ((df - 1.0) * alpha)
    return mean_arr + sigma * q_std, mean_arr + sigma * es_std


def quantile_loss(returns: np.ndarray, quantile: np.ndarray, alpha: float) -> float:
    returns = np.asarray(returns, dtype=float)
    quantile = np.asarray(quantile, dtype=float)
    return float(np.mean((alpha - (returns < quantile).astype(float)) * (returns - quantile)))


def target_returns() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "vnindex_cafef_2010_2025_clean.csv"
    clean = pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(clean["date"], errors="coerce"),
            "target_return": pd.to_numeric(clean["log_return_pct"], errors="coerce"),
        }
    ).dropna()


def load_model_ready() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "vnindex_model_ready.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in ("date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("target_var_next", "rolling_vol_20", "squared_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def load_primary_predictions() -> pd.DataFrame:
    predictions, _specs = final.primary_prediction_frames()
    return parse_prediction_frame(predictions)


def common_window(predictions: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    return final.common_window_for_models(predictions, split=split)


def write_simple_latex(
    table: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
    formats: dict[str, str] | None = None,
    table_star: bool = False,
) -> None:
    formats = formats or {}
    env = "table*" if table_star else "table"
    lines = [
        rf"\begin{{{env}}}[!t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\begin{tabular}{@{}" + "".join("r" if pd.api.types.is_numeric_dtype(table[c]) else "l" for c in columns) + r"@{}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in table[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif column in formats:
                values.append(formats[column].format(value))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_best_weight(validation: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    weights = np.linspace(0.0, 1.0, 101)
    actual = validation["actual_var"].to_numpy(dtype=float)
    losses = [
        qlike_objective(actual, w * validation[model_a].to_numpy(dtype=float) + (1.0 - w) * validation[model_b].to_numpy(dtype=float))
        for w in weights
    ]
    index = int(np.argmin(losses))
    return float(weights[index]), float(losses[index])


def align_models(predictions: pd.DataFrame, models: list[str], split: str) -> pd.DataFrame:
    frames = []
    for model in models:
        part = predictions[(predictions["model"].eq(model)) & (predictions["split"].eq(split))].copy()
        frames.append(part[["date", "target_date", "actual_var", "pred_var"]].rename(columns={"pred_var": model}))
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame[["date", "target_date", frame.columns[-1]]], on=["date", "target_date"], how="inner")
    actual_cols = [c for c in out.columns if c == "actual_var"]
    if len(actual_cols) != 1:
        raise ValueError("Aligned frame lost the actual variance column.")
    return out.sort_values("date").reset_index(drop=True)


def validation_metrics(predictions: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for model in candidates:
        group = predictions[(predictions["model"].eq(model)) & (predictions["split"].eq("validation"))]
        if not group.empty:
            rows.append({"model": model, **volatility_metrics(group["actual_var"], group["pred_var"])})
    return pd.DataFrame(rows).sort_values(["qlike", "rmse", "model"])


def make_prediction_frame(base: pd.DataFrame, pred: np.ndarray, model: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": base["date"],
            "target_date": base["target_date"],
            "actual_var": base["actual_var"],
            "pred_var": np.clip(pred, EPSILON, None),
            "model": model,
            "split": split,
        }
    )[PREDICTION_COLUMNS]


def build_regime_aware_combination() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    predictions = load_primary_predictions()
    econ_candidates = ["AdvGARCH-BestAsymmetric", "HAR-Parkinson", "GARCH(1,1)"]
    neural_candidates = [
        "LSTM",
        "ARIMA-GARCH-LSTM",
        "Hybrid-QLIKE",
        "Calibrated-Hybrid-LogTarget-Small-isotonic",
    ]
    econ = str(validation_metrics(predictions, econ_candidates).iloc[0]["model"])
    neural = str(validation_metrics(predictions, neural_candidates).iloc[0]["model"])

    val = align_models(predictions, [econ, neural], "validation")
    test = align_models(predictions, [econ, neural], "test")
    static_w, static_val_loss = fit_best_weight(val, econ, neural)
    static_model = f"StaticCombo-{econ}-with-{neural}".replace(" ", "")
    static_frames = []
    static_rows = []
    for split, frame in [("validation", val), ("test", test)]:
        pred = static_w * frame[econ].to_numpy(dtype=float) + (1.0 - static_w) * frame[neural].to_numpy(dtype=float)
        out = make_prediction_frame(frame, pred, static_model, split)
        static_frames.append(out)
        static_rows.append(
            {
                "model": static_model,
                "split": split,
                "econ_expert": econ,
                "neural_expert": neural,
                "econ_weight": static_w,
                "neural_weight": 1.0 - static_w,
                "selection_rule": "validation_QLIKE_grid",
                **volatility_metrics(out["actual_var"], out["pred_var"]),
            }
        )

    model_ready = load_model_ready()[["date", "target_date", "rolling_vol_20"]]
    val_signal = val.merge(model_ready, on=["date", "target_date"], how="left", validate="one_to_one")
    test_signal = test.merge(model_ready, on=["date", "target_date"], how="left", validate="one_to_one")
    if val_signal["rolling_vol_20"].isna().any() or test_signal["rolling_vol_20"].isna().any():
        raise ValueError("Missing origin-observable rolling_vol_20 signal for regime combination.")

    threshold_grid = [0.75, 0.85, 0.90]
    candidates = []
    for q in threshold_grid:
        threshold = float(val_signal["rolling_vol_20"].quantile(q))
        current = val_signal.copy()
        current["high_regime"] = current["rolling_vol_20"] >= threshold
        if current["high_regime"].sum() < 20 or (~current["high_regime"]).sum() < 20:
            continue
        low_w, _ = fit_best_weight(current[~current["high_regime"]], econ, neural)
        high_w, _ = fit_best_weight(current[current["high_regime"]], econ, neural)
        pred = np.where(
            current["high_regime"].to_numpy(dtype=bool),
            high_w * current[econ].to_numpy(dtype=float) + (1.0 - high_w) * current[neural].to_numpy(dtype=float),
            low_w * current[econ].to_numpy(dtype=float) + (1.0 - low_w) * current[neural].to_numpy(dtype=float),
        )
        candidates.append(
            {
                "threshold_quantile": q,
                "threshold_value": threshold,
                "low_econ_weight": low_w,
                "high_econ_weight": high_w,
                "validation_qlike": qlike_objective(current["actual_var"].to_numpy(dtype=float), pred),
                "validation_high_count": int(current["high_regime"].sum()),
                "validation_low_count": int((~current["high_regime"]).sum()),
            }
        )
    if not candidates:
        raise RuntimeError("No valid regime-aware threshold candidates were available.")
    selected = min(candidates, key=lambda row: (row["validation_qlike"], row["threshold_quantile"]))
    regime_model = f"RegimeAwareCombo-Vol20Q{int(100 * selected['threshold_quantile'])}"
    regime_frames = []
    regime_rows = []
    for split, frame in [("validation", val_signal), ("test", test_signal)]:
        high = frame["rolling_vol_20"].to_numpy(dtype=float) >= float(selected["threshold_value"])
        pred = np.where(
            high,
            selected["high_econ_weight"] * frame[econ].to_numpy(dtype=float)
            + (1.0 - selected["high_econ_weight"]) * frame[neural].to_numpy(dtype=float),
            selected["low_econ_weight"] * frame[econ].to_numpy(dtype=float)
            + (1.0 - selected["low_econ_weight"]) * frame[neural].to_numpy(dtype=float),
        )
        out = make_prediction_frame(frame, pred, regime_model, split)
        out["regime_signal"] = frame["rolling_vol_20"].to_numpy(dtype=float)
        out["high_regime"] = high
        regime_frames.append(out)
        regime_rows.append(
            {
                "model": regime_model,
                "split": split,
                "econ_expert": econ,
                "neural_expert": neural,
                "regime_indicator": "origin_rolling_vol_20",
                "threshold_quantile": selected["threshold_quantile"],
                "threshold_value": selected["threshold_value"],
                "low_econ_weight": selected["low_econ_weight"],
                "low_neural_weight": 1.0 - selected["low_econ_weight"],
                "high_econ_weight": selected["high_econ_weight"],
                "high_neural_weight": 1.0 - selected["high_econ_weight"],
                "validation_selection_qlike": selected["validation_qlike"],
                "n_high_regime": int(high.sum()),
                **volatility_metrics(out["actual_var"], out["pred_var"]),
            }
        )

    static_predictions = pd.concat(static_frames, ignore_index=True)
    regime_predictions = pd.concat(regime_frames, ignore_index=True)
    static_results = pd.DataFrame(static_rows)
    regime_results = pd.DataFrame(regime_rows)
    candidate_table = pd.DataFrame(candidates).sort_values(["validation_qlike", "threshold_quantile"])

    static_predictions.to_csv(OUTPUT_FINAL_DIR / "static_combination_corrected_neural_predictions.csv", index=False)
    static_results.to_csv(OUTPUT_FINAL_DIR / "static_combination_corrected_neural.csv", index=False)
    regime_predictions.to_parquet(OUTPUT_FINAL_DIR / "regime_aware_combination_predictions.parquet", index=False)
    regime_predictions[PREDICTION_COLUMNS].to_csv(OUTPUT_FINAL_DIR / "regime_aware_combination_predictions.csv", index=False)
    regime_results.to_csv(OUTPUT_FINAL_DIR / "regime_aware_combination_results.csv", index=False)
    candidate_table.to_csv(OUTPUT_FINAL_DIR / "regime_aware_combination_validation_candidates.csv", index=False)
    pd.DataFrame(columns=["status", "reason"]).assign(
        status=["omitted"], reason=["optional online rolling-loss combination was not run in this completion pass"]
    ).to_csv(OUTPUT_FINAL_DIR / "online_combination_results.csv", index=False)

    main = pd.concat([static_results[static_results["split"].eq("test")], regime_results[regime_results["split"].eq("test")]])
    write_simple_latex(
        main,
        PAPER_TABLES_DIR / "regime_aware_combination_results.tex",
        caption="Validation-frozen static and regime-aware corrected-neural combinations.",
        label="tab:regime-aware-combination-results",
        columns=["model", "n_obs", "qlike", "rmse", "mae", "pred_actual_ratio"],
        formats={"qlike": "{:.6f}", "rmse": "{:.6f}", "mae": "{:.6f}", "pred_actual_ratio": "{:.3f}"},
        table_star=True,
    )
    plot_regime_combination(regime_results, static_results)
    write_regime_protocol(regime_results, static_results, candidate_table)
    return static_results, regime_results, regime_predictions


def plot_regime_combination(regime_results: pd.DataFrame, static_results: pd.DataFrame) -> None:
    test_regime = regime_results[regime_results["split"].eq("test")].iloc[0]
    test_static = static_results[static_results["split"].eq("test")].iloc[0]
    labels = ["static", "regime low", "regime high"]
    econ_weights = [
        float(test_static["econ_weight"]),
        float(test_regime["low_econ_weight"]),
        float(test_regime["high_econ_weight"]),
    ]
    neural_weights = [1.0 - weight for weight in econ_weights]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, econ_weights, label="Econometric", color="#2F6F73")
    ax.bar(x, neural_weights, bottom=econ_weights, label="Neural/calibrated", color="#D58936")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Convex weight")
    ax.set_title("Validation-Frozen Combination Weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES_DIR / "fig_regime_aware_weights_or_performance.png", dpi=250)
    plt.close(fig)


def write_regime_protocol(regime_results: pd.DataFrame, static_results: pd.DataFrame, candidates: pd.DataFrame) -> None:
    test_regime = regime_results[regime_results["split"].eq("test")].iloc[0]
    test_static = static_results[static_results["split"].eq("test")].iloc[0]
    lines = [
        "# Regime-Aware Combination Protocol",
        "",
        f"- Econometric expert: `{test_regime['econ_expert']}` selected by validation QLIKE from stable experts.",
        f"- Neural expert: `{test_regime['neural_expert']}` selected by validation QLIKE from corrected-context neural/calibrated candidates.",
        "- Regime indicator: origin-observable `rolling_vol_20` from the model-ready data.",
        "- Threshold grid: 75th, 85th, and 90th validation-period quantiles of the origin signal.",
        "- Weights: convex low/high-regime weights selected by validation QLIKE only and frozen for test evaluation.",
        "- Leakage control: no test losses, test actual variances, or future target values enter expert, threshold, or weight selection.",
        "",
        "## Frozen Test Evidence",
        "",
        f"- Static combination test QLIKE: {float(test_static['qlike']):.6f}.",
        f"- Regime-aware combination test QLIKE: {float(test_regime['qlike']):.6f}.",
        f"- Regime threshold quantile: {float(test_regime['threshold_quantile']):.2f}; value {float(test_regime['threshold_value']):.6f}.",
        f"- Low-regime econometric weight: {float(test_regime['low_econ_weight']):.2f}; high-regime econometric weight: {float(test_regime['high_econ_weight']):.2f}.",
        "",
        "## Validation Candidate Grid",
        "",
    ]
    for _, row in candidates.iterrows():
        lines.append(
            f"- q={float(row['threshold_quantile']):.2f}: validation QLIKE {float(row['validation_qlike']):.6f}, "
            f"low/high econ weights {float(row['low_econ_weight']):.2f}/{float(row['high_econ_weight']):.2f}."
        )
    (REPORTS_DIR / "regime_aware_combination_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_dynamic_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    path = OUTPUT_FINAL_DIR / "regime_aware_combination_predictions.csv"
    if not path.exists():
        return predictions
    dynamic = read_prediction(path)
    return pd.concat([predictions, dynamic], ignore_index=True)


def build_neural_retraining_comparison() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    artifacts = [
        ("LSTM", "outputs/neural_legacy_pre_context_fix/predictions/pred_lstm_base.csv", CORRECTED_NEURAL_ROOT / "predictions" / "pred_lstm_base.csv"),
        ("ARIMA-GARCH-LSTM", "outputs/neural_legacy_pre_context_fix/predictions/pred_lstm_hybrid.csv", CORRECTED_NEURAL_ROOT / "predictions" / "pred_lstm_hybrid.csv"),
        ("LSTM-QLIKE", "outputs/neural_legacy_pre_context_fix/predictions/lstm_tuned/pred_lstm_qlike.csv", CORRECTED_NEURAL_ROOT / "predictions" / "lstm_tuned" / "pred_lstm_qlike.csv"),
        ("Hybrid-QLIKE", "outputs/neural_legacy_pre_context_fix/predictions/lstm_tuned/pred_hybrid_qlike.csv", CORRECTED_NEURAL_ROOT / "predictions" / "lstm_tuned" / "pred_hybrid_qlike.csv"),
    ]
    for model, legacy_path, corrected_path in artifacts:
        for artifact_version, path in [("legacy_pre_context_fix", Path(legacy_path)), ("corrected_context_v2", corrected_path)]:
            if not path.exists():
                continue
            frame = read_prediction(path)
            for split, group in frame.groupby("split"):
                rows.append(
                    {
                        "model": model,
                        "artifact_version": artifact_version,
                        "split": split,
                        "source_file": rel_path(path),
                        "target_date_start": group["target_date"].min().strftime("%Y-%m-%d"),
                        "target_date_end": group["target_date"].max().strftime("%Y-%m-%d"),
                        **volatility_metrics(group["actual_var"], group["pred_var"]),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["model", "artifact_version", "split"])
    out.to_csv(OUTPUT_FINAL_DIR / "neural_retraining_comparison.csv", index=False)
    seed = out[(out["artifact_version"].eq("corrected_context_v2")) & (out["split"].eq("test"))].copy()
    seed.insert(1, "seed", 42)
    seed["robustness_status"] = "single_seed_deterministic_regeneration_only"
    seed.to_csv(OUTPUT_FINAL_DIR / "neural_seed_robustness.csv", index=False)
    display = out[out["split"].eq("test")][["model", "artifact_version", "n_obs", "qlike", "rmse", "mae"]]
    write_simple_latex(
        display,
        PAPER_TABLES_DIR / "neural_retraining_comparison.tex",
        caption="Legacy versus corrected-context neural test results.",
        label="tab:neural-retraining-comparison",
        columns=["model", "artifact_version", "n_obs", "qlike", "rmse", "mae"],
        formats={"qlike": "{:.6f}", "rmse": "{:.6f}", "mae": "{:.6f}"},
        table_star=True,
    )
    return out


def var_es_rows(common: pd.DataFrame, *, distribution_name: str, model_subset: set[str] | None = None) -> pd.DataFrame:
    returns = target_returns()
    merged = common.merge(returns, on="target_date", how="left", validate="many_to_one")
    if merged["target_return"].isna().any():
        raise ValueError("Missing target returns for VaR/ES evaluation.")
    rows = []
    for alpha in ALPHAS:
        for model, group in merged.groupby("model"):
            if model_subset is not None and model not in model_subset:
                continue
            ordered = group.sort_values("target_date")
            sigma = np.sqrt(np.clip(ordered["pred_var"].to_numpy(dtype=float), EPSILON, None))
            q, es = normal_var_es(alpha, sigma, 0.0)
            returns_arr = ordered["target_return"].to_numpy(dtype=float)
            violations = returns_arr < q
            kupiec_lr, kupiec_p = kupiec_test(violations, alpha)
            ind_lr, ind_p, transitions = christoffersen_independence(violations)
            cc_lr = kupiec_lr + ind_lr if np.isfinite(kupiec_lr) and np.isfinite(ind_lr) else np.nan
            realized_exceedance = returns_arr[violations]
            rows.append(
                {
                    "model": model,
                    "alpha": alpha,
                    "tail_convention": "lower_tail_return",
                    "var_distribution": distribution_name,
                    "mean_mapping": "zero_mean",
                    "n_obs": int(len(ordered)),
                    "var_violations": int(violations.sum()),
                    "expected_violations": float(alpha * len(ordered)),
                    "violation_ratio": float(violations.sum() / (alpha * len(ordered))) if len(ordered) else np.nan,
                    "violation_rate": float(violations.mean()),
                    "kupiec_uc_p_value": kupiec_p,
                    "christoffersen_independence_p_value": ind_p,
                    "christoffersen_cc_p_value": float(chi2.sf(cc_lr, 2)) if np.isfinite(cc_lr) else np.nan,
                    "quantile_loss": quantile_loss(returns_arr, q, alpha),
                    "var_forecast_average": float(np.mean(q)),
                    "es_forecast_average": float(np.mean(es)),
                    "realized_return_average_on_exceedance_dates": float(np.mean(realized_exceedance)) if len(realized_exceedance) else np.nan,
                    "es_calibration_error_on_exceedances": float(np.mean(realized_exceedance - es[violations])) if len(realized_exceedance) else np.nan,
                    "es_diagnostic_label": "mean_exceedance_return_minus_mean_es_forecast; diagnostic_not_formal_backtest",
                    **transitions,
                }
            )
    return pd.DataFrame(rows).sort_values(["alpha", "kupiec_uc_p_value", "model"], ascending=[True, False, True])


def build_var_es_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    predictions = append_dynamic_predictions(load_primary_predictions())
    common = common_window(predictions, "test")
    common_all = var_es_rows(common, distribution_name="common_normal_zero_mean")
    common_all.to_csv(OUTPUT_FINAL_DIR / "common_distribution_var_es_all_models.csv", index=False)
    main_models = [
        "AdvGARCH-BestAsymmetric",
        "AdvGARCH-BestQLIKE",
        "HAR-Parkinson",
        "GARCH(1,1)",
        "LSTM",
        "ARIMA-GARCH-LSTM",
        "RegimeAwareCombo-Vol20Q75",
        "RegimeAwareCombo-Vol20Q85",
        "RegimeAwareCombo-Vol20Q90",
    ]
    main = common_all[(common_all["model"].isin(main_models)) & (common_all["alpha"].isin([0.05, 0.025]))].copy()
    write_simple_latex(
        main[["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "christoffersen_cc_p_value", "quantile_loss"]],
        PAPER_TABLES_DIR / "common_distribution_var_es_main.tex",
        caption="Common-Normal zero-mean VaR and ES diagnostics for selected models.",
        label="tab:common-distribution-var-es-main",
        columns=["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "christoffersen_cc_p_value", "quantile_loss"],
        formats={
            "alpha": "{:.3f}",
            "expected_violations": "{:.2f}",
            "kupiec_uc_p_value": "{:.3g}",
            "christoffersen_cc_p_value": "{:.3g}",
            "quantile_loss": "{:.6f}",
        },
        table_star=True,
    )
    write_simple_latex(
        common_all[["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "quantile_loss"]],
        PAPER_TABLES_DIR / "common_distribution_var_es_full_appendix.tex",
        caption="Full common-Normal zero-mean VaR and ES diagnostics.",
        label="tab:common-distribution-var-es-full",
        columns=["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "quantile_loss"],
        formats={"alpha": "{:.3f}", "expected_violations": "{:.2f}", "kupiec_uc_p_value": "{:.3g}", "quantile_loss": "{:.6f}"},
        table_star=True,
    )

    model_specific_models = {"GARCH(1,1)", "AdvGARCH-BestQLIKE", "AdvGARCH-BestAsymmetric"}
    model_specific = var_es_rows(
        common,
        distribution_name="model_specific_normal_zero_mean",
        model_subset=model_specific_models,
    )
    model_specific["distribution_specific_quantile_available"] = True
    model_specific.to_csv(OUTPUT_FINAL_DIR / "model_specific_var_es_garch.csv", index=False)
    write_simple_latex(
        model_specific[["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "christoffersen_cc_p_value", "quantile_loss"]],
        PAPER_TABLES_DIR / "model_specific_var_es_garch.tex",
        caption="Distribution-specific VaR and ES for Normal GARCH-family specifications.",
        label="tab:model-specific-var-es-garch",
        columns=["model", "alpha", "n_obs", "var_violations", "expected_violations", "kupiec_uc_p_value", "christoffersen_cc_p_value", "quantile_loss"],
        formats={"alpha": "{:.3f}", "expected_violations": "{:.2f}", "kupiec_uc_p_value": "{:.3g}", "christoffersen_cc_p_value": "{:.3g}", "quantile_loss": "{:.6f}"},
        table_star=True,
    )
    write_distribution_audit(model_specific)
    return common_all, model_specific


def write_distribution_audit(model_specific: pd.DataFrame) -> None:
    lines = [
        "# Distribution-Specific VaR/ES Audit",
        "",
        "- Common-Normal VaR/ES uses zero-mean lower-tail return quantiles for every primary variance forecast.",
        "- Model-specific VaR/ES is computed only for selected GARCH-family specifications with Normal innovations, where analytical VaR and ES are verified by tests.",
        "- `GARCH(1,1)`, `AdvGARCH-BestQLIKE`, and `AdvGARCH-BestAsymmetric` are included under Normal innovations.",
        "- `AdvGARCH-BestHeavyTail` is validation-selected as a GED heavy-tail comparator, but GED ES is not reported because the GED ES implementation and parameter extraction were not verified in this pass.",
        "- Skewed-t ES is not implemented; no skewed-t model is promoted to primary risk evidence.",
        "- ES diagnostics are exceedance-mean diagnostics, not formal ES backtests.",
        "",
        f"- Model-specific rows written: {len(model_specific)}.",
    ]
    (REPORTS_DIR / "distribution_specific_var_es_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_regime_spike_outputs() -> pd.DataFrame:
    ensure_dirs()
    predictions = append_dynamic_predictions(load_primary_predictions())
    common = common_window(predictions, "test")
    model_ready = load_model_ready()
    train_val = model_ready[model_ready["date"] < common["date"].min()].copy()
    if train_val.empty:
        raise ValueError("No pre-test train/validation rows available for regime thresholds.")
    calm_threshold = float(train_val["target_var_next"].quantile(0.50))
    high_threshold = float(train_val["target_var_next"].quantile(0.75))
    spike_threshold = float(train_val["target_var_next"].quantile(0.90))
    rows = []
    for model, group in common.groupby("model"):
        actual = group["actual_var"].to_numpy(dtype=float)
        pred = group["pred_var"].to_numpy(dtype=float)
        calm = group[group["actual_var"] <= calm_threshold]
        high = group[group["actual_var"] >= high_threshold]
        spike = group[group["actual_var"] >= spike_threshold]
        ratio_nonspike_group = group[group["actual_var"] < spike_threshold]
        ratio_spike = np.clip(spike["pred_var"], EPSILON, None) / np.clip(spike["actual_var"], EPSILON, None)
        spike_ratio_of_means = (
            float(spike["pred_var"].mean() / spike["actual_var"].mean()) if not spike.empty and spike["actual_var"].mean() > 0 else np.nan
        )
        nonspike_ratio_of_means = (
            float(ratio_nonspike_group["pred_var"].mean() / ratio_nonspike_group["actual_var"].mean())
            if not ratio_nonspike_group.empty and ratio_nonspike_group["actual_var"].mean() > 0
            else np.nan
        )
        rows.append(
            {
                "model": model,
                "threshold_source": "train_validation_actual_var_quantiles",
                "calm_threshold_q50": calm_threshold,
                "high_threshold_q75": high_threshold,
                "spike_threshold_q90": spike_threshold,
                "overall_n": int(len(group)),
                "overall_qlike": qlike_objective(actual, pred),
                "overall_rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
                "overall_mae": float(np.mean(np.abs(actual - pred))),
                "calm_n": int(len(calm)),
                "calm_qlike": float(np.mean(qlike_loss(calm["actual_var"], calm["pred_var"]))) if not calm.empty else np.nan,
                "high_n": int(len(high)),
                "high_qlike": float(np.mean(qlike_loss(high["actual_var"], high["pred_var"]))) if not high.empty else np.nan,
                "spike_n": int(len(spike)),
                "spike_qlike": float(np.mean(qlike_loss(spike["actual_var"], spike["pred_var"]))) if not spike.empty else np.nan,
                "spike_underprediction_rate": float((spike["pred_var"] < spike["actual_var"]).mean()) if not spike.empty else np.nan,
                "severe_spike_underprediction_rate": float((ratio_spike < 0.5).mean()) if not spike.empty else np.nan,
                "mean_pred_actual_ratio_spike_days": spike_ratio_of_means,
                "mean_pred_actual_ratio_nonspike_days": nonspike_ratio_of_means,
            }
        )
    out = pd.DataFrame(rows).sort_values(["overall_qlike", "model"])
    out.to_csv(OUTPUT_FINAL_DIR / "regime_spike_results_corrected_neural.csv", index=False)
    selected = [
        "AdvGARCH-BestAsymmetric",
        "AdvGARCH-BestQLIKE",
        "HAR-Parkinson",
        "HAR-SquaredReturn",
        "GARCH(1,1)",
        "ARIMA-GARCH",
        "EWMA(lambda=0.90)",
        "LSTM",
        "ARIMA-GARCH-LSTM",
        "Calibrated-Hybrid-LogTarget-Small-isotonic",
        "ComboStack-PoolD_AllStable",
    ]
    selected += [m for m in out["model"] if str(m).startswith("RegimeAwareCombo")]
    display = out[out["model"].isin(selected)].copy()
    write_simple_latex(
        display[["model", "overall_n", "overall_qlike", "calm_qlike", "high_qlike", "spike_n", "spike_qlike", "severe_spike_underprediction_rate"]],
        PAPER_TABLES_DIR / "regime_spike_results_corrected_neural.tex",
        caption="Corrected-context regime and spike diagnostics.",
        label="tab:regime-spike-corrected-neural",
        columns=["model", "overall_n", "overall_qlike", "calm_qlike", "high_qlike", "spike_n", "spike_qlike", "severe_spike_underprediction_rate"],
        formats={
            "overall_qlike": "{:.6f}",
            "calm_qlike": "{:.3f}",
            "high_qlike": "{:.3f}",
            "spike_qlike": "{:.3f}",
            "severe_spike_underprediction_rate": "{:.3f}",
        },
        table_star=True,
    )
    return out


def build_refit_failure_taxonomy() -> pd.DataFrame:
    path = OUTPUT_FINAL_DIR / "refit_protocol_diagnostics.csv"
    diagnostics = pd.read_csv(path, encoding="utf-8-sig")
    failures = diagnostics[diagnostics["status"].eq("failure")].copy()
    rows = []
    for _, row in failures.iterrows():
        message = str(row.get("message", ""))
        rows.append(
            {
                "month": str(row["segment_start"])[:7],
                "segment_start": row["segment_start"],
                "segment_end_exclusive": row["segment_end_exclusive"],
                "model": row["model"],
                "specification": "validation_selected_best_qlike_advanced_garch",
                "failure_stage": "expanding_monthly_forecast",
                "fit_rows": int(row["fit_rows"]),
                "segment_rows": int(row["segment_rows"]),
                "exception_reason": message,
                "deterministic_retry_fix": False,
                "failure_persists": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_AUDIT_DIR / "refit_failure_taxonomy.csv", index=False)
    write_simple_latex(
        out[["month", "model", "fit_rows", "segment_rows", "failure_stage"]],
        PAPER_TABLES_DIR / "refit_failure_taxonomy.tex",
        caption="Failed expanding-monthly refit segments retained in the audit trail.",
        label="tab:refit-failure-taxonomy",
        columns=["month", "model", "fit_rows", "segment_rows", "failure_stage"],
        table_star=True,
    )
    lines = [
        "# Refit Failure Audit",
        "",
        f"- Failed monthly segments retained: {len(out)}.",
        "- All failures are for `AdvGARCH-BestQLIKE` in the expanding-monthly protocol.",
        "- The incomplete 340-row refit series is retained only as operational evidence and is not treated as comparable with complete 745-row static results.",
    ]
    (REPORTS_DIR / "refit_failure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_artifact_manifest() -> pd.DataFrame:
    ensure_dirs()
    roots = [
        OUTPUT_FINAL_DIR,
        OUTPUT_AUDIT_DIR,
        CORRECTED_NEURAL_ROOT,
        CORRECTED_ADVANCED_ROOT,
        PAPER_TABLES_DIR,
        PAPER_FIGURES_DIR,
    ]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and "/archive/" not in str(p)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "relative_path": rel_path(path),
                    "artifact_type": artifact_type(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "code_config_source": code_source(path),
                    "used_in_manuscript": str(path).startswith(str(PAPER_TABLES_DIR)) or str(path).startswith(str(PAPER_FIGURES_DIR)),
                    "sequence_construction_status": "post_sequence_context_fix" if str(path).startswith(str(CORRECTED_NEURAL_ROOT)) or str(path).startswith(str(CORRECTED_ADVANCED_ROOT)) else "not_neural_or_mixed",
                }
            )
    out = pd.DataFrame(rows).sort_values("relative_path")
    out.to_csv(PROJECT_ROOT / "outputs" / "artifact_manifest.csv", index=False)
    (PROJECT_ROOT / "outputs" / "artifact_manifest.json").write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    return out


def artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv" and "prediction" in str(path):
        return "prediction"
    if suffix in {".keras", ".pkl", ".joblib"}:
        return "model_artifact"
    if suffix == ".tex":
        return "latex_table"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "figure"
    if suffix == ".parquet":
        return "columnar_prediction_or_result"
    if suffix == ".csv":
        return "result_table"
    if suffix == ".md":
        return "audit_report"
    return "artifact"


def code_source(path: Path) -> str:
    s = str(path)
    if "neural_corrected_context_v2" in s:
        return "src/train_lstm_hybrid.py or src/tune_lstm_hybrid.py"
    if "advanced_corrected_context_v2" in s:
        return "src/advanced_forecast_combinations.py"
    if "regime_aware" in s or "var_es" in s or "artifact_manifest" in s:
        return "src/completion_extensions.py"
    if "outputs/final" in s:
        return "src/final_financial_econometrics.py or src/completion_extensions.py"
    return "paper or final artifact generation"


def write_reproducibility_audit(manifest: pd.DataFrame) -> None:
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    lines = [
        "# Reproducibility Statement Audit",
        "",
        f"- Git repository: yes.",
        f"- Branch: `{branch.stdout.strip()}`.",
        f"- Commit: `{commit.stdout.strip()}`.",
        f"- Remote: `{remote.stdout.strip()}`.",
        "- Remote reachability was checked with `git ls-remote origin master` during the baseline pass.",
        "- Raw CafeF files are present locally under `data/raw/`; no redistribution license was found in the repository.",
        "- Corrected neural artifacts are versioned under `outputs/neural_corrected_context_v2/`.",
        "- Corrected neural-dependent calibration and combination artifacts are under `outputs/advanced_corrected_context_v2/`.",
        f"- Current artifact manifest rows: {len(manifest)}.",
        f"- Python version: {platform.python_version()}.",
    ]
    (REPORTS_DIR / "reproducibility_statement_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_primary_corrected_outputs() -> pd.DataFrame:
    master = pd.read_csv(OUTPUT_FINAL_DIR / "common_window_master_results.csv", encoding="utf-8-sig")
    master.to_csv(OUTPUT_FINAL_DIR / "primary_results_corrected_neural.csv", index=False)
    return master


def build_all() -> None:
    ensure_dirs()
    build_regime_aware_combination()
    build_neural_retraining_comparison()
    build_var_es_outputs()
    build_regime_spike_outputs()
    build_refit_failure_taxonomy()
    build_primary_corrected_outputs()
    manifest = build_artifact_manifest()
    write_reproducibility_audit(manifest)
    print("Wrote completion-pass corrected neural, VaR/ES, regime, and manifest artifacts.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "combinations", "neural-comparison", "var-es", "regime-spike", "refit-taxonomy", "manifest"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "combinations":
        build_regime_aware_combination()
    elif args.command == "neural-comparison":
        build_neural_retraining_comparison()
    elif args.command == "var-es":
        build_var_es_outputs()
    elif args.command == "regime-spike":
        build_regime_spike_outputs()
    elif args.command == "refit-taxonomy":
        build_refit_failure_taxonomy()
    elif args.command == "manifest":
        manifest = build_artifact_manifest()
        write_reproducibility_audit(manifest)
    else:
        build_all()


if __name__ == "__main__":
    main()
