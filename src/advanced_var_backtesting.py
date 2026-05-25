#!/usr/bin/env python3
"""VaR backtesting from one-step-ahead volatility forecasts."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2, norm
except ImportError:  # pragma: no cover
    chi2 = None
    norm = None

from advanced_experiment_utils import (
    ADVANCED_AUDIT_DIR,
    ADVANCED_TABLES_DIR,
    DATA_DIR,
    EPSILON,
    compute_metrics,
    ensure_advanced_dirs,
    load_all_predictions,
    safe_write_csv,
    write_markdown,
)


def normal_quantile(alpha: float) -> float:
    if norm is not None:
        return float(norm.ppf(alpha))
    # Constants used only if scipy is unavailable.
    if abs(alpha - 0.05) < 1e-12:
        return -1.6448536269514729
    if abs(alpha - 0.01) < 1e-12:
        return -2.3263478740408408
    raise ValueError("Only 5% and 1% fallback quantiles are available without scipy.")


def chi2_sf(value: float, df: int) -> float:
    if not np.isfinite(value):
        return np.nan
    if chi2 is None:
        return np.nan
    return float(chi2.sf(value, df))


def bernoulli_loglik(count_success: int, count_failure: int, probability: float) -> float:
    probability = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return count_success * np.log(probability) + count_failure * np.log(1.0 - probability)


def kupiec_test(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    n_obs = int(len(violations))
    x = int(np.sum(violations))
    phat = x / n_obs if n_obs else np.nan
    if n_obs == 0:
        return np.nan, np.nan
    ll_null = bernoulli_loglik(x, n_obs - x, alpha)
    ll_alt = bernoulli_loglik(x, n_obs - x, phat)
    lr = float(max(0.0, -2.0 * (ll_null - ll_alt)))
    return lr, chi2_sf(lr, 1)


def christoffersen_independence(violations: np.ndarray) -> tuple[float, float, dict[str, int]]:
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan, {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev = v[:-1]
    curr = v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else np.nan
    pi01 = n01 / (n00 + n01) if (n00 + n01) else pi
    pi11 = n11 / (n10 + n11) if (n10 + n11) else pi
    ll_restricted = bernoulli_loglik(n01 + n11, n00 + n10, pi)
    ll_unrestricted = bernoulli_loglik(n01, n00, pi01) + bernoulli_loglik(n11, n10, pi11)
    lr = float(max(0.0, -2.0 * (ll_restricted - ll_unrestricted)))
    return lr, chi2_sf(lr, 1), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def var_backtest_row(df: pd.DataFrame, *, model: str, alpha: float) -> dict[str, Any]:
    z = normal_quantile(alpha)
    sigma = np.sqrt(np.clip(df["pred_var"].to_numpy(dtype=float), EPSILON, None))
    var_forecast = z * sigma
    returns = df["target_return"].to_numpy(dtype=float)
    violations = returns < var_forecast
    violation_count = int(np.sum(violations))
    n_obs = int(len(df))
    violation_rate = violation_count / n_obs if n_obs else np.nan
    kupiec_lr, kupiec_p = kupiec_test(violations, alpha)
    christ_lr, christ_p, transitions = christoffersen_independence(violations)
    cc_lr = kupiec_lr + christ_lr if np.isfinite(kupiec_lr) and np.isfinite(christ_lr) else np.nan
    cc_p = chi2_sf(cc_lr, 2)
    shortfall = var_forecast[violations] - returns[violations]
    return {
        "model": model,
        "alpha": alpha,
        "n_obs": n_obs,
        "violation_count": violation_count,
        "expected_violation_count": float(alpha * n_obs),
        "violation_rate": float(violation_rate),
        "kupiec_lr_uc": kupiec_lr,
        "kupiec_p_value": kupiec_p,
        "christoffersen_lr_ind": christ_lr,
        "christoffersen_ind_p_value": christ_p,
        "christoffersen_lr_cc": cc_lr,
        "christoffersen_cc_p_value": cc_p,
        "average_var": float(np.mean(var_forecast)),
        "average_shortfall_on_violation_days": float(np.mean(shortfall)) if len(shortfall) else np.nan,
        "average_return_on_violation_days": float(np.mean(returns[violations])) if violation_count else np.nan,
        **transitions,
    }


def target_returns() -> pd.DataFrame:
    clean_path = DATA_DIR / "vnindex_cafef_2010_2025_clean.csv"
    clean = pd.read_csv(clean_path, encoding="utf-8-sig")
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["log_return_pct"] = pd.to_numeric(clean["log_return_pct"], errors="coerce")
    if clean["date"].isna().any():
        raise ValueError(f"{clean_path} contains bad date values.")
    clean = clean.dropna(subset=["log_return_pct"]).copy()
    returns = clean[["date", "log_return_pct"]].rename(
        columns={"date": "target_date", "log_return_pct": "target_return"}
    )
    return returns


def choose_var_models(predictions: pd.DataFrame) -> list[str]:
    available = set(predictions["model"].unique())
    models: list[str] = ["GARCH(1,1)", "ARIMA-GARCH", "AdvGARCH-BestQLIKE"]
    validation_rows = []
    for model, group in predictions[predictions["split"] == "validation"].groupby("model"):
        validation_rows.append({"model": model, **compute_metrics(group)})
    metrics = pd.DataFrame(validation_rows)
    if not metrics.empty:
        refit = metrics[metrics["model"].astype(str).str.startswith("Refit-")]
        if not refit.empty:
            models.append(str(refit.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]["model"]))
        calibrated = metrics[metrics["model"].astype(str).str.startswith("Calibrated-")]
        if not calibrated.empty:
            models.append(str(calibrated.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]["model"]))
        combos = metrics[metrics["model"].astype(str).str.startswith("Combo")]
        if not combos.empty:
            models.append(str(combos.sort_values(["QLIKE", "RMSE", "model"]).iloc[0]["model"]))
    models.extend(["Hybrid-QLIKE", "LSTM-LogTarget"])
    seen = set()
    out = []
    for model in models:
        if model in available and model not in seen:
            out.append(model)
            seen.add(model)
    return out


def run_var_backtesting(*, force: bool = False) -> dict[float, pd.DataFrame]:
    ensure_advanced_dirs()
    predictions = load_all_predictions(include_advanced=True)
    returns = target_returns()
    selected_models = choose_var_models(predictions)
    test_predictions = predictions[(predictions["split"] == "test") & (predictions["model"].isin(selected_models))].copy()
    merged = test_predictions.merge(returns, on="target_date", how="left", validate="many_to_one")
    if merged["target_return"].isna().any():
        missing = merged.loc[merged["target_return"].isna(), ["model", "target_date"]].head(10)
        raise ValueError(f"Missing target returns for VaR backtesting:\n{missing.to_string(index=False)}")

    outputs: dict[float, pd.DataFrame] = {}
    for alpha, suffix in [(0.05, "5pct"), (0.01, "1pct")]:
        rows = [
            var_backtest_row(group.sort_values("target_date"), model=model, alpha=alpha)
            for model, group in merged.groupby("model")
        ]
        table = pd.DataFrame(rows).sort_values(["alpha", "kupiec_p_value", "model"], ascending=[True, False, True])
        safe_write_csv(table, ADVANCED_TABLES_DIR / f"table_var_backtesting_{suffix}.csv", force=force)
        outputs[alpha] = table
    write_var_report(outputs, force=force)
    return outputs


def write_var_report(outputs: dict[float, pd.DataFrame], *, force: bool) -> None:
    lines = [
        "# VaR Backtesting Report",
        "",
        "## VaR Construction",
        "",
        "- One-day Normal VaR is computed as `VaR_alpha_t = z_alpha * sqrt(pred_var_t)`.",
        "- Returns and volatility predictions are both in percent-return units.",
        "- Violations occur when the target-date log return is below the VaR forecast.",
        "",
        "## Coverage Results",
        "",
    ]
    for alpha, table in outputs.items():
        lines.append(f"### {int(alpha * 100)}% VaR")
        if table.empty:
            lines.append("- No models were available.")
        else:
            for _, row in table.sort_values("violation_rate").iterrows():
                lines.append(
                    f"- {row['model']}: violations {int(row['violation_count'])}/{int(row['n_obs'])} "
                    f"({float(row['violation_rate']):.3f}); Kupiec p={float(row['kupiec_p_value']):.6g}."
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Low-MAE models with severe variance underprediction should show excessive VaR violations if their volatility scale is too low.",
            "- Kupiec tests assess unconditional coverage; Christoffersen tests add violation independence and conditional coverage when transition counts are informative.",
            "",
            "## Output Tables",
            "",
            "- `outputs/advanced/tables/table_var_backtesting_5pct.csv`",
            "- `outputs/advanced/tables/table_var_backtesting_1pct.csv`",
        ]
    )
    write_markdown(ADVANCED_AUDIT_DIR / "var_backtesting_report.md", lines, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing advanced outputs.")
    args = parser.parse_args()
    run_var_backtesting(force=args.force)


if __name__ == "__main__":
    main()
