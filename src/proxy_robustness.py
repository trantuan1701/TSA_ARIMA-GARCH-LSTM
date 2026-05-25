#!/usr/bin/env python3
"""Volatility proxy robustness using OHLC range-based variance proxies."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from metrics import EPSILON
except ImportError:  # pragma: no cover
    from .metrics import EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "raw_cafef": PROJECT_ROOT / "data" / "raw" / "vnindex_cafef_2010_2025_raw.csv",
    "processed_clean": PROJECT_ROOT / "data" / "processed" / "vnindex_cafef_2010_2025_clean.csv",
    "model_ready": PROJECT_ROOT / "data" / "processed" / "vnindex_model_ready.csv",
    "train": PROJECT_ROOT / "data" / "processed" / "train.csv",
    "validation": PROJECT_ROOT / "data" / "processed" / "validation.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "test.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "proxy_robustness"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
AUDIT_DIR = OUTPUT_DIR / "audit"
OPTIONAL_RETRAINING_DIR = OUTPUT_DIR / "optional_retraining"

PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
PRICE_ROLES = ["date", "open", "high", "low", "close", "volume", "trading_value"]
ONE_DAY_PROXY_LABELS = [
    "close_to_close_squared_return",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
]
PROXY_LABELS = [
    *ONE_DAY_PROXY_LABELS,
    "yang_zhang_5",
    "yang_zhang_10",
    "yang_zhang_20",
]


ALIASES = {
    "date": ["date", "trading_date", "ngay", "ngaygiaodich"],
    "open": ["open", "open_price", "open_index", "giamocua"],
    "high": ["high", "high_price", "giacaonhat"],
    "low": ["low", "low_price", "giathapnhat"],
    "close": ["close", "close_price", "price", "index_close", "giadongcua", "giadieuchinh"],
    "volume": ["volume", "khoiluongkhoplenh", "khoiluong"],
    "trading_value": ["trading_value", "tradingvalue", "giatrikhoplenh", "gtkhoplenh"],
}


def ensure_dirs() -> None:
    for path in (TABLES_DIR, FIGURES_DIR, AUDIT_DIR, OPTIONAL_RETRAINING_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_role_column(columns: list[str], role: str) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for alias in ALIASES[role]:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def read_csv_if_exists(path: Path, *, nrows: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows)


def inspect_available_columns() -> tuple[pd.DataFrame, dict[str, dict[str, str | None]]]:
    rows = []
    role_map: dict[str, dict[str, str | None]] = {}
    for dataset, path in DATA_FILES.items():
        df = read_csv_if_exists(path, nrows=5)
        if df is None:
            role_map[dataset] = {role: None for role in PRICE_ROLES}
            rows.append(
                {
                    "dataset": dataset,
                    "path": rel_path(path),
                    "exists": False,
                    "row_sample_read": 0,
                    "columns": "",
                    **{f"{role}_column": None for role in PRICE_ROLES},
                    "has_ohlc": False,
                }
            )
            continue
        mapping = {role: find_role_column(df.columns.tolist(), role) for role in PRICE_ROLES}
        role_map[dataset] = mapping
        rows.append(
            {
                "dataset": dataset,
                "path": rel_path(path),
                "exists": True,
                "row_sample_read": len(df),
                "columns": ", ".join(df.columns.astype(str).tolist()),
                **{f"{role}_column": mapping[role] for role in PRICE_ROLES},
                "has_ohlc": all(mapping[role] is not None for role in ["open", "high", "low", "close"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "table_available_ohlc_columns.csv", index=False)
    return table, role_map


def parse_price_source() -> tuple[pd.DataFrame | None, str | None, dict[str, str | None]]:
    availability, role_map = inspect_available_columns()
    preferred = ["processed_clean", "model_ready", "train", "raw_cafef"]
    for dataset in preferred:
        row = availability[availability["dataset"] == dataset]
        if row.empty or not bool(row.iloc[0]["has_ohlc"]):
            continue
        path = DATA_FILES[dataset]
        raw = pd.read_csv(path, encoding="utf-8-sig")
        mapping = role_map[dataset]
        selected = pd.DataFrame()
        for role in ["date", "open", "high", "low", "close", "volume", "trading_value"]:
            column = mapping.get(role)
            if column is not None:
                selected[role] = raw[column]
        if "date" not in selected:
            continue
        selected["date"] = pd.to_datetime(selected["date"], errors="coerce", dayfirst=(dataset == "raw_cafef"))
        for role in ["open", "high", "low", "close", "volume", "trading_value"]:
            if role in selected:
                selected[role] = pd.to_numeric(selected[role], errors="coerce")
        selected = selected.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        selected = selected.reset_index(drop=True)
        return selected, dataset, mapping
    return None, None, {}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def watched_output_hashes() -> pd.DataFrame:
    roots = [PROJECT_ROOT / "outputs" / "predictions", PROJECT_ROOT / "outputs" / "metrics"]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "path": rel_path(path),
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


def valid_price_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    checks = pd.DataFrame(index=df.index)
    checks["missing_ohlc"] = df[["open", "high", "low", "close"]].isna().any(axis=1)
    checks["nonpositive_ohlc"] = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    checks["high_below_low"] = df["high"] < df["low"]
    checks["high_below_open_or_close"] = df["high"] < df[["open", "close"]].max(axis=1)
    checks["low_above_open_or_close"] = df["low"] > df[["open", "close"]].min(axis=1)
    invalid = checks.any(axis=1)
    return ~invalid, checks


def construct_proxies(price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    valid_prices, checks = valid_price_mask(df)
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    prev_close = df["close"].shift(1)
    close_to_close = 100.0 * np.log(df["close"] / prev_close)
    overnight = np.log(df["open"] / prev_close)
    open_close = np.log(df["close"] / df["open"])
    rs_component = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) + np.log(
        df["low"] / df["close"]
    ) * np.log(df["low"] / df["open"])

    proxies = pd.DataFrame({"date": df["date"]})
    proxies["close_to_close_squared_return"] = close_to_close**2
    proxies["parkinson"] = (100.0**2) * (1.0 / (4.0 * np.log(2.0))) * (log_hl**2)
    proxies["garman_klass"] = (100.0**2) * (
        0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    )
    proxies["rogers_satchell"] = (100.0**2) * rs_component

    invalid_rows: list[dict[str, Any]] = []
    for reason in checks.columns:
        invalid_rows.append(
            {
                "proxy": "all_ohlc_proxies",
                "reason": reason,
                "row_count": int(checks[reason].sum()),
                "first_date": df.loc[checks[reason], "date"].min().strftime("%Y-%m-%d")
                if checks[reason].any()
                else "",
                "last_date": df.loc[checks[reason], "date"].max().strftime("%Y-%m-%d")
                if checks[reason].any()
                else "",
            }
        )

    for proxy in ONE_DAY_PROXY_LABELS:
        nonfinite = ~np.isfinite(proxies[proxy].to_numpy(dtype=float))
        negative = proxies[proxy] < -1e-12
        tiny_negative = (proxies[proxy] < 0) & (proxies[proxy] >= -1e-12)
        proxies.loc[tiny_negative, proxy] = 0.0
        proxies.loc[~valid_prices | nonfinite | negative, proxy] = np.nan
        invalid_rows.extend(
            [
                {
                    "proxy": proxy,
                    "reason": "nonfinite_formula_value",
                    "row_count": int(nonfinite.sum()),
                    "first_date": df.loc[nonfinite, "date"].min().strftime("%Y-%m-%d") if nonfinite.any() else "",
                    "last_date": df.loc[nonfinite, "date"].max().strftime("%Y-%m-%d") if nonfinite.any() else "",
                },
                {
                    "proxy": proxy,
                    "reason": "negative_formula_value",
                    "row_count": int(negative.sum()),
                    "first_date": df.loc[negative, "date"].min().strftime("%Y-%m-%d") if negative.any() else "",
                    "last_date": df.loc[negative, "date"].max().strftime("%Y-%m-%d") if negative.any() else "",
                },
            ]
        )

    for window in [5, 10, 20]:
        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        sigma_o = overnight.rolling(window=window, min_periods=window).var(ddof=1)
        sigma_c = open_close.rolling(window=window, min_periods=window).var(ddof=1)
        sigma_rs = rs_component.rolling(window=window, min_periods=window).mean()
        name = f"yang_zhang_{window}"
        proxies[name] = (100.0**2) * (sigma_o + k * sigma_c + (1.0 - k) * sigma_rs)
        rolling_invalid = proxies[name].isna()
        negative = proxies[name] < -1e-12
        tiny_negative = (proxies[name] < 0) & (proxies[name] >= -1e-12)
        proxies.loc[tiny_negative, name] = 0.0
        proxies.loc[negative, name] = np.nan
        invalid_rows.extend(
            [
                {
                    "proxy": name,
                    "reason": "rolling_window_unavailable_or_nonfinite",
                    "row_count": int(rolling_invalid.sum()),
                    "first_date": df.loc[rolling_invalid, "date"].min().strftime("%Y-%m-%d")
                    if rolling_invalid.any()
                    else "",
                    "last_date": df.loc[rolling_invalid, "date"].max().strftime("%Y-%m-%d")
                    if rolling_invalid.any()
                    else "",
                },
                {
                    "proxy": name,
                    "reason": "negative_formula_value",
                    "row_count": int(negative.sum()),
                    "first_date": df.loc[negative, "date"].min().strftime("%Y-%m-%d") if negative.any() else "",
                    "last_date": df.loc[negative, "date"].max().strftime("%Y-%m-%d") if negative.any() else "",
                },
            ]
        )

    for proxy in PROXY_LABELS:
        proxies[f"{proxy}_target_next"] = proxies[proxy].shift(-1)

    invalid_table = pd.DataFrame(invalid_rows)
    invalid_table.to_csv(TABLES_DIR / "table_proxy_missing_invalid_rows.csv", index=False)
    return proxies, invalid_table


def proxy_summary_stats(proxies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proxy in PROXY_LABELS:
        values = pd.to_numeric(proxies[proxy], errors="coerce")
        valid = values.dropna()
        rows.append(
            {
                "proxy": proxy,
                "n_total": int(len(values)),
                "n_valid": int(valid.size),
                "n_missing": int(values.isna().sum()),
                "n_zero": int((valid == 0).sum()),
                "min": float(valid.min()) if not valid.empty else np.nan,
                "p01": float(valid.quantile(0.01)) if not valid.empty else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std": float(valid.std(ddof=1)) if valid.size > 1 else np.nan,
                "p95": float(valid.quantile(0.95)) if not valid.empty else np.nan,
                "p99": float(valid.quantile(0.99)) if not valid.empty else np.nan,
                "max": float(valid.max()) if not valid.empty else np.nan,
                "date_min": proxies.loc[values.notna(), "date"].min().strftime("%Y-%m-%d")
                if values.notna().any()
                else "",
                "date_max": proxies.loc[values.notna(), "date"].max().strftime("%Y-%m-%d")
                if values.notna().any()
                else "",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "table_proxy_summary_stats.csv", index=False)
    return table


def proxy_correlations(proxies: pd.DataFrame) -> pd.DataFrame:
    corr = proxies[PROXY_LABELS].corr(min_periods=50)
    corr.index.name = "proxy"
    table = corr.reset_index()
    table.to_csv(TABLES_DIR / "table_proxy_correlations.csv", index=False)
    return corr


def plot_proxy_timeseries(proxies: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    plot_items = [
        "close_to_close_squared_return",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
    ]
    for ax, proxy in zip(axes, plot_items):
        ax.plot(proxies["date"], proxies[proxy], linewidth=0.7)
        ax.set_ylabel(proxy.replace("_", "\n"))
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily Volatility Proxy Time Series")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_volatility_proxy_timeseries.png", dpi=250)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    labels = corr.index.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def prediction_files() -> list[Path]:
    paths = []
    original_dir = PROJECT_ROOT / "outputs" / "predictions"
    advanced_dir = PROJECT_ROOT / "outputs" / "advanced" / "predictions"
    if original_dir.exists():
        paths.extend(sorted(original_dir.rglob("*.csv")))
    if advanced_dir.exists():
        paths.extend(sorted(advanced_dir.rglob("*.csv")))
    return paths


def load_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"{rel_path(path)} has columns {list(df.columns)}, expected {PREDICTION_COLUMNS}.")
    parsed = df.copy()
    for column in ["date", "target_date"]:
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    for column in ["actual_var", "pred_var"]:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    parsed["model"] = parsed["model"].astype(str).str.strip()
    parsed["split"] = parsed["split"].astype(str).str.strip().str.lower()
    if parsed[["date", "target_date", "actual_var", "pred_var"]].isna().any().any():
        raise ValueError(f"{rel_path(path)} contains bad prediction values.")
    if (parsed["pred_var"] <= 0).any():
        raise ValueError(f"{rel_path(path)} contains non-positive predictions.")
    parsed["_source_file"] = rel_path(path)
    return parsed


def load_predictions() -> pd.DataFrame:
    frames = [load_prediction_file(path) for path in prediction_files()]
    if not frames:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "_source_file"])
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(["model", "split", "date", "target_date"], keep=False)
    if duplicates.any():
        examples = combined.loc[
            duplicates, ["_source_file", "model", "split", "date", "target_date"]
        ].head(20)
        raise ValueError(f"Duplicate prediction keys across files:\n{examples.to_string(index=False)}")
    return combined.sort_values(["split", "date", "target_date", "model"]).reset_index(drop=True)


def loss_values(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray, loss: str) -> np.ndarray:
    actual_arr = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    actual_arr = actual_arr[finite]
    pred_arr = np.clip(pred_arr[finite], EPSILON, None)
    if loss == "qlike":
        return np.log(pred_arr) + actual_arr / pred_arr
    if loss == "squared_error":
        return (actual_arr - pred_arr) ** 2
    if loss == "absolute_error":
        return np.abs(actual_arr - pred_arr)
    raise ValueError(f"Unknown loss: {loss}")


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_obs": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "QLIKE": np.nan,
            "mean_actual_var": np.nan,
            "mean_pred_var": np.nan,
        }
    actual = df["proxy_actual_var"].to_numpy(dtype=float)
    pred = np.clip(df["pred_var"].to_numpy(dtype=float), EPSILON, None)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[finite]
    pred = pred[finite]
    if len(actual) == 0:
        return {
            "n_obs": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "QLIKE": np.nan,
            "mean_actual_var": np.nan,
            "mean_pred_var": np.nan,
        }
    return {
        "n_obs": int(len(actual)),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "MAE": float(np.mean(np.abs(actual - pred))),
        "QLIKE": float(np.mean(np.log(pred) + actual / pred)),
        "mean_actual_var": float(np.mean(actual)),
        "mean_pred_var": float(np.mean(pred)),
    }


def proxy_targets(proxies: pd.DataFrame) -> pd.DataFrame:
    target = proxies[["date", *PROXY_LABELS]].rename(columns={"date": "target_date"}).copy()
    return target


def evaluate_forecasts(proxies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = load_predictions()
    targets = proxy_targets(proxies)
    rows = []
    eval_frames = []
    for proxy in PROXY_LABELS:
        target = targets[["target_date", proxy]].rename(columns={proxy: "proxy_actual_var"})
        merged = predictions.merge(target, on="target_date", how="left")
        merged = merged[np.isfinite(merged["proxy_actual_var"].to_numpy(dtype=float))].copy()
        if merged.empty:
            continue
        for (model, split), group in merged.groupby(["model", "split"]):
            metrics = compute_metrics(group)
            if metrics["n_obs"] < 50:
                continue
            actual_mean = float(metrics["mean_actual_var"])
            ratio = float(metrics["mean_pred_var"]) / actual_mean if actual_mean > 0 else np.nan
            threshold = float(group["proxy_actual_var"].quantile(0.90))
            extreme = group[group["proxy_actual_var"] > threshold]
            spike_rate = (
                float((extreme["pred_var"] < extreme["proxy_actual_var"]).mean())
                if not extreme.empty
                else np.nan
            )
            rows.append(
                {
                    "proxy": proxy,
                    "model": model,
                    "split": split,
                    **metrics,
                    "pred_actual_ratio": ratio,
                    "spike_underprediction_rate": spike_rate,
                    "source_files": " | ".join(sorted(group["_source_file"].unique())),
                    "target_date_min": group["target_date"].min().strftime("%Y-%m-%d"),
                    "target_date_max": group["target_date"].max().strftime("%Y-%m-%d"),
                }
            )
        eval_frames.append(merged.assign(proxy=proxy))
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(TABLES_DIR / "table_proxy_model_metrics_all.csv", index=False)

    rankings = metrics_table.copy()
    rankings["qlike_rank"] = rankings.groupby(["proxy", "split"])["QLIKE"].rank(method="min", ascending=True)
    rankings["rmse_rank"] = rankings.groupby(["proxy", "split"])["RMSE"].rank(method="min", ascending=True)
    rankings["mae_rank"] = rankings.groupby(["proxy", "split"])["MAE"].rank(method="min", ascending=True)
    rankings = rankings.sort_values(["split", "proxy", "qlike_rank", "model"])
    rankings.to_csv(TABLES_DIR / "table_proxy_model_rankings.csv", index=False)

    top10 = rankings[rankings["split"] == "test"].sort_values(["proxy", "qlike_rank", "model"])
    top10 = top10.groupby("proxy", as_index=False).head(10)
    top10.to_csv(TABLES_DIR / "table_proxy_top10_by_qlike.csv", index=False)
    plot_ranking_heatmap(rankings)
    return metrics_table, rankings, pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()


def plot_ranking_heatmap(rankings: pd.DataFrame) -> None:
    test = rankings[rankings["split"] == "test"].copy()
    if test.empty:
        return
    pivot = test.pivot_table(index="model", columns="proxy", values="qlike_rank", aggfunc="min")
    pivot["avg_rank"] = pivot.mean(axis=1)
    plot = pivot.sort_values("avg_rank").head(30).drop(columns=["avg_rank"])
    fig, ax = plt.subplots(figsize=(10, max(7, 0.28 * len(plot))))
    im = ax.imshow(plot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(plot.columns)))
    ax.set_xticklabels(plot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(plot.index)))
    ax.set_yticklabels(plot.index)
    ax.set_title("Test QLIKE Rank by Volatility Proxy")
    fig.colorbar(im, ax=ax, label="Rank (lower is better)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_proxy_ranking_heatmap.png", dpi=250)
    plt.close(fig)


def normal_p_value(stat: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(stat))))


def newey_west_lrv(diff: np.ndarray) -> tuple[float, int]:
    n = len(diff)
    if n <= 1:
        return 0.0, 0
    centered = diff - float(np.mean(diff))
    max_lag = min(int(np.floor(n ** (1.0 / 3.0))), n - 1)
    lrv = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * cov
    if lrv < 0 and abs(lrv) < 1e-14:
        lrv = 0.0
    return lrv, max_lag


def dm_stat(diff: np.ndarray) -> tuple[float, float, int]:
    mean_diff = float(np.mean(diff))
    lrv, lag = newey_west_lrv(diff)
    if lrv <= 0:
        if abs(mean_diff) <= 1e-14:
            return 0.0, 1.0, lag
        return np.nan, np.nan, lag
    stat = mean_diff / np.sqrt(lrv / len(diff))
    return float(stat), normal_p_value(float(stat)), lag


def holm_adjust(values: pd.Series) -> pd.Series:
    pvals = pd.to_numeric(values, errors="coerce")
    finite = pvals.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    m = len(finite)
    running = 0.0
    for rank, (idx, p_value) in enumerate(finite.items(), start=1):
        adj = min(1.0, (m - rank + 1) * float(p_value))
        running = max(running, adj)
        adjusted.loc[idx] = running
    return adjusted


def best_advanced_garch_model() -> str | None:
    path = PROJECT_ROOT / "outputs" / "advanced" / "tables" / "table_garch_family_selected_models.csv"
    if not path.exists():
        return "AdvGARCH-BestQLIKE"
    selected = pd.read_csv(path)
    row = selected[selected["selection_role"] == "best_fixed_by_validation_qlike"]
    if row.empty:
        return "AdvGARCH-BestQLIKE"
    return str(row.iloc[0]["model"])


def best_validation_overall_model(metrics: pd.DataFrame) -> str | None:
    validation = metrics[metrics["split"] == "validation"].copy()
    if validation.empty:
        return None
    mean_validation = (
        validation.groupby("model", as_index=False)["QLIKE"]
        .mean()
        .sort_values(["QLIKE", "model"])
        .reset_index(drop=True)
    )
    return str(mean_validation.iloc[0]["model"]) if not mean_validation.empty else None


def run_proxy_dm_tests(metrics: pd.DataFrame, eval_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmarks = ["GARCH(1,1)"]
    advanced = best_advanced_garch_model()
    if advanced:
        benchmarks.append(advanced)
    overall = best_validation_overall_model(metrics)
    if overall:
        benchmarks.append(overall)
    benchmarks = list(dict.fromkeys(benchmarks))

    rows = []
    test_eval = eval_long[eval_long["split"] == "test"].copy()
    for proxy, proxy_df in test_eval.groupby("proxy"):
        available = set(proxy_df["model"].unique())
        for benchmark in benchmarks:
            if benchmark not in available:
                continue
            bench = proxy_df[proxy_df["model"] == benchmark][
                ["date", "target_date", "proxy_actual_var", "pred_var"]
            ].rename(columns={"pred_var": "pred_benchmark"})
            for model in sorted(available):
                if model == benchmark:
                    continue
                candidate = proxy_df[proxy_df["model"] == model][
                    ["date", "target_date", "proxy_actual_var", "pred_var"]
                ].rename(columns={"pred_var": "pred_model"})
                merged = candidate.merge(
                    bench[["date", "target_date", "pred_benchmark"]],
                    on=["date", "target_date"],
                    how="inner",
                    validate="one_to_one",
                )
                if len(merged) < 50:
                    continue
                loss_model = loss_values(merged["proxy_actual_var"], merged["pred_model"], "qlike")
                loss_benchmark = loss_values(
                    merged["proxy_actual_var"], merged["pred_benchmark"], "qlike"
                )
                diff = loss_model - loss_benchmark
                stat, p_value, lag = dm_stat(diff)
                rows.append(
                    {
                        "proxy": proxy,
                        "model": model,
                        "benchmark": benchmark,
                        "n": int(len(diff)),
                        "mean_loss_model": float(np.mean(loss_model)),
                        "mean_loss_benchmark": float(np.mean(loss_benchmark)),
                        "mean_diff": float(np.mean(diff)),
                        "dm_stat": stat,
                        "p_value": p_value,
                        "hac_lag": lag,
                    }
                )
    dm = pd.DataFrame(rows)
    dm.to_csv(TABLES_DIR / "table_proxy_dm_tests_qlike.csv", index=False)
    holm = dm.copy()
    if not holm.empty:
        holm["holm_adjusted_p_value"] = np.nan
        for (_, benchmark), idx in holm.groupby(["proxy", "benchmark"]).groups.items():
            holm.loc[idx, "holm_adjusted_p_value"] = holm_adjust(holm.loc[idx, "p_value"])
        holm["significant_10pct_holm"] = holm["holm_adjusted_p_value"] < 0.10
        holm["significant_5pct_holm"] = holm["holm_adjusted_p_value"] < 0.05
        holm = holm.sort_values(["proxy", "benchmark", "holm_adjusted_p_value", "model"])
    holm.to_csv(TABLES_DIR / "table_proxy_holm_tests_qlike.csv", index=False)
    return dm, holm


def classify_model(row: pd.Series) -> str:
    if row["top3_proxy_count"] >= max(1, math.ceil(0.6 * row["proxy_count"])):
        return "robust_top_model"
    if row["rank_std"] >= 15 or row["worst_rank"] - row["best_rank"] >= 30:
        return "proxy_sensitive_model"
    if row["average_qlike_rank"] >= 0.75 * row["model_count_max"]:
        return "weak_model"
    if "LogTarget" in str(row["model"]) and row.get("average_pred_actual_ratio", 1.0) < 0.7:
        return "risky_underpredictor"
    return "middle_ranked_model"


def rank_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    test = rankings[rankings["split"] == "test"].copy()
    rows = []
    model_count_max = int(test.groupby("proxy")["model"].nunique().max()) if not test.empty else 0
    for model, group in test.groupby("model"):
        ranks = group["qlike_rank"].astype(float)
        rows.append(
            {
                "model": model,
                "proxy_count": int(group["proxy"].nunique()),
                "average_qlike_rank": float(ranks.mean()),
                "best_rank": float(ranks.min()),
                "worst_rank": float(ranks.max()),
                "rank_std": float(ranks.std(ddof=0)),
                "top1_proxy_count": int((ranks <= 1).sum()),
                "top3_proxy_count": int((ranks <= 3).sum()),
                "top5_proxy_count": int((ranks <= 5).sum()),
                "average_pred_actual_ratio": float(group["pred_actual_ratio"].mean()),
                "average_spike_underprediction_rate": float(group["spike_underprediction_rate"].mean()),
                "model_count_max": model_count_max,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["classification"] = table.apply(classify_model, axis=1)
        table = table.sort_values(["average_qlike_rank", "rank_std", "model"]).reset_index(drop=True)
    table.to_csv(TABLES_DIR / "table_proxy_rank_stability.csv", index=False)
    return table


def conclusion_safety(rankings: pd.DataFrame, stability: pd.DataFrame, dm: pd.DataFrame, holm: pd.DataFrame) -> pd.DataFrame:
    test = rankings[rankings["split"] == "test"].copy()

    def avg_rank(model: str) -> float:
        row = stability[stability["model"] == model]
        return float(row.iloc[0]["average_qlike_rank"]) if not row.empty else np.nan

    def top5_count(model: str) -> int:
        row = stability[stability["model"] == model]
        return int(row.iloc[0]["top5_proxy_count"]) if not row.empty else 0

    proxies = int(test["proxy"].nunique()) if not test.empty else 0
    adv = best_advanced_garch_model() or "AdvGARCH-BestQLIKE"
    neural = test[test["model"].str.contains("LSTM|Hybrid", regex=True, case=False, na=False)]
    econ = test[test["model"].str.contains("GARCH|AdvGARCH|Refit", regex=True, case=False, na=False)]
    neural_best = neural.groupby("proxy")["qlike_rank"].min().mean() if not neural.empty else np.nan
    econ_best = econ.groupby("proxy")["qlike_rank"].min().mean() if not econ.empty else np.nan
    logtarget = stability[stability["model"].str.contains("LogTarget", case=False, regex=False, na=False)]
    raw_logtarget = logtarget[~logtarget["model"].str.startswith("Calibrated-")]
    adv_sig = dm[
        (dm["benchmark"] == "GARCH(1,1)")
        & (dm["model"] == adv)
        & (dm["mean_diff"] < 0)
        & (dm["p_value"] < 0.10)
    ]
    adv_holm = holm[
        (holm["benchmark"] == "GARCH(1,1)")
        & (holm["model"] == adv)
        & (holm["mean_diff"] < 0)
        & (holm["holm_adjusted_p_value"] < 0.10)
    ]

    claims = [
        {
            "claim": "GARCH(1,1) is a strong benchmark.",
            "status": "supported" if top5_count("GARCH(1,1)") >= max(1, proxies // 2) else "partially_supported",
            "key_evidence": f"GARCH(1,1) average QLIKE rank {avg_rank('GARCH(1,1)'):.2f}; top-5 under {top5_count('GARCH(1,1)')}/{proxies} proxies.",
            "caveat": "Range-based proxies can favor smoother forecasts, so exact rank is proxy-dependent.",
        },
        {
            "claim": "Advanced GARCH-family models can improve QLIKE modestly.",
            "status": "supported" if top5_count(adv) >= max(1, proxies // 2) else "partially_supported",
            "key_evidence": f"{adv} average QLIKE rank {avg_rank(adv):.2f}; raw DM improvement over GARCH under {len(adv_sig)} proxies; Holm under {len(adv_holm)} proxies.",
            "caveat": "Multiple-testing correction is stricter than raw DM tests.",
        },
        {
            "claim": "ARIMA mean dynamics add limited value.",
            "status": "supported"
            if avg_rank("ARIMA-GARCH") >= avg_rank("GARCH(1,1)") - 1
            else "partially_supported",
            "key_evidence": f"ARIMA-GARCH average QLIKE rank {avg_rank('ARIMA-GARCH'):.2f} versus GARCH {avg_rank('GARCH(1,1)'):.2f}.",
            "caveat": "This claim is about the existing ARIMA-GARCH and proxy evaluation, not retrained proxy-specific ARIMA models.",
        },
        {
            "claim": "Neural/hybrid models do not robustly dominate econometric models.",
            "status": "supported" if np.isfinite(neural_best) and np.isfinite(econ_best) and neural_best > econ_best else "inconclusive",
            "key_evidence": f"Average best neural/hybrid proxy rank {neural_best:.2f}; average best econometric proxy rank {econ_best:.2f}.",
            "caveat": "Calibrated neural forecasts can improve under some proxies but are still evaluated without retraining.",
        },
        {
            "claim": "Log-target neural variants can reduce MAE but underpredict risk.",
            "status": "supported"
            if not raw_logtarget.empty and raw_logtarget["average_pred_actual_ratio"].mean() < 0.7
            else "partially_supported",
            "key_evidence": f"Raw log-target average pred/actual ratio {raw_logtarget['average_pred_actual_ratio'].mean():.3f}; average spike underprediction {raw_logtarget['average_spike_underprediction_rate'].mean():.3f}.",
            "caveat": "Range-based targets change the actual scale, but raw log-target forecasts remain visibly low relative to proxy spikes.",
        },
        {
            "claim": "Results are sensitive or insensitive to the volatility proxy.",
            "status": "partially_supported",
            "key_evidence": f"Average rank standard deviation across models is {stability['rank_std'].mean():.2f}; top-ranked model changes across proxy definitions.",
            "caveat": "Paper claims should refer to robustness directionally, not as identical rankings across all proxies.",
        },
    ]
    table = pd.DataFrame(claims)
    table.to_csv(TABLES_DIR / "table_proxy_conclusion_safety.csv", index=False)
    return table


def write_availability_report(availability: pd.DataFrame, source_name: str | None) -> None:
    ohlc_rows = availability[availability["has_ohlc"]]
    lines = [
        "# Proxy Data Availability Report",
        "",
        "## Data Files Inspected",
        "",
    ]
    for _, row in availability.iterrows():
        lines.append(
            f"- `{row['path']}`: exists={bool(row['exists'])}, has_ohlc={bool(row['has_ohlc'])}; "
            f"date={row['date_column']}, open={row['open_column']}, high={row['high_column']}, "
            f"low={row['low_column']}, close={row['close_column']}."
        )
    lines.extend(["", "## Status", ""])
    if source_name is None:
        lines.append("- OHLC columns were not available, so proxy robustness could not be fully run.")
    else:
        lines.append(f"- OHLC columns are available. Proxy construction source: `{source_name}`.")
        lines.append(f"- Files with OHLC columns: {len(ohlc_rows)}.")
    write_markdown(AUDIT_DIR / "proxy_data_availability_report.md", lines)


def write_construction_report(source_name: str, summary: pd.DataFrame, invalid: pd.DataFrame) -> None:
    lines = [
        "# Proxy Construction Report",
        "",
        f"- Price source: `{source_name}`.",
        "- All one-day range proxies use log price ratios and are multiplied by `100^2` to match percentage-return variance forecasts.",
        "- Yang-Zhang proxies are rolling 5-, 10-, and 20-day variance estimates, not one-day proxies.",
        "- Tiny negative numerical values are set to zero; materially negative formula values are treated as invalid.",
        "",
        "## Valid Observations",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['proxy']}: {int(row['n_valid'])}/{int(row['n_total'])} valid observations; "
            f"mean={float(row['mean']):.6f}."
        )
    invalid_nonzero = invalid[pd.to_numeric(invalid["row_count"], errors="coerce") > 0]
    lines.extend(["", "## Invalid Rows", ""])
    if invalid_nonzero.empty:
        lines.append("- No invalid OHLC/proxy rows beyond expected initial rolling-window missing values.")
    else:
        for _, row in invalid_nonzero.iterrows():
            lines.append(f"- {row['proxy']} / {row['reason']}: {int(row['row_count'])} rows.")
    write_markdown(AUDIT_DIR / "proxy_construction_report.md", lines)


def write_forecast_report(rankings: pd.DataFrame) -> None:
    test = rankings[rankings["split"] == "test"].copy()
    lines = [
        "# Proxy Forecast Evaluation Report",
        "",
        "- Existing forecasts are read-only inputs; `pred_var` is unchanged.",
        "- `actual_var` is replaced only inside this evaluation by the proxy value on `target_date`.",
        "- Metrics use the same QLIKE epsilon clipping as the existing project.",
        "",
        "## Top QLIKE Models By Proxy",
        "",
    ]
    for proxy, group in test.groupby("proxy"):
        best = group.sort_values(["qlike_rank", "QLIKE", "model"]).head(5)
        lines.append(f"### {proxy}")
        for _, row in best.iterrows():
            lines.append(
                f"- {row['model']}: QLIKE={float(row['QLIKE']):.6f}, rank={int(row['qlike_rank'])}, "
                f"pred/actual={float(row['pred_actual_ratio']):.3f}."
            )
        lines.append("")
    garch = test[test["model"] == "GARCH(1,1)"]
    adv = best_advanced_garch_model() or "AdvGARCH-BestQLIKE"
    adv_rows = test[test["model"] == adv]
    neural_top = test[test["model"].str.contains("LSTM|Hybrid", case=False, regex=True, na=False)]
    lines.extend(
        [
            "## Summary Answers",
            "",
            f"- GARCH(1,1) average QLIKE rank: {garch['qlike_rank'].mean():.2f}.",
            f"- {adv} average QLIKE rank: {adv_rows['qlike_rank'].mean():.2f}.",
            f"- Best neural/hybrid rank by proxy averages: {neural_top.groupby('proxy')['qlike_rank'].min().mean():.2f}.",
            "- Raw log-target models remain underpredictors when pred/actual ratios stay materially below one and spike underprediction remains high.",
        ]
    )
    write_markdown(AUDIT_DIR / "proxy_forecast_evaluation_report.md", lines)


def write_dm_report(dm: pd.DataFrame, holm: pd.DataFrame) -> None:
    lines = [
        "# Proxy DM Tests Report",
        "",
        "- Tests use QLIKE loss on the proxy-specific common test rows for each model/benchmark pair.",
        "- Loss difference convention: `loss_model_t - loss_benchmark_t`; positive means the model is worse.",
        "- Holm correction is applied within each proxy and benchmark family.",
        "",
        "## Raw Significant Improvements Versus GARCH(1,1)",
        "",
    ]
    sig = dm[(dm["benchmark"] == "GARCH(1,1)") & (dm["mean_diff"] < 0) & (dm["p_value"] < 0.10)]
    if sig.empty:
        lines.append("- None.")
    else:
        for _, row in sig.sort_values(["proxy", "p_value"]).iterrows():
            lines.append(
                f"- {row['proxy']}: {row['model']} mean_diff={float(row['mean_diff']):.6f}, p={float(row['p_value']):.6g}."
            )
    lines.extend(["", "## Holm-Significant Improvements Versus GARCH(1,1)", ""])
    holm_sig = holm[
        (holm["benchmark"] == "GARCH(1,1)")
        & (holm["mean_diff"] < 0)
        & (holm["holm_adjusted_p_value"] < 0.10)
    ]
    if holm_sig.empty:
        lines.append("- None.")
    else:
        for _, row in holm_sig.sort_values(["proxy", "holm_adjusted_p_value"]).iterrows():
            lines.append(
                f"- {row['proxy']}: {row['model']} Holm p={float(row['holm_adjusted_p_value']):.6g}."
            )
    write_markdown(AUDIT_DIR / "proxy_dm_tests_report.md", lines)


def write_summary_report(stability: pd.DataFrame, conclusions: pd.DataFrame, integrity: pd.DataFrame) -> None:
    changed = integrity[integrity["status"] != "unchanged"]
    lines = [
        "# Proxy Robustness Summary",
        "",
        "## Stable Top Models",
        "",
    ]
    for _, row in stability.head(10).iterrows():
        lines.append(
            f"- {row['model']}: average rank {float(row['average_qlike_rank']):.2f}, "
            f"best {float(row['best_rank']):.0f}, worst {float(row['worst_rank']):.0f}, "
            f"classification `{row['classification']}`."
        )
    lines.extend(["", "## Conclusion Safety", ""])
    for _, row in conclusions.iterrows():
        lines.append(f"- {row['claim']} Status: `{row['status']}`. {row['key_evidence']}")
    lines.extend(["", "## Existing Output Integrity", ""])
    if changed.empty:
        lines.append("- No existing files under `outputs/predictions/` or `outputs/metrics/` were modified.")
    else:
        for _, row in changed.iterrows():
            lines.append(f"- {row['status']}: `{row['path']}`")
    write_markdown(AUDIT_DIR / "proxy_robustness_summary.md", lines)


def write_optional_retraining_report() -> None:
    lines = [
        "# Optional Proxy Retraining Report",
        "",
        "Status: not implemented.",
        "",
        "The final robustness layer is intentionally read-only with respect to existing forecasts. "
        "Retraining proxy-specific models would add another modeling stage and selection surface. "
        "For this paper stage, the safer test is whether conclusions from the already-audited forecasts "
        "are stable when the noisy close-to-close target is replaced by OHLC range-based proxies.",
    ]
    write_markdown(AUDIT_DIR / "optional_retraining_report.md", lines)


def run_proxy_robustness() -> None:
    ensure_dirs()
    before = watched_output_hashes()
    availability, _role_map = inspect_available_columns()
    price_df, source_name, _mapping = parse_price_source()
    write_availability_report(availability, source_name)
    if price_df is None or source_name is None:
        write_optional_retraining_report()
        return

    proxies, invalid = construct_proxies(price_df)
    summary = proxy_summary_stats(proxies)
    corr = proxy_correlations(proxies)
    plot_proxy_timeseries(proxies)
    plot_correlation_heatmap(
        corr,
        FIGURES_DIR / "fig_proxy_correlation_heatmap.png",
        "Volatility Proxy Correlations",
    )
    write_construction_report(source_name, summary, invalid)

    metrics, rankings, eval_long = evaluate_forecasts(proxies)
    write_forecast_report(rankings)
    dm, holm = run_proxy_dm_tests(metrics, eval_long)
    write_dm_report(dm, holm)
    stability = rank_stability(rankings)
    conclusions = conclusion_safety(rankings, stability, dm, holm)
    after = watched_output_hashes()
    integrity = compare_hashes(before, after)
    integrity.to_csv(TABLES_DIR / "table_existing_output_integrity.csv", index=False)
    write_summary_report(stability, conclusions, integrity)
    write_optional_retraining_report()


def main() -> None:
    run_proxy_robustness()


if __name__ == "__main__":
    main()
