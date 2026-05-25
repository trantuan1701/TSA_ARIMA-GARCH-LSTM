#!/usr/bin/env python3
"""Read-only audits for the VN-Index volatility experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Works when imported as ``src.audit_experiment``.
    from .metrics import EPSILON
except ImportError:  # Works when executed as ``python src/audit_experiment.py``.
    from metrics import EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

SPLIT_NAMES = ("train", "validation", "test")
PREDICTION_COLUMNS = ["date", "target_date", "actual_var", "pred_var", "model", "split"]
EXPECTED_COMMON_COUNTS = {"validation": 731, "test": 726}
LOWER_IS_BETTER_METRICS = {"mse", "rmse", "mae", "qlike"}


@dataclass(frozen=True)
class AuditResult:
    title: str
    errors: list[str]
    warnings: list[str]
    lines: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_splits(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        path = project_root / "data" / "processed" / f"{split}.csv"
        df = read_csv(path)
        splits[split] = parse_dates(df, path)
    return splits


def load_model_ready(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "vnindex_model_ready.csv"
    return parse_dates(read_csv(path), path)


def parse_dates(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    parsed = df.copy()
    for column in ("date", "target_date"):
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
            if parsed[column].isna().any():
                bad = parsed.index[parsed[column].isna()].tolist()[:10]
                raise ValueError(f"{source_path.relative_to(PROJECT_ROOT)} has bad {column} rows: {bad}")
    return parsed


def prediction_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    pred_dir = project_root / "outputs" / "predictions"
    paths = sorted(pred_dir.glob("pred_*.csv"))
    paths.extend(sorted((pred_dir / "lstm_tuned").glob("pred_*.csv")))
    return paths


def load_prediction_file(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    return parse_dates(df, path)


def load_all_predictions(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in prediction_files(project_root):
        df = load_prediction_file(path)
        df["_source_file"] = str(path.relative_to(project_root))
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No prediction CSV files found under outputs/predictions.")
    return pd.concat(frames, ignore_index=True)


def load_expected_split_counts(project_root: Path = PROJECT_ROOT) -> dict[str, int]:
    path = project_root / "outputs" / "metrics" / "data_summary.csv"
    if not path.exists():
        return {}
    summary = read_csv(path)
    rows = summary[(summary["dataset"].isin(SPLIT_NAMES)) & (summary["metric"] == "row_count")]
    return {str(row["dataset"]): int(row["value"]) for _, row in rows.iterrows()}


def metric_is_improvement(metric: str, candidate: float, baseline: float) -> bool:
    metric_key = metric.lower()
    if metric_key not in LOWER_IS_BETTER_METRICS:
        raise ValueError(f"Unknown lower-is-better metric: {metric}")
    return float(candidate) < float(baseline)


def validate_split_integrity(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    splits = load_splits(project_root)
    expected_counts = load_expected_split_counts(project_root)

    for split, df in splits.items():
        required = {"date", "target_date", "target_var_next", "squared_return"}
        missing = sorted(required - set(df.columns))
        if missing:
            errors.append(f"{split}.csv is missing required columns: {missing}")
            continue
        if df.empty:
            errors.append(f"{split}.csv is empty")
            continue
        if not df["date"].is_monotonic_increasing:
            errors.append(f"{split}.csv dates are not chronological")
        duplicate_dates = int(df["date"].duplicated().sum())
        if duplicate_dates:
            errors.append(f"{split}.csv has {duplicate_dates} duplicate date rows")
        if split in expected_counts and len(df) != expected_counts[split]:
            errors.append(
                f"{split}.csv row count changed: observed {len(df)}, "
                f"expected {expected_counts[split]} from outputs/metrics/data_summary.csv"
            )

    ordered = [splits[name] for name in SPLIT_NAMES]
    for left_name, right_name, left_df, right_df in zip(SPLIT_NAMES, SPLIT_NAMES[1:], ordered, ordered[1:]):
        if not left_df["date"].max() < right_df["date"].min():
            errors.append(f"{left_name} and {right_name} date ranges overlap or are out of order")
        if set(left_df["date"]).intersection(set(right_df["date"])):
            errors.append(f"{left_name} and {right_name} share date rows")

    return errors


def computed_prepared_data(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    clean_path = project_root / "data" / "processed" / "vnindex_cafef_2010_2025_clean.csv"
    clean = read_csv(clean_path)
    if "date" not in clean.columns or "close" not in clean.columns:
        raise ValueError("Clean data must contain date and close columns.")
    prepared = clean.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(prepared["close"], errors="coerce")
    prepared["computed_log_return_pct"] = 100.0 * np.log(close / close.shift(1))
    prepared["computed_squared_return"] = prepared["computed_log_return_pct"] ** 2
    prepared["computed_target_var_next"] = prepared["computed_squared_return"].shift(-1)
    prepared["computed_target_date"] = prepared["date"].shift(-1)
    return prepared


def validate_target_construction(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    model_ready = load_model_ready(project_root)
    prepared = computed_prepared_data(project_root)

    required = {"date", "target_date", "target_var_next"}
    missing = sorted(required - set(model_ready.columns))
    if missing:
        return [f"vnindex_model_ready.csv is missing required columns: {missing}"]

    merged = model_ready[["date", "target_date", "target_var_next"]].merge(
        prepared[["date", "computed_target_date", "computed_target_var_next"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    missing_matches = merged["computed_target_date"].isna()
    if missing_matches.any():
        examples = merged.loc[missing_matches, "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
        errors.append(f"Model-ready rows missing from clean prepared data: {examples}")

    target_dates_match = merged["target_date"].eq(merged["computed_target_date"])
    if not target_dates_match.all():
        examples = merged.loc[~target_dates_match, ["date", "target_date", "computed_target_date"]].head(5)
        errors.append("target_date does not match next available trading date:\n" + examples.to_string(index=False))

    observed = pd.to_numeric(merged["target_var_next"], errors="coerce").to_numpy(dtype=float)
    expected = merged["computed_target_var_next"].to_numpy(dtype=float)
    target_values_match = np.isclose(observed, expected, rtol=1e-10, atol=1e-12)
    if not bool(target_values_match.all()):
        bad = merged.loc[~target_values_match, ["date", "target_var_next", "computed_target_var_next"]].head(5)
        errors.append("target_var_next does not equal next-row squared return:\n" + bad.to_string(index=False))

    if not (model_ready["target_date"] > model_ready["date"]).all():
        bad = model_ready.loc[~(model_ready["target_date"] > model_ready["date"]), ["date", "target_date"]].head(5)
        errors.append("target_date must be strictly after date:\n" + bad.to_string(index=False))

    unavailable_target_dates = set(prepared.loc[prepared["computed_target_var_next"].isna(), "date"])
    leaked_unavailable = sorted(unavailable_target_dates.intersection(set(model_ready["date"])))
    if leaked_unavailable:
        examples = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in leaked_unavailable[:5]]
        errors.append(f"Rows without an available next-day target are present in model-ready data: {examples}")

    for split, df in load_splits(project_root).items():
        if not set(df["date"]).issubset(set(model_ready["date"])):
            errors.append(f"{split}.csv contains rows not present in vnindex_model_ready.csv")
        if pd.to_numeric(df["target_var_next"], errors="coerce").isna().any():
            errors.append(f"{split}.csv contains unavailable target_var_next values")

    return errors


def validate_prediction_schema(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    paths = prediction_files(project_root)
    if not paths:
        return ["No prediction files found under outputs/predictions."]

    for path in paths:
        rel = path.relative_to(project_root)
        df = load_prediction_file(path)
        if list(df.columns) != PREDICTION_COLUMNS:
            errors.append(f"{rel} has columns {list(df.columns)}, expected {PREDICTION_COLUMNS}")
            continue
        for column in ("date", "target_date"):
            if df[column].isna().any():
                errors.append(f"{rel} has unparseable {column} values")
        actual = pd.to_numeric(df["actual_var"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(df["pred_var"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            errors.append(f"{rel} contains non-finite actual_var values")
        if np.isfinite(actual).all() and (actual < 0).any():
            errors.append(f"{rel} contains negative actual_var values")
        if not np.isfinite(pred).all():
            errors.append(f"{rel} contains non-finite pred_var values")
        if np.isfinite(pred).all() and (pred <= 0).any():
            errors.append(f"{rel} contains non-positive pred_var values")
        for column in ("model", "split"):
            if df[column].astype(str).str.strip().eq("").any() or df[column].isna().any():
                errors.append(f"{rel} contains empty {column} values")
    return errors


def validate_prediction_target_alignment(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    predictions = load_all_predictions(project_root)
    model_ready = load_model_ready(project_root)[["date", "target_date", "target_var_next"]]

    predictions["actual_var"] = pd.to_numeric(predictions["actual_var"], errors="coerce")
    merged = predictions.merge(model_ready, on=["date", "target_date"], how="left", indicator=True)
    missing = merged["_merge"].eq("left_only")
    if missing.any():
        examples = merged.loc[missing, ["_source_file", "model", "split", "date", "target_date"]].head(10)
        errors.append("Predictions missing model-ready target rows:\n" + examples.to_string(index=False))
        merged = merged.loc[~missing].copy()

    if not merged.empty:
        matches = np.isclose(
            merged["actual_var"].to_numpy(dtype=float),
            merged["target_var_next"].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        if not bool(matches.all()):
            examples = merged.loc[
                ~matches,
                ["_source_file", "model", "split", "date", "target_date", "actual_var", "target_var_next"],
            ].head(10)
            errors.append("Prediction actual_var does not match target_var_next:\n" + examples.to_string(index=False))

    return errors


def common_window_summary(project_root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    predictions = load_all_predictions(project_root)
    common_metrics_path = project_root / "outputs" / "metrics" / "final_model_comparison_common.csv"
    common_metrics = read_csv(common_metrics_path)
    summary: dict[str, dict[str, Any]] = {}

    for split in ("validation", "test"):
        split_predictions = predictions[predictions["split"].astype(str).str.lower() == split].copy()
        split_metrics = common_metrics[common_metrics["split"].astype(str).str.lower() == split].copy()
        models = sorted(split_metrics["model"].astype(str).unique())
        key_sets = {
            model: set(
                split_predictions.loc[
                    split_predictions["model"].astype(str) == model,
                    ["date", "target_date"],
                ].itertuples(index=False, name=None)
            )
            for model in models
        }
        common_keys = set.intersection(*key_sets.values()) if key_sets else set()
        wide_path = project_root / "outputs" / "metrics" / f"predictions_{split}_common_wide.csv"
        wide = parse_dates(read_csv(wide_path), wide_path)
        summary[split] = {
            "models": models,
            "common_keys": common_keys,
            "common_count": len(common_keys),
            "wide": wide,
            "metrics": split_metrics,
        }
    return summary


def validate_common_window(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    summary = common_window_summary(project_root)
    for split, expected_count in EXPECTED_COMMON_COUNTS.items():
        item = summary[split]
        wide = item["wide"]
        metrics = item["metrics"]
        common_count = int(item["common_count"])
        if common_count != expected_count:
            errors.append(f"{split} common key count changed: observed {common_count}, expected {expected_count}")
        if len(wide) != expected_count:
            errors.append(f"{split} common wide row count changed: observed {len(wide)}, expected {expected_count}")
        if wide.duplicated(["date", "target_date"]).any():
            errors.append(f"{split} common wide table has duplicate date/target_date rows")
        if not (wide["target_date"] > wide["date"]).all():
            errors.append(f"{split} common wide target_date values are not strictly after date")
        pred_columns = [column for column in wide.columns if column.startswith("pred_")]
        if not pred_columns:
            errors.append(f"{split} common wide table has no prediction columns")
        for column in pred_columns:
            values = pd.to_numeric(wide[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                errors.append(f"{split} common wide column {column} has invalid predictions")
        if metrics.empty:
            errors.append(f"{split} common metrics are empty")
        else:
            n_obs_values = set(pd.to_numeric(metrics["n_obs"], errors="coerce").astype(int).tolist())
            if n_obs_values != {expected_count}:
                errors.append(
                    f"{split} final_model_comparison_common.csv n_obs values changed: "
                    f"observed {sorted(n_obs_values)}, expected {expected_count}"
                )
        if len(item["models"]) != len(pred_columns):
            errors.append(
                f"{split} common model count ({len(item['models'])}) does not match "
                f"wide prediction columns ({len(pred_columns)})"
            )
    return errors


def split_summary_lines(project_root: Path = PROJECT_ROOT) -> list[str]:
    lines = [
        "| Split | Rows | Date min | Date max | Target date min | Target date max |",
        "|---|---:|---|---|---|---|",
    ]
    for split, df in load_splits(project_root).items():
        lines.append(
            "| {split} | {rows} | {date_min} | {date_max} | {target_min} | {target_max} |".format(
                split=split,
                rows=len(df),
                date_min=df["date"].min().strftime("%Y-%m-%d"),
                date_max=df["date"].max().strftime("%Y-%m-%d"),
                target_min=df["target_date"].min().strftime("%Y-%m-%d"),
                target_max=df["target_date"].max().strftime("%Y-%m-%d"),
            )
        )
    return lines


def prediction_summary_lines(project_root: Path = PROJECT_ROOT) -> list[str]:
    lines = ["| File | Rows | Models | Splits |", "|---|---:|---|---|"]
    for path in prediction_files(project_root):
        df = load_prediction_file(path)
        models = ", ".join(sorted(df["model"].astype(str).unique())) if "model" in df else ""
        splits = ", ".join(sorted(df["split"].astype(str).unique())) if "split" in df else ""
        lines.append(f"| `{path.relative_to(project_root)}` | {len(df)} | {models} | {splits} |")
    return lines


def common_window_summary_lines(project_root: Path = PROJECT_ROOT) -> list[str]:
    summary = common_window_summary(project_root)
    lines = ["| Split | Common rows | Expected rows | Models |", "|---|---:|---:|---:|"]
    for split, item in summary.items():
        lines.append(
            f"| {split} | {item['common_count']} | {EXPECTED_COMMON_COUNTS[split]} | {len(item['models'])} |"
        )
    return lines


def make_report(title: str, lines: list[str], errors: list[str], warnings: list[str] | None = None) -> str:
    warnings = warnings or []
    status = "PASS" if not errors else "FAIL"
    body = [f"# {title}", "", f"Status: **{status}**", "", *lines, ""]
    if warnings:
        body.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    if errors:
        body.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
    else:
        body.extend(["## Errors", "", "None.", ""])
    return "\n".join(body)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_audits(project_root: Path = PROJECT_ROOT) -> list[AuditResult]:
    data_errors = validate_split_integrity(project_root) + validate_target_construction(project_root)
    prediction_errors = validate_prediction_schema(project_root) + validate_prediction_target_alignment(project_root)
    common_errors = validate_common_window(project_root)
    return [
        AuditResult("Data Contract Audit", data_errors, [], split_summary_lines(project_root)),
        AuditResult("Prediction Schema Audit", prediction_errors, [], prediction_summary_lines(project_root)),
        AuditResult("Common Window Audit", common_errors, [], common_window_summary_lines(project_root)),
    ]


def main() -> None:
    results = run_audits(PROJECT_ROOT)
    report_paths = {
        "Data Contract Audit": AUDIT_DIR / "data_contract_report.md",
        "Prediction Schema Audit": AUDIT_DIR / "prediction_schema_report.md",
        "Common Window Audit": AUDIT_DIR / "common_window_report.md",
    }

    for result in results:
        write_report(
            report_paths[result.title],
            make_report(result.title, result.lines, result.errors, result.warnings),
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.title} -> {report_paths[result.title].relative_to(PROJECT_ROOT)}")

    failures = [result for result in results if not result.passed]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
