#!/usr/bin/env python3
"""Prepare VN-Index CafeF data for volatility forecasting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "vnindex_cafef_2010_2025_clean.csv"

TRAIN_END = pd.Timestamp("2019-12-31")
VALIDATION_START = pd.Timestamp("2020-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")

CLEAN_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "log_return_pct",
    "squared_return",
    "abs_return",
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
]

MODEL_COLUMNS = CLEAN_COLUMNS + ["target_var_next", "target_date"]
ESSENTIAL_COLUMNS = [
    "date",
    "close",
    "log_return_pct",
    "squared_return",
    "abs_return",
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
    "target_var_next",
    "target_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare model-ready VN-Index volatility forecasting data."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Clean CafeF CSV input. Defaults to data/processed/vnindex_cafef_2010_2025_clean.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project output root. Defaults to the repository root.",
    )
    return parser.parse_args()


def load_clean_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError("Input data must contain a date column.")
    if "close" not in df.columns:
        raise ValueError("Input data must contain a close column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df.loc[df["date"].isna()].index.tolist()[:10]
        raise ValueError(f"Could not parse date values at input rows: {bad_rows}")

    df = df.sort_values("date").reset_index(drop=True)

    duplicate_dates = df.loc[df["date"].duplicated(), "date"]
    if not duplicate_dates.empty:
        examples = duplicate_dates.dt.strftime("%Y-%m-%d").head(10).tolist()
        raise ValueError(f"Duplicate dates found in input data: {examples}")

    numeric_columns = [column for column in CLEAN_COLUMNS if column != "date" and column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["close"].isna().any():
        missing_count = int(df["close"].isna().sum())
        raise ValueError(f"Input data contains {missing_count} missing close values.")

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    for column in CLEAN_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan

    close = prepared["close"].astype(float)
    prepared["log_return_pct"] = 100.0 * np.log(close / close.shift(1))
    prepared["log_return_pct"] = prepared["log_return_pct"].replace([np.inf, -np.inf], np.nan)
    prepared["squared_return"] = prepared["log_return_pct"] ** 2
    prepared["abs_return"] = prepared["log_return_pct"].abs()

    for window in (5, 10, 20):
        prepared[f"rolling_vol_{window}"] = prepared["log_return_pct"].rolling(window=window).std()

    prepared["target_var_next"] = prepared["squared_return"].shift(-1)
    prepared["target_date"] = prepared["date"].shift(-1)

    return prepared[MODEL_COLUMNS]


def make_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    model_ready = df.dropna(subset=ESSENTIAL_COLUMNS).copy()
    model_ready = model_ready.sort_values("date").reset_index(drop=True)

    if model_ready.empty:
        raise ValueError("No model-ready rows remain after dropping missing essential values.")

    return model_ready


def split_data(model_ready: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = model_ready[
        (model_ready["date"] <= TRAIN_END) & (model_ready["target_date"] <= TRAIN_END)
    ].copy()
    validation = model_ready[
        (model_ready["date"] >= VALIDATION_START)
        & (model_ready["date"] <= VALIDATION_END)
        & (model_ready["target_date"] <= VALIDATION_END)
    ].copy()
    test = model_ready[model_ready["date"] >= TEST_START].copy()

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    for name, split in splits.items():
        if split.empty:
            raise ValueError(f"{name} split is empty.")
        if not split["date"].is_monotonic_increasing:
            raise ValueError(f"{name} split is not sorted chronologically.")
        missing_columns = [column for column in MODEL_COLUMNS if column not in split.columns]
        if missing_columns:
            raise ValueError(f"{name} split is missing columns: {missing_columns}")

    if not splits["train"]["date"].max() < splits["validation"]["date"].min():
        raise ValueError("Train and validation split dates overlap or are out of order.")
    if not splits["validation"]["date"].max() < splits["test"]["date"].min():
        raise ValueError("Validation and test split dates overlap or are out of order.")

    if splits["train"]["target_date"].max() > TRAIN_END:
        raise ValueError("Train target_date crosses into the validation period.")
    if splits["validation"]["target_date"].max() > VALIDATION_END:
        raise ValueError("Validation target_date crosses into the test period.")


def save_csv_outputs(
    model_ready: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    output_root: Path,
) -> dict[str, Path]:
    processed_dir = output_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model_ready": processed_dir / "vnindex_model_ready.csv",
        "train": processed_dir / "train.csv",
        "validation": processed_dir / "validation.csv",
        "test": processed_dir / "test.csv",
    }

    model_ready.to_csv(paths["model_ready"], index=False, encoding="utf-8-sig")
    for name, split in splits.items():
        split.to_csv(paths[name], index=False, encoding="utf-8-sig")

    return paths


def save_figures(model_ready: pd.DataFrame, output_root: Path) -> dict[str, Path]:
    figures_dir = output_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "close_price": figures_dir / "fig_close_price.png",
        "log_return": figures_dir / "fig_log_return.png",
        "squared_return": figures_dir / "fig_squared_return.png",
        "rolling_volatility": figures_dir / "fig_rolling_volatility.png",
    }

    plot_series(
        model_ready,
        y_columns=["close"],
        title="VN-Index Close Price",
        ylabel="Index level",
        output_path=figure_paths["close_price"],
    )
    plot_series(
        model_ready,
        y_columns=["log_return_pct"],
        title="VN-Index Log Return",
        ylabel="Log return (%)",
        output_path=figure_paths["log_return"],
    )
    plot_series(
        model_ready,
        y_columns=["squared_return"],
        title="VN-Index Squared Return",
        ylabel="Squared log return",
        output_path=figure_paths["squared_return"],
    )
    plot_series(
        model_ready,
        y_columns=["rolling_vol_5", "rolling_vol_10", "rolling_vol_20"],
        title="VN-Index Rolling Volatility",
        ylabel="Rolling standard deviation",
        output_path=figure_paths["rolling_volatility"],
    )

    return figure_paths


def plot_series(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in y_columns:
        ax.plot(df["date"], df[column], linewidth=1.1, label=column)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(y_columns) > 1:
        ax.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_data_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dataset_name, df in datasets.items():
        rows.extend(
            [
                {"dataset": dataset_name, "metric": "row_count", "value": len(df)},
                {"dataset": dataset_name, "metric": "date_min", "value": format_date_min(df, "date")},
                {"dataset": dataset_name, "metric": "date_max", "value": format_date_max(df, "date")},
            ]
        )

        if "target_date" in df.columns:
            rows.extend(
                [
                    {
                        "dataset": dataset_name,
                        "metric": "target_date_min",
                        "value": format_date_min(df, "target_date"),
                    },
                    {
                        "dataset": dataset_name,
                        "metric": "target_date_max",
                        "value": format_date_max(df, "target_date"),
                    },
                ]
            )

        for column, missing_count in df.isna().sum().items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": f"missing_{column}",
                    "value": int(missing_count),
                }
            )

    return pd.DataFrame(rows)


def format_date_min(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return ""
    return pd.to_datetime(df[column]).min().strftime("%Y-%m-%d")


def format_date_max(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return ""
    return pd.to_datetime(df[column]).max().strftime("%Y-%m-%d")


def save_data_summary(
    raw_prepared: pd.DataFrame,
    model_ready: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    output_root: Path,
) -> Path:
    metrics_dir = output_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    datasets = {"prepared_full": raw_prepared, "model_ready": model_ready, **splits}
    summary = build_data_summary(datasets)
    summary_path = metrics_dir / "data_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def validate_target_definition(prepared_full: pd.DataFrame) -> None:
    recalculated = prepared_full["squared_return"].shift(-1)
    comparable = prepared_full["target_var_next"].iloc[:-1].reset_index(drop=True)
    expected = recalculated.iloc[:-1].reset_index(drop=True)
    if not np.allclose(comparable, expected, equal_nan=True):
        raise ValueError("target_var_next is not equal to next-row squared_return.")


def validate_rolling_features(prepared_full: pd.DataFrame) -> None:
    source = prepared_full["log_return_pct"]
    for window in (5, 10, 20):
        column = f"rolling_vol_{window}"
        expected = source.rolling(window=window).std()
        if not np.allclose(prepared_full[column], expected, equal_nan=True):
            raise ValueError(f"{column} is not consistent with log_return_pct.")


def validate_outputs(
    csv_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    summary_path: Path,
) -> None:
    all_paths = list(csv_paths.values()) + list(figure_paths.values()) + [summary_path]
    missing = [path for path in all_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected output files were not created: {missing}")

    empty_files = [path for path in all_paths if path.stat().st_size == 0]
    if empty_files:
        raise ValueError(f"Expected output files are empty: {empty_files}")


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_root = args.output_root.resolve()

    clean_df = load_clean_data(input_csv)
    prepared_full = add_volatility_features(clean_df)
    validate_target_definition(prepared_full)
    validate_rolling_features(prepared_full)
    model_ready = make_model_ready(prepared_full)

    splits = split_data(model_ready)
    csv_paths = save_csv_outputs(model_ready, splits, output_root)
    figure_paths = save_figures(prepared_full, output_root)
    summary_path = save_data_summary(prepared_full, model_ready, splits, output_root)
    validate_outputs(csv_paths, figure_paths, summary_path)

    print("Stage 1 data preparation complete")
    print(f"Model-ready rows: {len(model_ready)}")
    print(f"Model-ready date range: {format_date_min(model_ready, 'date')} to {format_date_max(model_ready, 'date')}")
    for name, split in splits.items():
        print(
            f"{name}: {len(split)} rows, "
            f"{format_date_min(split, 'date')} to {format_date_max(split, 'date')}, "
            f"target_date {format_date_min(split, 'target_date')} to {format_date_max(split, 'target_date')}"
        )

    print("\nSaved files:")
    for path in [*csv_paths.values(), *figure_paths.values(), summary_path]:
        print(path)


if __name__ == "__main__":
    main()
