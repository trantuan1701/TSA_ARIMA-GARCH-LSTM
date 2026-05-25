# Modeling and Forecasting VN-Index Volatility

This repository contains a reproducible empirical pipeline for one-step-ahead VN-Index volatility forecasting. The target is next-day variance, not the VN-Index level:

```text
target_var_next = squared_return.shift(-1)
squared_return = log_return_pct ** 2
log_return_pct = 100 * log(close_t / close_{t-1})
```

The project compares historical and rolling baselines, GARCH, ARIMA-GARCH, LSTM, ARIMA-GARCH-LSTM, and tuned LSTM/hybrid variants on chronological train, validation, and test splits.

## Data

The source data are daily `VNINDEX` historical prices from CafeF for `01/01/2010` through `31/12/2025`.

| Item | Value |
|---|---|
| Symbol | `VNINDEX` |
| Historical page | `https://s.cafef.vn/lich-su-giao-dich-vnindex-1.chn` |
| Ajax endpoint | `https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx` |

The checked-in clean input is:

- `data/processed/vnindex_cafef_2010_2025_clean.csv`
- `data/processed/vnindex_cafef_2010_2025_clean.xlsx`

The modeling split is chronological:

| Split | Date rule |
|---|---|
| Train | `date <= 2019-12-31` |
| Validation | `2020-01-01 <= date <= 2022-12-31` |
| Test | `date >= 2023-01-01` |

No time-series shuffling is used.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── experiment_contract.md
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── tables/
├── paper/
│   ├── sections/
│   ├── tables/
│   ├── figures/
│   ├── main.tex
│   └── Makefile
├── scripts/
│   └── download_vnindex_cafef.py
├── src/
│   ├── prepare_data.py
│   ├── train_econometric.py
│   ├── train_lstm_hybrid.py
│   ├── tune_lstm_hybrid.py
│   ├── evaluate_all.py
│   └── metrics.py
├── Makefile
├── pyproject.toml
├── requirements.txt
├── run_experiment.py
└── README.md
```

## Pipeline Scripts

| Script | Function |
|---|---|
| `scripts/download_vnindex_cafef.py` | Downloads and cleans CafeF VNINDEX data. |
| `src/prepare_data.py` | Builds volatility features, target, chronological splits, EDA figures, and data summary. |
| `src/train_econometric.py` | Trains historical mean, rolling volatility, GARCH(1,1), and ARIMA-GARCH models; writes econometric bridge features. |
| `src/train_lstm_hybrid.py` | Trains the base LSTM and ARIMA-GARCH-LSTM models. |
| `src/tune_lstm_hybrid.py` | Trains tuned LSTM and hybrid variants under `outputs/*/lstm_tuned/`. |
| `src/evaluate_all.py` | Validates predictions, computes final metrics, and writes paper-ready tables and figures. |
| `src/metrics.py` | Shared RMSE, MAE, and QLIKE metric implementation. |
| `run_experiment.py` | Canonical full-pipeline runner. |

## Setup

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If you need to refresh the CafeF source data:

```bash
python scripts/download_vnindex_cafef.py
```

## Make Targets

The root `Makefile` provides stage-level commands:

```bash
make data         # run src/prepare_data.py
make econometric  # run src/train_econometric.py
make lstm         # run src/train_lstm_hybrid.py
make tune         # run src/tune_lstm_hybrid.py
make evaluate     # run src/evaluate_all.py
make paper        # build paper/main.tex through paper/Makefile
make all          # run the full canonical pipeline through run_experiment.py
```

The full runner can also execute selected stages:

```bash
python run_experiment.py data econometric evaluate
python run_experiment.py --skip-tune --skip-paper
```

Running the pipeline overwrites generated artifacts under `data/processed/`, `outputs/`, and `paper/`.

## Outputs

Important generated files include:

### Data

- `data/processed/vnindex_model_ready.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/vnindex_with_econometric_features.csv`

### Predictions

- `outputs/predictions/pred_baseline_mean.csv`
- `outputs/predictions/pred_rolling_vol_5.csv`
- `outputs/predictions/pred_rolling_vol_10.csv`
- `outputs/predictions/pred_rolling_vol_20.csv`
- `outputs/predictions/pred_garch_11.csv`
- `outputs/predictions/pred_arima_garch.csv`
- `outputs/predictions/pred_lstm_base.csv`
- `outputs/predictions/pred_lstm_hybrid.csv`
- `outputs/predictions/lstm_tuned/pred_*.csv`

### Metrics and Tables

- `outputs/metrics/data_summary.csv`
- `outputs/metrics/econometric_metrics.csv`
- `outputs/metrics/lstm_metrics.csv`
- `outputs/metrics/lstm_tuned/lstm_tuned_metrics.csv`
- `outputs/metrics/final_model_comparison_available.csv`
- `outputs/metrics/final_model_comparison_available_test_only.csv`
- `outputs/metrics/final_model_comparison_common.csv`
- `outputs/metrics/final_model_comparison_common_test_only.csv`
- `outputs/tables/table_model_comparison_available.md`
- `outputs/tables/table_model_comparison_available.tex`
- `outputs/tables/table_model_comparison_common.md`
- `outputs/tables/table_model_comparison_common.tex`

### Figures

- `outputs/figures/fig_close_price.png`
- `outputs/figures/fig_log_return.png`
- `outputs/figures/fig_squared_return.png`
- `outputs/figures/fig_rolling_volatility.png`
- `outputs/figures/fig_actual_vs_predicted_test_common.png`
- `outputs/figures/fig_actual_vs_predicted_test_common_zoom.png`
- `outputs/figures/fig_rmse_comparison_common_test.png`
- `outputs/figures/fig_mae_comparison_common_test.png`
- `outputs/figures/fig_qlike_comparison_common_test.png`

## Prediction Schema

Every prediction file uses:

```text
date,target_date,actual_var,pred_var,model,split
```

`date` is the prediction origin date. `target_date` is the date being forecasted. `actual_var` must equal `target_var_next` from the corresponding split file.

## Evaluation Metrics

Let `actual_var` denote the realized variance proxy and `pred_var` denote the model forecast.

| Metric | Definition |
|---|---|
| RMSE | `sqrt(mean((actual_var - pred_var)^2))` |
| MAE | `mean(abs(actual_var - pred_var))` |
| QLIKE | `mean(log(pred_var) + actual_var / pred_var)` |

Predicted variances are clipped with `epsilon = 1e-8` before QLIKE is computed.

## Reproducibility Notes

- The experiment contract is documented in `docs/experiment_contract.md`.
- Splits are chronological and non-overlapping.
- `target_var_next`, `target_date`, `date`, and `split` are forbidden model input features.
- LSTM scalers and imputers are fit on training rows only.
- Validation data are used for model selection and early stopping.
- Test data are reserved for final evaluation.
- Common-window metrics are the preferred paper comparison because all models are evaluated on the same `date,target_date` rows.

## Paper

The IEEE-style paper source is in `paper/`. Build it with:

```bash
make paper
```

The current compiled artifact is `paper/main.pdf`.

## Research Positioning

This is an applied empirical comparison, not a claim of a new model architecture. The repository supports reproducible comparison of econometric, neural, and hybrid volatility forecasting approaches on VN-Index data.
