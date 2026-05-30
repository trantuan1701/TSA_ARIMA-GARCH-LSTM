# Reproducibility Notes

This project forecasts one-step-ahead VN-Index volatility using daily CafeF OHLC data.
The primary target is next-day squared percentage log return:

```text
log_return_pct = 100 * log(close_t / close_{t-1})
squared_return = log_return_pct ** 2
target_var_next = squared_return.shift(-1)
```

## Environment

Install the checked-in dependency set from the repository root:

```bash
python -m pip install -r requirements.txt
```

The current regenerated final artifacts were verified with Python 3.13.11 and:

```text
pandas 3.0.2
numpy 2.4.4
matplotlib 3.10.9
statsmodels 0.14.6
arch 8.0.0
tensorflow 2.21.0
scipy 1.17.1
pyarrow 24.0.0
pytest 9.0.3
```

## Canonical Commands

Regenerate the original data/model/evaluation pipeline:

```bash
PYTHONDONTWRITEBYTECODE=1 python run_experiment.py --skip-paper
```

Regenerate extension and final empirical artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py all
PYTHONDONTWRITEBYTECODE=1 python src/final_financial_econometrics.py all
```

Equivalent Make target for the final artifacts:

```bash
make final-artifacts
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Compile the manuscript:

```bash
make -C paper all
```

## Primary Evidence Files

- `reports/repo_audit_financial_econometrics.md`
- `outputs/final/common_window_master_results.csv`
- `outputs/final/common_window_predictions.parquet`
- `outputs/final/common_window_dm_tests.csv`
- `outputs/final/common_window_var_backtests.csv`
- `outputs/final/garch_parameter_diagnostics.csv`
- `outputs/final/garch_residual_diagnostics.csv`
- `outputs/final/refit_protocol_results.csv`
- `outputs/final/daily_proxy_robustness_metrics.csv`
- `outputs/audit/advanced_garch_failure_taxonomy.csv`
- `paper/tables/common_window_master_results.tex`
- `paper/tables/advanced_garch_failure_audit.tex`
- `paper/tables/return_descriptive_statistics.tex`
- `paper/tables/volatility_clustering_diagnostics.tex`
- `paper/tables/garch_parameter_diagnostics.tex`
- `paper/tables/garch_residual_diagnostics.tex`
- `paper/tables/refit_protocol_results.tex`
- `paper/tables/common_window_var_backtests.tex`
- `paper/tables/common_window_spike_diagnostics.tex`

Use `outputs/final/common_window_master_results.csv` as the authoritative main leaderboard. Older available-window tables mix 726-row neural windows with 745-row econometric windows and should be treated as secondary or diagnostic.
