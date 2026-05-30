# Stage 1 Repository Audit

Scope: source code, experiments, tests, reproducibility, and empirical artifacts only.

## Repository Structure

```text
.
.gitignore
Makefile
README.md
data/
    processed/
        test.csv
        train.csv
        validation.csv
        vnindex_cafef_2010_2025_clean.csv
        vnindex_cafef_2010_2025_clean.xlsx
        vnindex_model_ready.csv
        vnindex_with_econometric_features.csv
    raw/
        vnindex_cafef_2010_2025_raw.csv
docs/
    experiment_contract.md
    vnindex_volatility_codex_pipeline.md
outputs/
    advanced/
        audit/
        figures/
        models/
        predictions/
        tables/
    audit/
        common_window_report.md
        data_contract_report.md
        econometric_diagnostics_report.md
        prediction_schema_report.md
        robustness_report.md
        statistical_tests_report.md
    figures/
        fig_actual_vs_predicted_test_common.png
        fig_actual_vs_predicted_test_common_zoom.png
        fig_close_price.png
        fig_log_return.png
        fig_loss_difference_vs_garch.png
        fig_lstm_base_loss.png
        fig_lstm_hybrid_loss.png
        fig_mae_comparison_common_test.png
        fig_qlike_comparison_common_test.png
        fig_regime_qlike.png
        fig_rmse_comparison_common_test.png
        fig_rolling_volatility.png
        fig_squared_return.png
        lstm_tuned/
    metrics/
        arima_garch_summary.txt
        arima_order_selection.csv
        arima_summary.txt
        data_summary.csv
        econometric_metrics.csv
        econometric_metrics_test_only.csv
        final_evaluation_summary.txt
        final_model_comparison_available.csv
        final_model_comparison_available_test_only.csv
        final_model_comparison_common.csv
        final_model_comparison_common_test_only.csv
        garch_summary.txt
        lstm_base_training_history.csv
        lstm_feature_summary.csv
        lstm_hybrid_training_history.csv
        lstm_metrics.csv
        lstm_metrics_test_only.csv
        lstm_tuned/
        predictions_test_common_wide.csv
        predictions_validation_common_wide.csv
    models/
        arima_garch_garch.pkl
        arima_model.pkl
        garch_11.pkl
        lstm_base.keras
        lstm_base_features.txt
        lstm_base_imputer_values.json
        lstm_base_scaler.pkl
        lstm_hybrid.keras
        lstm_hybrid_features.txt
        lstm_hybrid_imputer_values.json
        lstm_hybrid_scaler.pkl
        lstm_tuned/
    predictions/
        lstm_tuned/
        pred_arima_garch.csv
        pred_baseline_mean.csv
        pred_garch_11.csv
        pred_lstm_base.csv
        pred_lstm_hybrid.csv
        pred_rolling_vol_10.csv
        pred_rolling_vol_20.csv
        pred_rolling_vol_5.csv
    proxy_robustness/
        audit/
        figures/
        optional_retraining/
        tables/
    stage1_extensions/
        figures/
        metrics/
        predictions/
        run_manifest.json
        tables/
    tables/
        table_arch_lm_tests.csv
        table_dm_tests_absolute_error.csv
        table_dm_tests_qlike.csv
        table_dm_tests_squared_error.csv
        table_garch_diagnostics.csv
        table_model_comparison_available.md
        table_model_comparison_available.tex
        table_model_comparison_common.md
        table_model_comparison_common.tex
        table_regime_results.csv
        table_stationarity_tests.csv
        table_underprediction_analysis.csv
        table_yearly_results.csv
paper/
    IEEEtran.cls
    Makefile
    figures/
        fig_actual_vs_predicted_test_common_zoom.png
        fig_lstm_tuned_qlike_comparison.png
        fig_proxy_correlation_heatmap.png
        fig_proxy_ranking_heatmap.png
        fig_qlike_comparison_common_test.png
        fig_rolling_volatility.png
        fig_squared_return.png
        fig_volatility_proxy_timeseries.png
    issues/
        20260525-vnindex-volatility-ieee.csv
    main.aux
    main.bbl
    main.blg
    main.log
    main.pdf
    main.tex
    main.xdv
    plan/
        20260525-vnindex-volatility-ieee.md
    references.bib
    report_materials.md
    sections/
        01_introduction.tex
        02_related_work.tex
        03_methodology.tex
        04_experimental_setup.tex
        05_results_and_discussion.tex
        06_conclusion.tex
    tables/
        advanced_garch_selected_models.tex
        calibration_combination_summary.tex
        data_splits.tex
        model_comparison_common_test.tex
        proxy_robustness_rank_summary.tex
        statistical_comparison_summary.tex
        tuned_selection.tex
        var_backtesting_summary.tex
paper.pdf
pyproject.toml
reports/
    stage1_extensions/
        repo_audit_stage1.md
requirements.txt
run_experiment.py
src/
    advanced_experiment_utils.py
    advanced_forecast_combinations.py
    advanced_hypothesis_tests.py
    advanced_model_search.py
    advanced_var_backtesting.py
    audit_experiment.py
    econometric_diagnostics.py
    empirical_rigor_utils.py
    evaluate_all.py
    metrics.py
    prepare_data.py
    proxy_robustness.py
    robustness_analysis.py
    run_advanced_experiments.py
    stage1_extensions.py
    statistical_tests.py
    train_econometric.py
    train_lstm_hybrid.py
    tune_lstm_hybrid.py
tests/
    test_common_window.py
    test_data_contract.py
    test_metrics.py
    test_no_leakage_alignment.py
    test_prediction_schema.py
    test_proxy_formulas.py
    test_var_backtests.py
```

## Main Files and Scripts

| Area | Files |
|---|---|
| data preparation | `src/prepare_data.py` |
| feature engineering | `src/prepare_data.py; src/train_econometric.py bridge features` |
| model training | `src/train_econometric.py; src/train_lstm_hybrid.py; src/tune_lstm_hybrid.py` |
| GARCH / ARIMA-GARCH | `src/train_econometric.py` |
| advanced GARCH-family search | `src/advanced_model_search.py; src/run_advanced_experiments.py` |
| LSTM / hybrid models | `src/train_lstm_hybrid.py; src/tune_lstm_hybrid.py` |
| evaluation | `src/evaluate_all.py; src/metrics.py` |
| proxy robustness | `src/proxy_robustness.py` |
| VaR backtesting | `src/advanced_var_backtesting.py` |
| Makefile | `Makefile; paper/Makefile` |
| configs | `pyproject.toml; requirements.txt` |
| tests | `tests/test_*.py` |
| Stage 1 extensions | `src/stage1_extensions.py` |

## Current Pipeline Summary

- `src/prepare_data.py` constructs percent log returns, squared returns, trailing rolling volatility, next-day target variance, and chronological train/validation/test splits.
- `src/train_econometric.py` trains HistoricalMean, RollingVol, GARCH(1,1), and ARIMA-GARCH with training-only parameter estimation and recursive state updates.
- `src/train_lstm_hybrid.py` trains LSTM and ARIMA-GARCH-LSTM with train-only imputers/scalers and split-local sequences.
- `src/tune_lstm_hybrid.py`, `src/advanced_model_search.py`, and advanced experiment scripts add tuned neural, calibrated, GARCH-family, refit, combination, and hypothesis-testing artifacts.
- `src/evaluate_all.py` computes common-window and available-window metrics with shared QLIKE clipping.

## Current Available Outputs

| Root | File count |
|---|---:|
| `outputs/metrics` | 29 |
| `outputs/predictions` | 14 |
| `outputs/advanced` | 99 |
| `outputs/proxy_robustness` | 21 |
| `outputs/stage1_extensions` | 43 |

## Leakage and Reproducibility Audit

| Check | Status | Detail |
|---|---|---|
| train_target_date_after_origin | pass | rows=2471 |
| validation_target_date_after_origin | pass | rows=750 |
| test_target_date_after_origin | pass | rows=745 |
| target_var_next_equals_next_squared_return | pass |  |
| rolling_vol_5_trailing_only_formula | pass | pandas rolling(window).std() matches checked-in feature |
| rolling_vol_10_trailing_only_formula | pass | pandas rolling(window).std() matches checked-in feature |
| rolling_vol_20_trailing_only_formula | pass | pandas rolling(window).std() matches checked-in feature |
| lstm_scaler_imputer_fit_scope | pass_by_code_audit | src/train_lstm_hybrid.py fits imputer/scaler on train then applies to validation/test. |
| lstm_sequences_do_not_cross_split_boundaries | pass_by_code_audit | create_sequences is called separately for train, validation, and test splits. |
| qlike_epsilon_consistency | pass | Shared EPSILON imported from src/metrics.py = 1e-08. |
| forbidden_predictive_columns | pass_by_code_audit | target_var_next, target_date, date, and split are excluded from LSTM feature lists. |

## Missing Pieces Found Before Stage 1 Extensions

- EWMA/RiskMetrics and HAR-type volatility baselines were not present in the canonical baseline set.
- Existing VaR tables covered common Normal VaR at fewer levels than the requested 5%, 2.5%, and 1% extended output.
- Regime/spike diagnostics existed partially, but not as a clean Stage 1 artifact spanning existing, EWMA, and HAR predictions.
- Advanced GARCH search artifacts existed, but no consolidated Stage 1 audit summary existed under the new output folder.
- Proxy robustness artifacts existed under `outputs/proxy_robustness`, but not in a cleaned Stage 1 folder with new baselines included.

## Tasks Implemented in Stage 1

- Added validation-selected EWMA baseline over lambda values 0.90, 0.94, 0.97, and 0.99.
- Added validation-selected HAR baselines: HAR-SquaredReturn, Log-HAR-SquaredReturn, and HAR-Parkinson when OHLC columns are available.
- Generated test-period regime metrics and spike diagnostics using train/validation spike thresholds.
- Generated extended Normal VaR backtests at 5%, 2.5%, and 1% with Kupiec, Christoffersen independence, conditional coverage, clustering, and Basel-style 1% zones.
- Generated an existing-prediction time-split diagnostic and explicitly labeled it as not true rolling-origin retraining.
- Generated advanced GARCH search audit and failure summaries.
- Regenerated proxy correlations and proxy rank stability with Stage 1 baselines included.
- Generated neural/hybrid risk diagnostics from existing prediction artifacts.
- Added Stage 1 Makefile targets and tests for proxy formulas, leakage alignment, and VaR backtest helpers.

## Tasks Skipped and Why

- Full rolling-origin retraining was skipped because retraining GARCH, ARIMA-GARCH, advanced GARCH, and neural models over multiple windows is materially more expensive; a lighter existing-prediction time-split diagnostic was generated instead.
- White Reality Check / Hansen SPA was not implemented in this stage because stationary/block bootstrap inference over the full model universe needs additional design choices and runtime budget.
- Distribution-specific VaR for Student-t, skewed-t, and GED was not computed because current prediction CSVs do not retain fitted innovation shape/skew parameters; the Stage 1 VaR output exposes `distribution_specific_quantile_available=False` and uses common Normal VaR.

## Stage 1 Artifacts Generated

- `outputs/stage1_extensions/figures/neural_spike_failure.png`
- `outputs/stage1_extensions/figures/proxy_correlation_heatmap.png`
- `outputs/stage1_extensions/figures/proxy_rank_heatmap_top_models.png`
- `outputs/stage1_extensions/figures/regime_qlike_comparison.png`
- `outputs/stage1_extensions/figures/rolling_origin_qlike.png`
- `outputs/stage1_extensions/figures/spike_underprediction_comparison.png`
- `outputs/stage1_extensions/figures/var_exceedances_advgarch_bestasymmetric.png`
- `outputs/stage1_extensions/figures/var_exceedances_advgarch_bestqlike.png`
- `outputs/stage1_extensions/figures/var_exceedances_arima_garch.png`
- `outputs/stage1_extensions/figures/var_exceedances_arima_garch_lstm.png`
- `outputs/stage1_extensions/figures/var_exceedances_garch_1_1.png`
- `outputs/stage1_extensions/figures/var_exceedances_historicalmean.png`
- `outputs/stage1_extensions/figures/var_exceedances_hybrid_qlike.png`
- `outputs/stage1_extensions/figures/var_exceedances_lstm.png`
- `outputs/stage1_extensions/figures/var_exceedances_lstm_qlike.png`
- `outputs/stage1_extensions/figures/var_exceedances_rollingvol_20.png`
- `outputs/stage1_extensions/metrics/advanced_search_audit.csv`
- `outputs/stage1_extensions/metrics/advanced_search_failures_summary.csv`
- `outputs/stage1_extensions/metrics/ewma_metrics.csv`
- `outputs/stage1_extensions/metrics/har_coefficients.csv`
- `outputs/stage1_extensions/metrics/har_metrics.csv`
- `outputs/stage1_extensions/metrics/leakage_audit.csv`
- `outputs/stage1_extensions/metrics/neural_diagnostics.csv`
- `outputs/stage1_extensions/metrics/prediction_inventory.csv`
- `outputs/stage1_extensions/metrics/proxy_correlations.csv`
- `outputs/stage1_extensions/metrics/proxy_model_metrics.csv`
- `outputs/stage1_extensions/metrics/proxy_rank_stability.csv`
- `outputs/stage1_extensions/metrics/regime_metrics.csv`
- `outputs/stage1_extensions/metrics/rolling_origin_metrics.csv`
- `outputs/stage1_extensions/metrics/spike_diagnostics.csv`
- `outputs/stage1_extensions/metrics/var_backtests_extended.csv`
- `outputs/stage1_extensions/predictions/ewma_predictions.csv`
- `outputs/stage1_extensions/predictions/har_predictions.csv`
- `outputs/stage1_extensions/run_manifest.json`
- `outputs/stage1_extensions/tables/advanced_search_summary.tex`
- `outputs/stage1_extensions/tables/ewma_metrics.tex`
- `outputs/stage1_extensions/tables/har_metrics.tex`
- `outputs/stage1_extensions/tables/neural_diagnostics.tex`
- `outputs/stage1_extensions/tables/proxy_rank_stability.tex`
- `outputs/stage1_extensions/tables/regime_metrics.tex`
- `outputs/stage1_extensions/tables/rolling_origin_metrics.tex`
- `outputs/stage1_extensions/tables/spike_diagnostics.tex`
- `outputs/stage1_extensions/tables/var_backtests_extended.tex`

## Exact Commands Run

- `pwd && tree -L 3 -a -I '.git|__pycache__|.pytest_cache|.venv|venv|node_modules'`
- `rg --files`
- `sed -n '1,240p' README.md`
- `sed -n '1,260p' Makefile`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py baselines`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py risk`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py robustness`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py diagnostics`
- `PYTHONDONTWRITEBYTECODE=1 make stage1-all`
- `PYTHONDONTWRITEBYTECODE=1 pytest`
- `PYTHONDONTWRITEBYTECODE=1 make stage1-test`
- `PYTHONDONTWRITEBYTECODE=1 make stage1-baselines stage1-risk stage1-robustness stage1-diagnostics`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py audit --test-status "pytest: PASS (22 passed); make stage1-test: PASS (22 passed); make stage1-all and component Stage 1 targets: PASS"`
- `PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py manifest`

## Test Status

- pytest: PASS (22 passed); make stage1-test: PASS (22 passed); make stage1-all and component Stage 1 targets: PASS

## Failures or Errors Encountered

- Initial `python src/stage1_extensions.py baselines` failed because pandas `DataFrame.to_latex()` required missing optional dependency `jinja2` in this environment.
- Resolution: replaced pandas `to_latex()` usage with a deterministic manual LaTeX tabular writer in `src/stage1_extensions.py`; all Stage 1 subcommands and tests passed after the fix.
- No unresolved Stage 1 script failures remain. Skipped methodological items are listed above.

## Next Steps for Stage 2

- Reframe primary claims around validation-selected models, proxy robustness, and risk-aware evaluation rather than test-best model discovery.
- Use EWMA and HAR results as benchmark-completeness evidence.
- Use regime, spike, and VaR outputs for economic interpretation, especially underestimation during high-volatility days.
- Clearly label advanced-test-best rankings as exploratory and keep the validation-selected GARCH-family model as the primary advanced result.
- State that full rolling-origin retraining and distribution-specific VaR are future robustness extensions unless separately run.
