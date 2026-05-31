# Final Revision Execution Log

Date: 2026-05-30

## Verification

- Ran `PYTHONDONTWRITEBYTECODE=1 pytest -q`: 25 tests passed before new diagnostics; 26 tests passed after adding the GARCH persistence test.
- Ran `PYTHONDONTWRITEBYTECODE=1 make final-artifacts`: regenerated final common-window outputs consistently.
- Confirmed the common-window primary leaderboard remains at \(N=726\), target dates 2023-02-07 to 2025-12-31.
- Confirmed the advanced GARCH failure taxonomy still reconciles to 576 failures:
  - 432 invalid forecast-start failures for AR-mean lag specifications.
  - 144 non-finite forecast failures.

## New Diagnostics

- Added GARCH parameter and residual diagnostics:
  - `outputs/final/garch_parameter_diagnostics.csv`
  - `outputs/final/garch_residual_diagnostics.csv`
  - `paper/tables/garch_parameter_diagnostics.tex`
  - `paper/tables/garch_residual_diagnostics.tex`
  - `paper/figures/fig_garch_residual_diagnostics.png`
- GARCH(1,1) persistence is \(\alpha+\beta=0.973264\); half-life is 25.58 trading days.
- EGARCH asymmetry parameters are negative and statistically significant for both AdvGARCH-BestQLIKE and AdvGARCH-BestAsymmetric.
- Squared standardized residual and ARCH-LM tests do not reject remaining ARCH effects at lags 5, 10, or 20 for the selected GARCH-family models.

## Refit Robustness

- Added a controlled expanding-window monthly refit experiment for:
  - GARCH(1,1)
  - ARIMA-GARCH
  - EWMA(lambda=0.90)
  - HAR-Parkinson
  - AdvGARCH-BestQLIKE
  - AdvGARCH-BestAsymmetric
- Outputs:
  - `outputs/final/refit_protocol_results.csv`
  - `outputs/final/refit_protocol_predictions.parquet`
  - `outputs/final/refit_protocol_predictions.csv`
  - `outputs/final/refit_protocol_diagnostics.csv`
  - `paper/tables/refit_protocol_results.tex`
  - `paper/figures/fig_refit_protocol_qlike.png`
- Complete-window expanding-monthly QLIKE results:
  - AdvGARCH-BestAsymmetric: 1.014849
  - HAR-Parkinson: 1.029873
  - GARCH(1,1): 1.047945
  - ARIMA-GARCH: 1.049005
  - EWMA(lambda=0.90): 1.219608
- AdvGARCH-BestQLIKE failed in 19 monthly refit segments and produced only 340 common-window refit forecasts; it is marked operationally incomplete.

## Neural Retraining Decision

- Neural sequence construction has been fixed in code, but full neural retraining was not performed.
- Reason: the scripts write to existing neural model/prediction paths and tuned training covers multiple TensorFlow variants for up to 200 epochs.
- Decision: keep the fair \(N=726\) common-window leaderboard as the primary evidence and state full neural retraining as a limitation.

## Manuscript

- Rewrote the abstract, introduction, data/econometric motivation, model/protocol, results/discussion, limitations, and conclusion sections.
- Removed main-text Stage 1 wording and old 745-row risk/result claims.
- Replaced the main leaderboard with `paper/tables/common_window_master_results.tex`.
- Added GARCH diagnostics, refit robustness, common-window spike, common-window VaR, and daily proxy robustness figures/tables.
- Described VaR as a common-Normal variance-to-risk mapping, not model-specific tail-distribution VaR.

## Compile

- Ran `make -C paper all`.
- Compilation succeeded and wrote `paper/main.pdf`.
- Copied final PDF to `paper/final_vnindex_volatility_financial_econometrics.pdf`.
- Checked `paper/main.log` for undefined references/citations and overfull boxes; none remained after table-format tightening.

## Remaining Limitations

- Model-specific VaR remains unimplemented because distribution-specific fitted parameters are not consistently retained in prediction artifacts.
- Rolling-window refit was omitted; expanding-window monthly refit was completed.
- Full-window neural retraining remains pending but does not affect the aligned \(N=726\) primary comparison.

## 2026-05-31 Baseline Preservation Before Neural/VaR/ES/Dynamic Completion

- Archived current verified artifacts under `outputs/archive/before_neural_var_es_dynamic_completion/`, `models/archive/before_neural_var_es_dynamic_completion/`, and `paper/archive/before_neural_var_es_dynamic_completion/`.
- Generated `outputs/audit/baseline_artifact_manifest.csv`, `outputs/audit/baseline_artifact_manifest.json`, and `reports/baseline_artifact_manifest.md` with SHA-256 hashes and conservative neural legacy labels.
- Ran `PYTHONDONTWRITEBYTECODE=1 pytest -q`: 26 tests passed.
- Ran `PYTHONDONTWRITEBYTECODE=1 make final-artifacts`: completed successfully.
- Ran `make -C paper all`: completed successfully and wrote `paper/main.pdf`; baseline underfull-box warnings were emitted by Tectonic.
- Reproduced the authoritative common-window leaderboard at \(N=726\), target dates 2023-02-07 to 2025-12-31:
  - AdvGARCH-BestAsymmetric QLIKE 1.021413.
  - AdvGARCH-BestQLIKE QLIKE 1.022898.
  - HAR-Parkinson QLIKE 1.029373.
  - GARCH(1,1) QLIKE 1.052888.
- Reproduced GARCH(1,1) persistence \(\alpha+\beta=0.973264\) and half-life 25.58 trading days.
- Reproduced negative EGARCH asymmetry parameters for AdvGARCH-BestAsymmetric and AdvGARCH-BestQLIKE.
- Reproduced the expanding-monthly refit instability: AdvGARCH-BestQLIKE has 19 failed monthly segments and only 340 expanding-monthly forecasts.

## 2026-05-31 Neural/VaR/ES/Dynamic Completion Pass

- Verified repository metadata:
  - Branch: `master`.
  - Commit: `c1bd976d8d31dfcb417293dffdbf56060c236564`.
  - Remote: `https://github.com/trantuan1701/TSA_ARIMA-GARCH-LSTM.git`.
- Updated reproducibility files and manifests:
  - `README_REPRODUCIBILITY.md`.
  - `outputs/artifact_manifest.csv`.
  - `outputs/artifact_manifest.json`.
  - `reports/reproducibility_statement_audit.md`.
- Preserved legacy neural artifacts under:
  - `outputs/neural_legacy_pre_context_fix/`.
  - `models/neural_legacy_pre_context_fix/`.
- Retrained corrected-context neural and hybrid models under:
  - `outputs/neural_corrected_context_v2/`.
  - `outputs/advanced_corrected_context_v2/`.
- Corrected-context neural test coverage is complete for the full test window:
  - Validation forecasts: 750.
  - Test forecasts: 745.
  - Primary test target dates: 2023-01-04 to 2025-12-31.
- The authoritative primary leaderboard moved from \(N=726\) to \(N=745\) because all primary rows now have aligned corrected-context forecasts:
  - AdvGARCH-BestAsymmetric QLIKE 1.029992.
  - AdvGARCH-BestQLIKE QLIKE 1.032548.
  - HAR-Parkinson QLIKE 1.034521.
  - GARCH(1,1) QLIKE 1.062958.
  - LSTM QLIKE 1.163862.
  - ARIMA-GARCH-LSTM QLIKE 1.172986.
- Generated corrected neural comparison artifacts:
  - `outputs/final/primary_results_corrected_neural.csv`.
  - `outputs/final/neural_retraining_comparison.csv`.
  - `outputs/final/neural_seed_robustness.csv`.
  - `paper/tables/neural_retraining_comparison.tex`.
- Recomputed corrected regime/spike diagnostics:
  - `outputs/final/regime_spike_results_corrected_neural.csv`.
  - `paper/tables/regime_spike_results_corrected_neural.tex`.
  - Corrected neural and hybrid forecasts remain weaker than the best econometric models overall, but selected neural-dependent calibrated/combination forecasts retain lower spike-day QLIKE than the leading static econometric rows.
- Added common-Normal VaR/ES for all primary models and the regime-aware candidate:
  - `outputs/final/common_distribution_var_es_all_models.csv`.
  - `paper/tables/common_distribution_var_es_main.tex`.
  - `paper/tables/common_distribution_var_es_full_appendix.tex`.
  - The common mapping is zero-mean lower-tail Normal; it is not described as model-specific VaR.
- Added model-specific Normal VaR/ES for retained GARCH-family specifications:
  - `outputs/final/model_specific_var_es_garch.csv`.
  - `paper/tables/model_specific_var_es_garch.tex`.
  - `reports/distribution_specific_var_es_audit.md`.
  - GED/skewed/heavy-tail ES was not reported because the ES implementation and parameter extraction were not verified enough for formal results.
- Implemented a validation-selected regime-aware combination:
  - Experts: HAR-Parkinson and Calibrated-Hybrid-LogTarget-Small-isotonic.
  - Regime indicator: origin-available trailing 20-day volatility.
  - Selected threshold: validation 75th percentile.
  - Low-regime weights: 0.45 econometric / 0.55 neural.
  - High-regime weights: 0.00 econometric / 1.00 neural.
  - Test QLIKE: 1.064346.
  - Static baseline QLIKE: 1.077023.
  - The regime-aware combination improves over the static neural/econometric combination and improves spike behavior, but it does not dominate the strongest econometric models overall.
- Re-audited expanding monthly refit failures:
  - AdvGARCH-BestQLIKE has 20 failed segments over the expanded \(N=745\) window, all classified as ARIMA convergence failures.
  - Its 340-forecast incomplete refit result is separated from complete-window refit models.
- Added tests for completion logic in `tests/test_completion_extensions.py`.
- Final verification:
  - `PYTHONDONTWRITEBYTECODE=1 pytest -q`: 32 tests passed.
  - `PYTHONDONTWRITEBYTECODE=1 make final-artifacts`: completed successfully.
  - `make -C paper all`: completed successfully; Tectonic emitted underfull-box warnings but no undefined references, citation failures, overfull boxes, LaTeX errors, or emergency stops were detected in the final log check.
- Final PDFs:
  - `paper/main.pdf`.
  - `paper/final_vnindex_volatility_financial_econometrics_v3.pdf`.
