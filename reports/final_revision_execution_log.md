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
