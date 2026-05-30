# Manuscript Revision Plan After Final Evidence Audit

## Required Claim Replacements

- Replace the current primary leaderboard with `paper/tables/common_window_master_results.tex`.
- Replace any statement that compares 745-row econometric results directly to 726-row neural results with the common-window result from `outputs/final/common_window_master_results.csv`.
- Replace the old advanced-search failure wording. The corrected taxonomy is:
  - 432 invalid forecast-start failures for AR-mean lag specifications.
  - 144 non-finite forecast failures.
- Replace 1% GARCH(1,1) VaR wording based on 745 rows with the final common-window result: 15 violations out of 726, expected 7.26, Kupiec p-value 0.011582.
- Do not state that no advanced GARCH result survives Holm correction without specifying the comparison family. In the final primary-model QLIKE family, AdvGARCH-BestAsymmetric vs GARCH(1,1) has Holm-adjusted p-value 0.021481; AdvGARCH-BestQLIKE does not.

## Section-Level Edits

- Introduction: frame the paper as a reproducible, proxy-aware, risk-aware VN-Index volatility forecasting study rather than a hybrid-model contribution.
- Data and Econometric Motivation: add `paper/tables/return_descriptive_statistics.tex`, `paper/tables/volatility_clustering_diagnostics.tex`, `fig_return_descriptive_diagnostics.png`, and `fig_return_acf_diagnostics.png`.
- Models: keep model descriptions concise; clearly label AdvGARCH-BestAsymmetric as category-selected robustness, not the single primary advanced winner.
- Experimental Protocol: define the final primary common window as target dates 2023-02-07 to 2025-12-31, N=726.
- Results: use `common_window_master_results.tex` as the only authoritative main leaderboard; move old full-window and test-best tables to appendix or remove.
- Risk Evaluation: describe VaR as common-Normal variance-to-risk mapping, not distribution-specific VaR.
- Discussion/Conclusion: state class-level conclusions conservatively: selected GARCH-family models and HAR-Parkinson form the strongest group; implemented neural/hybrid models do not robustly dominate under this daily-data protocol.

## Tables And Figures To Insert

- Main results: `paper/tables/common_window_master_results.tex`.
- Failure audit: `paper/tables/advanced_garch_failure_audit.tex`.
- Descriptive stats: `paper/tables/return_descriptive_statistics.tex`.
- Volatility clustering: `paper/tables/volatility_clustering_diagnostics.tex`.
- Main QLIKE figure: `paper/figures/fig_common_window_master_qlike.png`.
- Daily proxy rank heatmap: `paper/figures/fig_daily_proxy_rank_heatmap_primary.png`.

## Results That Must Move To Appendix Or Be Marked Exploratory

- Test-best forecast combinations.
- Full advanced-search rankings.
- Available-window 745-row metrics when neural models are included elsewhere.
- Broad calibration grids and all neural variant lists.
- Rolling Yang-Zhang sensitivity as a separate proxy-sensitivity exercise.

## Remaining Limitation To State

Neural sequence construction has been patched so future reruns can use leakage-safe pre-test context, but the checked-in neural prediction artifacts were not retrained in this pass. Therefore the current final primary window remains the 726-row intersection across existing neural and econometric predictions.
