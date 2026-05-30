# Stage 2 Report Revision Changelog

## Objective

Revise the report for "Modeling and Forecasting VN-Index Volatility: A Robust Empirical Comparison of GARCH-Family, Neural, and Hybrid Approaches" so that it is framed as a validation-selected, proxy-robust, and risk-aware empirical benchmark rather than a loose model comparison.

## Files Modified

- `paper/main.tex`
- `paper/references.bib`
- `paper/sections/01_introduction.tex`
- `paper/sections/02_related_work.tex`
- `paper/sections/03_methodology.tex`
- `paper/sections/04_experimental_setup.tex`
- `paper/sections/05_results_and_discussion.tex`
- `paper/sections/06_conclusion.tex`
- `paper/tables/proxy_robustness_rank_summary.tex`
- `paper/tables/var_backtesting_summary.tex`

## Files Added

- `paper/tables/stage1_benchmark_extensions.tex`
- `paper/tables/advanced_search_audit_summary.tex`
- `paper/tables/stage1_risk_diagnostics_summary.tex`
- `paper/figures/fig_stage1_regime_qlike_comparison.png`
- `paper/figures/fig_stage1_spike_underprediction_comparison.png`
- `paper/figures/fig_stage1_proxy_rank_heatmap_top_models.png`
- `paper/figures/fig_stage1_proxy_correlation_heatmap.png`
- `paper/figures/fig_stage1_var_exceedances_garch_1_1.png`
- `paper/figures/fig_stage1_neural_spike_failure.png`
- `paper/main_stage2.pdf`
- `reports/stage1_extensions/report_revision_stage2.md`

## Sections Rewritten

- Abstract: reframed around validation-selected, proxy-robust, risk-aware benchmarking.
- Introduction and contributions: rewritten into a four-contribution structure.
- Related Work: added EWMA/RiskMetrics and HAR context.
- Data and Forecasting Target: clarified percentage log-return scale, percentage-squared target, and proxy caveats.
- Models: added EWMA/RiskMetrics and HAR-type baseline subsections.
- Experimental Protocol: added Stage 1 leakage/test audit, validation-selection rules, and diagnostic protocol.
- Results: restructured into original benchmark results, EWMA/HAR benchmark completeness, advanced GARCH audit, regime/spike diagnostics, extended VaR, proxy robustness, and neural/hybrid diagnostics.
- Discussion: rewritten around GARCH-family reliability, HAR-Parkinson importance, neural/hybrid interpretation, proxy robustness, and economic risk interpretation.
- Limitations: updated with Stage 1 limitations and numerical search failures.
- Conclusion: rewritten around the new conservative benchmark narrative.

## Stage 1 Artifacts Used

- `reports/stage1_extensions/repo_audit_stage1.md`
- `outputs/stage1_extensions/run_manifest.json`
- `outputs/stage1_extensions/metrics/ewma_metrics.csv`
- `outputs/stage1_extensions/metrics/har_metrics.csv`
- `outputs/stage1_extensions/metrics/advanced_search_audit.csv`
- `outputs/stage1_extensions/metrics/advanced_search_failures_summary.csv`
- `outputs/stage1_extensions/metrics/regime_metrics.csv`
- `outputs/stage1_extensions/metrics/spike_diagnostics.csv`
- `outputs/stage1_extensions/metrics/var_backtests_extended.csv`
- `outputs/stage1_extensions/metrics/proxy_correlations.csv`
- `outputs/stage1_extensions/metrics/proxy_rank_stability.csv`
- `outputs/stage1_extensions/metrics/neural_diagnostics.csv`
- `outputs/stage1_extensions/figures/*.png`

## Important Numbers Integrated

- EWMA/RiskMetrics selected by validation QLIKE: `lambda = 0.90`.
- EWMA validation QLIKE: `1.687209`; test QLIKE: `1.225435`.
- HAR-Parkinson validation QLIKE: `1.558162`; test QLIKE: `1.034521`.
- Advanced GARCH search attempted candidates: `3,168`.
- Advanced GARCH successful fits: `2,592`.
- Advanced GARCH failed fits: `576`.
- Primary advanced GARCH validation-selected model: `AdvGARCH-BestQLIKE`.
- AdvGARCH-BestQLIKE test QLIKE: `1.032548`.
- AdvGARCH-BestAsymmetric standalone/category test QLIKE: `1.029992`, labeled exploratory.
- No advanced GARCH QLIKE improvement survives Holm correction at the `10%` level.
- Approximate MCS retains `68/73` models at both 90% and 95%.
- GARCH(1,1) 1% common-Normal VaR: `16/745` violations, Kupiec `p = 0.00631`.
- AdvGARCH-BestAsymmetric Stage 1 proxy-rank average: `3.57`, top-5 under `6/7` proxies.
- GARCH(1,1) Stage 1 proxy-rank average: `7.71`, top-5 under `2/7` proxies.
- Raw log-target neural average QLIKE: `2.225569`; best calibrated log-target QLIKE: `1.146217`.
- LSTM-LogTarget 1% VaR: `56/726` violations, Kupiec `p = 3.81e-31`.
- Hybrid-LogTarget-Small 1% VaR: `48/726` violations, Kupiec `p = 5.02e-24`.
- Stage 1 tests: `pytest: PASS (22 passed)` and `make stage1-test: PASS (22 passed)`.

## Tables and Figures Added to the Paper

New or updated tables:

- `Table III`: Stage 1 benchmark-completeness additions for EWMA and HAR.
- `Table IV`: Advanced GARCH search audit.
- `Table VII`: Stage 1 regime, spike, and 1% VaR diagnostic summary.
- `Table VIII`: Extended common-Normal VaR backtesting at 5%, 2.5%, and 1%.
- `Table IX`: Stage 1 proxy-rank stability summary.

New or updated figures:

- Stage 1 proxy-correlation heatmap.
- Stage 1 regime QLIKE comparison.
- Stage 1 spike-underprediction comparison.
- Stage 1 GARCH(1,1) VaR exceedance figure.
- Stage 1 proxy-rank heatmap for top models.
- Stage 1 neural spike-failure diagnostic.

## References

Added verified bibliography entries:

- `riskmetrics1996`: J.P. Morgan/Reuters, RiskMetrics Technical Document, 4th edition, 1996.
- `corsi2009har`: Fulvio Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility," Journal of Financial Econometrics, 2009.

## Compilation

Attempted command:

```bash
make fallback
```

Status: failed because `pdflatex` is not installed in the environment.

Successful command:

```bash
make tectonic
```

Status: completed and generated `paper/main.pdf`.

Stage 2 PDF copy:

```bash
cp main.pdf main_stage2.pdf
```

Output PDF:

- `paper/main.pdf`
- `paper/main_stage2.pdf`

## Compilation Warnings

- No unresolved citation or reference warnings found in `paper/main.log`.
- No remaining overfull hbox warning after resizing the Stage 1 risk diagnostics table.
- Remaining warnings are mainly underfull hboxes from narrow IEEE table columns and a small overfull vbox (`5.07219pt`) related to float placement.
- Tectonic reports an internal rerun warning because it keeps seeing `main.bbl` as changed, but it still writes the PDF successfully.
- Tectonic substitutes some unavailable Times font shapes in this environment.

## Manual Review Still Recommended

- Visually inspect the 12-page PDF for float placement because several Stage 1 tables and figures are large.
- Check whether the course/report page budget permits the expanded Stage 2 evidence; if not, move detailed diagnostic tables or figures to an appendix.
- Confirm whether the updated proxy-rank average `3.57` should replace the older paper value `3.29` everywhere; Stage 2 uses the current Stage 1 artifact.
- If a full TeX Live installation is available, optionally recompile with `pdflatex/bibtex/pdflatex/pdflatex` to compare output with Tectonic.
