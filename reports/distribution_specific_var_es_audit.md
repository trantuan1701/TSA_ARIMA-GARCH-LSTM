# Distribution-Specific VaR/ES Audit

- Common-Normal VaR/ES uses zero-mean lower-tail return quantiles for every primary variance forecast.
- Model-specific VaR/ES is computed only for selected GARCH-family specifications with Normal innovations, where analytical VaR and ES are verified by tests.
- `GARCH(1,1)`, `AdvGARCH-BestQLIKE`, and `AdvGARCH-BestAsymmetric` are included under Normal innovations.
- `AdvGARCH-BestHeavyTail` is validation-selected as a GED heavy-tail comparator, but GED ES is not reported because the GED ES implementation and parameter extraction were not verified in this pass.
- Skewed-t ES is not implemented; no skewed-t model is promoted to primary risk evidence.
- ES diagnostics are exceedance-mean diagnostics, not formal ES backtests.

- Model-specific rows written: 9.
