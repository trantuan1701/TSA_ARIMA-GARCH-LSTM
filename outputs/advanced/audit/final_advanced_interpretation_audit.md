# Final Advanced Interpretation Audit

## Scope

This audit records which results are safe to use as headline empirical claims in the paper. It does not modify predictions, metrics, or fitted-model outputs.

## Model Classification

- `validation_selected_primary`: GARCH(1,1), ARIMA-GARCH, and AdvGARCH-BestQLIKE are eligible for primary benchmark or primary advanced-search claims.
- `validation_selected_category`: AdvGARCH-BestAsymmetric, AdvGARCH-BestHeavyTail, Hybrid-QLIKE, and selected refit variants can support category-level claims.
- `validation_selected_combination`: ComboStack-PoolD_AllStable is the primary combination selected by validation QLIKE.
- `ex_post_test_best`: ComboPair-BestGARCHFamily_HybridQLIKE may be reported as the best observed test QLIKE, but only as exploratory.
- `exploratory_only`: raw log-target neural variants and proxy-specific winners such as RollingVol-20 under Yang-Zhang 20-day are diagnostic or robustness findings.

## Claim Safety Decisions

- Safe: GARCH-family models are the most reliable class; GARCH(1,1) is a strong benchmark; ARIMA mean dynamics add limited standalone value; neural and hybrid models do not robustly dominate econometric benchmarks; log-target neural variants underpredict spikes and risk; proxy choice matters.
- Needs caveat: asymmetric/EGARCH-type models are more robust than GARCH(1,1). This is supported by observed QLIKE and proxy rankings, but not by Holm-corrected superiority on the original common-window test.
- Unsafe overclaim: forecast combinations prove robust hybrid superiority. The validation-selected combination does not beat the best GARCH-family models, and the best-test combination must be treated as exploratory.

## Statistical Guardrails

- Raw QLIKE DM tests show improvements for several advanced GARCH-family variants against GARCH(1,1), including AdvGARCH-BestAsymmetric with p = 0.001534.
- No QLIKE improvement over GARCH(1,1) survives Holm correction at the 10 percent level in the advanced common-window comparison.
- The approximate MCS retains 68 of 73 models at both 90 percent and 95 percent confidence levels, so the paper must not claim a single decisive winner.

## Paper Wording Rules

- Use validation-selected models for primary claims.
- Label test-best models as exploratory unless they were selected before test evaluation.
- Report observed QLIKE improvements separately from multiplicity-adjusted statistical conclusions.
- Use "evidence suggests" and "is consistent with" for class-level conclusions.
- Do not state that GARCH(1,1) is always best or that neural/hybrid models are useless.
