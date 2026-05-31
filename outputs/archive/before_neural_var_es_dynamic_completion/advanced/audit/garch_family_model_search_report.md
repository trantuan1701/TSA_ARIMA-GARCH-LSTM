# GARCH-Family Model Search Report

## Candidate Grid

- Direct `arch` mean models: Zero, Constant, and AR lags 1, 2, 3, and 5.
- Two-step mean models: ARIMA(p,0,q), p and q in {0,1,2,3}; residuals are passed to GARCH-family volatility models.
- Volatility models: ARCH, GARCH, GJR-GARCH, EGARCH, APARCH, and FIGARCH specifications requested for this stage.
- Distributions: Normal, Student-t, skewed Student-t, and GED.
- Selection uses validation QLIKE only; test QLIKE is reported after selection.

## Attempt Summary

- Attempted candidates: 3168.
- Successful fits: 2592.
- Failed fits: 576.

## Failures By Reason

- numerical error: 576

## Selected Models

- best_fixed_by_validation_qlike: `AdvGARCH-BestQLIKE` from `arima_3_0_3_egarch_1_1_2_normal` (ARIMA(3,0,3), EGARCH(1,1,2), normal), validation QLIKE=1.611304, test QLIKE=1.032548.
- best_symmetric_garch_by_validation_qlike: `AdvGARCH-BestSymmetric` from `direct_constant_garch_1_3_normal` (Constant, GARCH(1,3), normal), validation QLIKE=1.628556, test QLIKE=1.063460.
- best_asymmetric_garch_by_validation_qlike: `AdvGARCH-BestAsymmetric` from `direct_zero_egarch_1_1_1_normal` (Zero, EGARCH(1,1,1), normal), validation QLIKE=1.612272, test QLIKE=1.029992.
- best_heavy_tail_by_validation_qlike: `AdvGARCH-BestHeavyTail` from `arima_3_0_3_egarch_1_1_1_ged` (ARIMA(3,0,3), EGARCH(1,1,1), ged), validation QLIKE=1.611777, test QLIKE=1.031536.
- best_two_step_arima_garch_by_validation_qlike: `AdvGARCH-BestARIMA` from `arima_3_0_3_egarch_1_1_2_normal` (ARIMA(3,0,3), EGARCH(1,1,2), normal), validation QLIKE=1.611304, test QLIKE=1.032548.

## Test QLIKE Against Original GARCH(1,1)

- AdvGARCH-BestAsymmetric: test QLIKE 1.029992; delta vs original GARCH(1,1) = -0.032967.
- AdvGARCH-BestHeavyTail: test QLIKE 1.031536; delta vs original GARCH(1,1) = -0.031422.
- AdvGARCH-BestQLIKE: test QLIKE 1.032548; delta vs original GARCH(1,1) = -0.030410.
- AdvGARCH-BestARIMA: test QLIKE 1.032548; delta vs original GARCH(1,1) = -0.030410.
- AdvGARCH-BestSymmetric: test QLIKE 1.063460; delta vs original GARCH(1,1) = 0.000502.

## Interpretation

- Improvements below roughly 0.01 QLIKE should be treated as economically small unless supported by the advanced statistical tests.
- Positive variance clipping is used only inside metric computations and prediction-schema validation; non-positive raw model forecasts are recorded as failures.
- Forecasts are fixed-parameter unless a row is explicitly part of the refit protocol experiment.

## Output Tables

- `outputs/advanced/tables/table_garch_family_all_candidates.csv`
- `outputs/advanced/tables/table_garch_family_fit_failures.csv`
- `outputs/advanced/tables/table_garch_family_validation_ranking.csv`
- `outputs/advanced/tables/table_garch_family_selected_models.csv`
- `outputs/advanced/tables/table_garch_family_test_results.csv`
