# Advanced Experiment Summary

## Top Test QLIKE Models

- ComboPair-BestGARCHFamily_HybridQLIKE: test QLIKE 1.026265, validation QLIKE 1.580104, family `forecast_combination`.
- AdvGARCH-BestAsymmetric: test QLIKE 1.029992, validation QLIKE 1.612272, family `advanced_garch_family`.
- AdvGARCH-BestHeavyTail: test QLIKE 1.031536, validation QLIKE 1.611777, family `advanced_garch_family`.
- AdvGARCH-BestARIMA: test QLIKE 1.032548, validation QLIKE 1.611304, family `advanced_garch_family`.
- AdvGARCH-BestQLIKE: test QLIKE 1.032548, validation QLIKE 1.611304, family `advanced_garch_family`.
- Refit-BestGARCHFamily-fixed_train_only: test QLIKE 1.032548, validation QLIKE 1.611304, family `refit_garch`.
- ComboPair-BestRefitGARCH_HybridQLIKE: test QLIKE 1.047182, validation QLIKE 1.577375, family `forecast_combination`.
- ComboMean-PoolA_Econometric: test QLIKE 1.047302, validation QLIKE 1.611473, family `forecast_combination`.
- ComboStack-PoolB_GARCH_Neural: test QLIKE 1.047676, validation QLIKE 1.577371, family `forecast_combination`.
- ComboMedian-PoolA_Econometric: test QLIKE 1.049812, validation QLIKE 1.620988, family `forecast_combination`.

## Best Model

- Best test QLIKE: `ComboPair-BestGARCHFamily_HybridQLIKE` (1.026265).

## Hypothesis Summary

- H1: Mean dynamics add value beyond GARCH. Status: `not_supported`. ARIMA-GARCH validation/test QLIKE 1.636484/1.065995; GARCH 1.629432/1.062958; DM p=0.3995596304673623.
- H2: Heavy-tailed/asymmetric GARCH variants improve forecasts. Status: `supported`. Best advanced GARCH validation/test QLIKE 1.611304/1.032548; DM p=0.0181113653808202; Holm p=1.0.
- H3: Rolling/expanding refits improve forecasts. Status: `not_supported`. Best validation-selected refit model Refit-BestAsymmetric-rolling_1000_monthly; validation/test QLIKE 1.599196/1.067672; DM p=0.8489724206007725.
- H4: Hybrid/neural forecasts add incremental information. Status: `not_supported`. Best combo ComboStack-PoolD_AllStable validation/test QLIKE 1.496227/1.113530; best validation econometric Refit-BestAsymmetric-rolling_1000_monthly test QLIKE 1.067672; combo DM p=0.5073708506038361.
- H5: Log-target neural models mainly suffer from scale bias. Status: `partially_supported`. Average raw log-target test QLIKE 2.225569, best calibrated 1.146217; pred/actual ratio 0.273 -> 1.496.
- H6: Best QLIKE models improve risk management. Status: `inconclusive`. Closest 5% VaR coverage: ARIMA-GARCH violation rate 0.047; best test QLIKE model: ComboPair-BestGARCHFamily_HybridQLIKE.

## Original Output Integrity

- No existing output files outside `outputs/advanced/` changed during `run_advanced_experiments.py`.

## Output Tables

- `outputs/advanced/tables/table_advanced_final_leaderboard.csv`
- `outputs/advanced/tables/table_advanced_hypothesis_summary.csv`
- `outputs/advanced/tables/table_original_output_integrity.csv`
