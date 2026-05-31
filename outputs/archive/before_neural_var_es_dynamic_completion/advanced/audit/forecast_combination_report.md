# Forecast Combination Report

## Validation Context

- Best econometric model by validation QLIKE: `Refit-BestAsymmetric-rolling_1000_monthly`.
- Best selected GARCH-family model: `AdvGARCH-BestARIMA`.
- Best validation-selected refit model: `Refit-BestAsymmetric-rolling_1000_monthly`.
- Best calibrated neural model: `Calibrated-Hybrid-LogTarget-Small-isotonic`.

## Best Validation Combinations

- ComboStack-PoolD_AllStable: validation QLIKE 1.496227, method `multi_model_stacking`, pool `PoolD_AllStable`.
- ComboStack-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.500909, method `multi_model_stacking`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboPair-BestRefitGARCH_BestCalibratedNeural: validation QLIKE 1.508987, method `two_model_convex_grid`, pool `BestRefitGARCH_BestCalibratedNeural`.
- ComboPair-BestGARCHFamily_BestCalibratedNeural: validation QLIKE 1.510198, method `two_model_convex_grid`, pool `BestGARCHFamily_BestCalibratedNeural`.
- ComboMean-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.515070, method `multi_model_mean`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboMedian-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.529372, method `multi_model_median`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboMean-PoolD_AllStable: validation QLIKE 1.572826, method `multi_model_mean`, pool `PoolD_AllStable`.
- ComboStack-PoolB_GARCH_Neural: validation QLIKE 1.577371, method `multi_model_stacking`, pool `PoolB_GARCH_Neural`.
- ComboPair-BestRefitGARCH_HybridQLIKE: validation QLIKE 1.577375, method `two_model_convex_grid`, pool `BestRefitGARCH_HybridQLIKE`.
- ComboPair-BestGARCHFamily_HybridQLIKE: validation QLIKE 1.580104, method `two_model_convex_grid`, pool `BestGARCHFamily_HybridQLIKE`.

## Test Results For Validation-Fitted Combinations

- ComboPair-BestGARCHFamily_HybridQLIKE: test QLIKE 1.026265, RMSE 3.607900, MAE 1.431155.
- ComboPair-BestRefitGARCH_HybridQLIKE: test QLIKE 1.047182, RMSE 3.595016, MAE 1.491585.
- ComboMean-PoolA_Econometric: test QLIKE 1.047302, RMSE 3.590554, MAE 1.388504.
- ComboStack-PoolB_GARCH_Neural: test QLIKE 1.047676, RMSE 3.595564, MAE 1.493210.
- ComboMedian-PoolA_Econometric: test QLIKE 1.049812, RMSE 3.596279, MAE 1.383845.
- ComboStack-PoolA_Econometric: test QLIKE 1.050864, RMSE 3.553045, MAE 1.393250.
- ComboPair-GARCH11_HybridQLIKE: test QLIKE 1.058475, RMSE 3.641724, MAE 1.519432.
- ComboPair-GARCH11_ARIMAGARCH: test QLIKE 1.062958, RMSE 3.628230, MAE 1.407866.
- ComboMean-PoolC_GARCH_CalibratedNeural: test QLIKE 1.063831, RMSE 3.608997, MAE 1.535241.
- ComboMedian-PoolC_GARCH_CalibratedNeural: test QLIKE 1.071273, RMSE 3.637627, MAE 1.537095.

## Weight Fitting

- Pair weights and stacking weights are fitted on validation QLIKE only.
- Test data are never used to choose pair weights, stacking weights, pools, or calibrated inputs.

## Output Tables

- `outputs/advanced/tables/table_forecast_combination_validation.csv`
- `outputs/advanced/tables/table_forecast_combination_test.csv`
- `outputs/advanced/tables/table_forecast_combination_weights.csv`
