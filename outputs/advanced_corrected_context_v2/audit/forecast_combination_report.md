# Forecast Combination Report

## Validation Context

- Best econometric model by validation QLIKE: `Refit-BestAsymmetric-rolling_1000_monthly`.
- Best selected GARCH-family model: `AdvGARCH-BestARIMA`.
- Best validation-selected refit model: `Refit-BestAsymmetric-rolling_1000_monthly`.
- Best calibrated neural model: `Calibrated-Hybrid-LogTarget-Small-isotonic`.

## Best Validation Combinations

- ComboStack-PoolD_AllStable: validation QLIKE 1.496031, method `multi_model_stacking`, pool `PoolD_AllStable`.
- ComboStack-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.506806, method `multi_model_stacking`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboPair-BestRefitGARCH_BestCalibratedNeural: validation QLIKE 1.506997, method `two_model_convex_grid`, pool `BestRefitGARCH_BestCalibratedNeural`.
- ComboPair-BestGARCHFamily_BestCalibratedNeural: validation QLIKE 1.507740, method `two_model_convex_grid`, pool `BestGARCHFamily_BestCalibratedNeural`.
- ComboMean-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.520814, method `multi_model_mean`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboMedian-PoolC_GARCH_CalibratedNeural: validation QLIKE 1.529314, method `multi_model_median`, pool `PoolC_GARCH_CalibratedNeural`.
- ComboMean-PoolD_AllStable: validation QLIKE 1.573401, method `multi_model_mean`, pool `PoolD_AllStable`.
- ComboPair-BestGARCHFamily_HybridQLIKE: validation QLIKE 1.581446, method `two_model_convex_grid`, pool `BestGARCHFamily_HybridQLIKE`.
- ComboStack-PoolB_GARCH_Neural: validation QLIKE 1.582335, method `multi_model_stacking`, pool `PoolB_GARCH_Neural`.
- ComboPair-BestRefitGARCH_HybridQLIKE: validation QLIKE 1.582337, method `two_model_convex_grid`, pool `BestRefitGARCH_HybridQLIKE`.

## Test Results For Validation-Fitted Combinations

- ComboPair-BestGARCHFamily_HybridQLIKE: test QLIKE 1.032592, RMSE 3.586221, MAE 1.431935.
- ComboMean-PoolA_Econometric: test QLIKE 1.047302, RMSE 3.590554, MAE 1.388504.
- ComboMedian-PoolA_Econometric: test QLIKE 1.049812, RMSE 3.596279, MAE 1.383845.
- ComboStack-PoolA_Econometric: test QLIKE 1.050864, RMSE 3.553045, MAE 1.393250.
- ComboPair-BestRefitGARCH_HybridQLIKE: test QLIKE 1.053819, RMSE 3.574704, MAE 1.497080.
- ComboStack-PoolB_GARCH_Neural: test QLIKE 1.054082, RMSE 3.575021, MAE 1.497933.
- ComboPair-GARCH11_ARIMAGARCH: test QLIKE 1.062958, RMSE 3.628230, MAE 1.407866.
- ComboPair-GARCH11_HybridQLIKE: test QLIKE 1.065003, RMSE 3.621514, MAE 1.523118.
- ComboMean-PoolC_GARCH_CalibratedNeural: test QLIKE 1.081746, RMSE 3.605492, MAE 1.555079.
- ComboMedian-PoolC_GARCH_CalibratedNeural: test QLIKE 1.089254, RMSE 3.612102, MAE 1.562294.

## Weight Fitting

- Pair weights and stacking weights are fitted on validation QLIKE only.
- Test data are never used to choose pair weights, stacking weights, pools, or calibrated inputs.

## Output Tables

- `outputs/advanced/tables/table_forecast_combination_validation.csv`
- `outputs/advanced/tables/table_forecast_combination_test.csv`
- `outputs/advanced/tables/table_forecast_combination_weights.csv`
