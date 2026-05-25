# Advanced Hypothesis Tests Report

## Benchmarks

- `GARCH(1,1)`
- `AdvGARCH-BestQLIKE`
- `ComboStack-PoolD_AllStable`

## Loss Difference Convention

- `d_t = loss_model_t - loss_benchmark_t`.
- Positive mean differences mean the model is worse than the benchmark.
- Negative mean differences mean the model is better than the benchmark.

## QLIKE DM Results Against Original GARCH(1,1)

- AdvGARCH-BestAsymmetric: mean_diff -0.031476, p=0.00153435.
- AdvGARCH-BestHeavyTail: mean_diff -0.031617, p=0.017138.
- AdvGARCH-BestARIMA: mean_diff -0.029990, p=0.0181114.
- AdvGARCH-BestQLIKE: mean_diff -0.029990, p=0.0181114.
- Refit-BestGARCHFamily-fixed_train_only: mean_diff -0.029990, p=0.0181114.
- ComboMedian-PoolA_Econometric: mean_diff -0.012582, p=0.0779321.

## Holm Correction

- No QLIKE DM improvement over GARCH(1,1) survives Holm correction at the 10% level.

## Model Confidence Set

- MCS method: documented approximation using a moving-block bootstrap range statistic.
- Models retained in 90% MCS: AdvGARCH-BestHeavyTail, AdvGARCH-BestAsymmetric, AdvGARCH-BestQLIKE, AdvGARCH-BestARIMA, Refit-BestGARCHFamily-fixed_train_only, ComboPair-BestGARCHFamily_HybridQLIKE, ComboMean-PoolA_Econometric, ComboMedian-PoolA_Econometric, ComboStack-PoolA_Econometric, ComboPair-BestRefitGARCH_HybridQLIKE, ComboStack-PoolB_GARCH_Neural, Refit-OriginalGARCH-expanding_quarterly, Refit-OriginalARIMAGARCH-expanding_quarterly, ComboPair-GARCH11_ARIMAGARCH, GARCH(1,1), AdvGARCH-BestSymmetric, ARIMA-GARCH, ComboPair-GARCH11_HybridQLIKE, Refit-BestAsymmetric-rolling_1000_monthly, ComboMean-PoolC_GARCH_CalibratedNeural, ComboMedian-PoolC_GARCH_CalibratedNeural, Calibrated-LSTM-LogTarget-isotonic, ComboPair-BestRefitGARCH_BestCalibratedNeural, Calibrated-Hybrid-QLIKE-isotonic, ComboPair-BestGARCHFamily_BestCalibratedNeural, ComboStack-PoolC_GARCH_CalibratedNeural, ComboStack-PoolD_AllStable, ComboMean-PoolD_AllStable, Calibrated-LSTM-LogTarget-scale_mean, Calibrated-Hybrid-LogTarget-Small-isotonic, Calibrated-LSTM-LogTarget-affine_nonnegative, Calibrated-LSTM-LogTarget-scale_qlike, Calibrated-LSTM-isotonic, Calibrated-ARIMA-GARCH-LSTM-isotonic, ComboMean-PoolB_GARCH_Neural, Calibrated-Hybrid-LogTarget-Small-scale_qlike, Calibrated-Hybrid-LogTarget-Small-affine_nonnegative, Calibrated-Hybrid-LogTarget-Small-scale_mean, Calibrated-LSTM-QLIKE-isotonic, ComboMedian-PoolD_AllStable, LSTM, Calibrated-Hybrid-LogTarget-affine_nonnegative, Calibrated-Hybrid-LogTarget-isotonic, ARIMA-GARCH-LSTM, Calibrated-Hybrid-LogTarget-scale_qlike, ComboMedian-PoolB_GARCH_Neural, Calibrated-ARIMA-GARCH-LSTM-affine_nonnegative, Calibrated-ARIMA-GARCH-LSTM-scale_qlike, Calibrated-Hybrid-LogTarget-scale_mean, LSTM-QLIKE, HistoricalMean, Calibrated-LSTM-affine_nonnegative, Calibrated-LSTM-scale_qlike, Calibrated-ARIMA-GARCH-LSTM-scale_mean, Calibrated-LSTM-LogTarget-Small-isotonic, Calibrated-LSTM-scale_mean, Calibrated-LSTM-QLIKE-scale_qlike, Hybrid-QLIKE, Calibrated-LSTM-QLIKE-affine_nonnegative, Calibrated-Hybrid-QLIKE-scale_qlike, Calibrated-Hybrid-QLIKE-affine_nonnegative, Calibrated-LSTM-QLIKE-scale_mean, Calibrated-LSTM-LogTarget-Small-scale_qlike, Calibrated-LSTM-LogTarget-Small-affine_nonnegative, Calibrated-Hybrid-QLIKE-scale_mean, RollingVol-20, Calibrated-LSTM-LogTarget-Small-scale_mean, RollingVol-10.
- Models retained in 95% MCS: AdvGARCH-BestHeavyTail, AdvGARCH-BestAsymmetric, AdvGARCH-BestQLIKE, AdvGARCH-BestARIMA, Refit-BestGARCHFamily-fixed_train_only, ComboPair-BestGARCHFamily_HybridQLIKE, ComboMean-PoolA_Econometric, ComboMedian-PoolA_Econometric, ComboStack-PoolA_Econometric, ComboPair-BestRefitGARCH_HybridQLIKE, ComboStack-PoolB_GARCH_Neural, Refit-OriginalGARCH-expanding_quarterly, Refit-OriginalARIMAGARCH-expanding_quarterly, ComboPair-GARCH11_ARIMAGARCH, GARCH(1,1), AdvGARCH-BestSymmetric, ARIMA-GARCH, ComboPair-GARCH11_HybridQLIKE, Refit-BestAsymmetric-rolling_1000_monthly, ComboMean-PoolC_GARCH_CalibratedNeural, ComboMedian-PoolC_GARCH_CalibratedNeural, Calibrated-LSTM-LogTarget-isotonic, ComboPair-BestRefitGARCH_BestCalibratedNeural, Calibrated-Hybrid-QLIKE-isotonic, ComboPair-BestGARCHFamily_BestCalibratedNeural, ComboStack-PoolC_GARCH_CalibratedNeural, ComboStack-PoolD_AllStable, ComboMean-PoolD_AllStable, Calibrated-LSTM-LogTarget-scale_mean, Calibrated-Hybrid-LogTarget-Small-isotonic, Calibrated-LSTM-LogTarget-affine_nonnegative, Calibrated-LSTM-LogTarget-scale_qlike, Calibrated-LSTM-isotonic, Calibrated-ARIMA-GARCH-LSTM-isotonic, ComboMean-PoolB_GARCH_Neural, Calibrated-Hybrid-LogTarget-Small-scale_qlike, Calibrated-Hybrid-LogTarget-Small-affine_nonnegative, Calibrated-Hybrid-LogTarget-Small-scale_mean, Calibrated-LSTM-QLIKE-isotonic, ComboMedian-PoolD_AllStable, LSTM, Calibrated-Hybrid-LogTarget-affine_nonnegative, Calibrated-Hybrid-LogTarget-isotonic, ARIMA-GARCH-LSTM, Calibrated-Hybrid-LogTarget-scale_qlike, ComboMedian-PoolB_GARCH_Neural, Calibrated-ARIMA-GARCH-LSTM-affine_nonnegative, Calibrated-ARIMA-GARCH-LSTM-scale_qlike, Calibrated-Hybrid-LogTarget-scale_mean, LSTM-QLIKE, HistoricalMean, Calibrated-LSTM-affine_nonnegative, Calibrated-LSTM-scale_qlike, Calibrated-ARIMA-GARCH-LSTM-scale_mean, Calibrated-LSTM-LogTarget-Small-isotonic, Calibrated-LSTM-scale_mean, Calibrated-LSTM-QLIKE-scale_qlike, Hybrid-QLIKE, Calibrated-LSTM-QLIKE-affine_nonnegative, Calibrated-Hybrid-QLIKE-scale_qlike, Calibrated-Hybrid-QLIKE-affine_nonnegative, Calibrated-LSTM-QLIKE-scale_mean, Calibrated-LSTM-LogTarget-Small-scale_qlike, Calibrated-LSTM-LogTarget-Small-affine_nonnegative, Calibrated-Hybrid-QLIKE-scale_mean, RollingVol-20, Calibrated-LSTM-LogTarget-Small-scale_mean, RollingVol-10.

## Output Tables

- `outputs/advanced/tables/table_advanced_dm_tests_qlike.csv`
- `outputs/advanced/tables/table_advanced_dm_tests_squared_error.csv`
- `outputs/advanced/tables/table_advanced_dm_tests_absolute_error.csv`
- `outputs/advanced/tables/table_advanced_wilcoxon_tests.csv`
- `outputs/advanced/tables/table_advanced_holm_corrected_tests.csv`
- `outputs/advanced/tables/table_model_confidence_set_qlike.csv`
