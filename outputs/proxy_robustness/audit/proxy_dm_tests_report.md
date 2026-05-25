# Proxy DM Tests Report

- Tests use QLIKE loss on the proxy-specific common test rows for each model/benchmark pair.
- Loss difference convention: `loss_model_t - loss_benchmark_t`; positive means the model is worse.
- Holm correction is applied within each proxy and benchmark family.

## Raw Significant Improvements Versus GARCH(1,1)

- close_to_close_squared_return: AdvGARCH-BestAsymmetric mean_diff=-0.033363, p=0.00114382.
- close_to_close_squared_return: AdvGARCH-BestARIMA mean_diff=-0.030801, p=0.0171379.
- close_to_close_squared_return: AdvGARCH-BestQLIKE mean_diff=-0.030801, p=0.0171379.
- close_to_close_squared_return: Refit-BestGARCHFamily-fixed_train_only mean_diff=-0.030801, p=0.0171379.
- close_to_close_squared_return: AdvGARCH-BestHeavyTail mean_diff=-0.031846, p=0.019309.
- close_to_close_squared_return: ComboMedian-PoolA_Econometric mean_diff=-0.013268, p=0.0594187.
- garman_klass: AdvGARCH-BestAsymmetric mean_diff=-0.024301, p=0.000828055.
- garman_klass: AdvGARCH-BestARIMA mean_diff=-0.020529, p=0.0102261.
- garman_klass: AdvGARCH-BestQLIKE mean_diff=-0.020529, p=0.0102261.
- garman_klass: Refit-BestGARCHFamily-fixed_train_only mean_diff=-0.020529, p=0.0102261.
- garman_klass: AdvGARCH-BestHeavyTail mean_diff=-0.021535, p=0.0104622.
- parkinson: AdvGARCH-BestAsymmetric mean_diff=-0.026319, p=0.000588175.
- parkinson: AdvGARCH-BestHeavyTail mean_diff=-0.024048, p=0.00659385.
- parkinson: AdvGARCH-BestARIMA mean_diff=-0.023066, p=0.00666965.
- parkinson: AdvGARCH-BestQLIKE mean_diff=-0.023066, p=0.00666965.
- parkinson: Refit-BestGARCHFamily-fixed_train_only mean_diff=-0.023066, p=0.00666965.
- rogers_satchell: AdvGARCH-BestAsymmetric mean_diff=-0.019791, p=0.00437848.
- rogers_satchell: AdvGARCH-BestARIMA mean_diff=-0.015273, p=0.054777.
- rogers_satchell: AdvGARCH-BestQLIKE mean_diff=-0.015273, p=0.054777.
- rogers_satchell: Refit-BestGARCHFamily-fixed_train_only mean_diff=-0.015273, p=0.054777.
- rogers_satchell: AdvGARCH-BestHeavyTail mean_diff=-0.016423, p=0.0567107.
- yang_zhang_10: ComboPair-GARCH11_ARIMAGARCH mean_diff=-0.000000, p=0.0893605.
- yang_zhang_20: AdvGARCH-BestSymmetric mean_diff=-0.001718, p=0.0433341.
- yang_zhang_5: AdvGARCH-BestAsymmetric mean_diff=-0.010187, p=0.0464267.

## Holm-Significant Improvements Versus GARCH(1,1)

- close_to_close_squared_return: AdvGARCH-BestAsymmetric Holm p=0.07778.
- garman_klass: AdvGARCH-BestAsymmetric Holm p=0.0140769.
- parkinson: AdvGARCH-BestAsymmetric Holm p=0.00823446.
- parkinson: AdvGARCH-BestARIMA Holm p=0.08572.
- parkinson: AdvGARCH-BestHeavyTail Holm p=0.08572.
- parkinson: AdvGARCH-BestQLIKE Holm p=0.08572.
- parkinson: Refit-BestGARCHFamily-fixed_train_only Holm p=0.08572.
- rogers_satchell: AdvGARCH-BestAsymmetric Holm p=0.0569203.
