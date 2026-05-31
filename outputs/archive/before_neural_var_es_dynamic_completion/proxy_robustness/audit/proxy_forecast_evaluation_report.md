# Proxy Forecast Evaluation Report

- Existing forecasts are read-only inputs; `pred_var` is unchanged.
- `actual_var` is replaced only inside this evaluation by the proxy value on `target_date`.
- Metrics use the same QLIKE epsilon clipping as the existing project.

## Top QLIKE Models By Proxy

### close_to_close_squared_return
- ComboPair-BestGARCHFamily_HybridQLIKE: QLIKE=1.025716, rank=1, pred/actual=1.098.
- AdvGARCH-BestAsymmetric: QLIKE=1.029543, rank=2, pred/actual=0.939.
- AdvGARCH-BestHeavyTail: QLIKE=1.031060, rank=3, pred/actual=0.953.
- AdvGARCH-BestARIMA: QLIKE=1.032104, rank=4, pred/actual=0.936.
- AdvGARCH-BestQLIKE: QLIKE=1.032104, rank=4, pred/actual=0.936.

### garman_klass
- AdvGARCH-BestAsymmetric: QLIKE=0.568148, rank=1, pred/actual=1.563.
- AdvGARCH-BestHeavyTail: QLIKE=0.570914, rank=2, pred/actual=1.588.
- AdvGARCH-BestARIMA: QLIKE=0.571920, rank=3, pred/actual=1.559.
- AdvGARCH-BestQLIKE: QLIKE=0.571920, rank=3, pred/actual=1.559.
- Refit-BestGARCHFamily-fixed_train_only: QLIKE=0.571920, rank=3, pred/actual=1.559.

### parkinson
- AdvGARCH-BestAsymmetric: QLIKE=0.628968, rank=1, pred/actual=1.457.
- AdvGARCH-BestHeavyTail: QLIKE=0.631239, rank=2, pred/actual=1.480.
- AdvGARCH-BestARIMA: QLIKE=0.632221, rank=3, pred/actual=1.453.
- AdvGARCH-BestQLIKE: QLIKE=0.632221, rank=3, pred/actual=1.453.
- Refit-BestGARCHFamily-fixed_train_only: QLIKE=0.632221, rank=3, pred/actual=1.453.

### rogers_satchell
- AdvGARCH-BestAsymmetric: QLIKE=0.573488, rank=1, pred/actual=1.483.
- AdvGARCH-BestHeavyTail: QLIKE=0.576856, rank=2, pred/actual=1.507.
- AdvGARCH-BestARIMA: QLIKE=0.578006, rank=3, pred/actual=1.479.
- AdvGARCH-BestQLIKE: QLIKE=0.578006, rank=3, pred/actual=1.479.
- Refit-BestGARCHFamily-fixed_train_only: QLIKE=0.578006, rank=3, pred/actual=1.479.

### yang_zhang_10
- ComboPair-GARCH11_ARIMAGARCH: QLIKE=0.740668, rank=1, pred/actual=1.143.
- GARCH(1,1): QLIKE=0.740668, rank=1, pred/actual=1.143.
- AdvGARCH-BestSymmetric: QLIKE=0.740971, rank=3, pred/actual=1.143.
- AdvGARCH-BestAsymmetric: QLIKE=0.741653, rank=4, pred/actual=1.034.
- ARIMA-GARCH: QLIKE=0.741938, rank=5, pred/actual=1.138.

### yang_zhang_20
- RollingVol-20: QLIKE=0.779266, rank=1, pred/actual=1.115.
- AdvGARCH-BestSymmetric: QLIKE=0.804088, rank=2, pred/actual=1.139.
- ARIMA-GARCH: QLIKE=0.805338, rank=3, pred/actual=1.135.
- ComboPair-GARCH11_ARIMAGARCH: QLIKE=0.805806, rank=4, pred/actual=1.139.
- GARCH(1,1): QLIKE=0.805806, rank=4, pred/actual=1.139.

### yang_zhang_5
- AdvGARCH-BestAsymmetric: QLIKE=0.720831, rank=1, pred/actual=1.039.
- AdvGARCH-BestHeavyTail: QLIKE=0.720947, rank=2, pred/actual=1.055.
- AdvGARCH-BestARIMA: QLIKE=0.724635, rank=3, pred/actual=1.036.
- AdvGARCH-BestQLIKE: QLIKE=0.724635, rank=3, pred/actual=1.036.
- Refit-BestGARCHFamily-fixed_train_only: QLIKE=0.724635, rank=3, pred/actual=1.036.

## Summary Answers

- GARCH(1,1) average QLIKE rank: 7.00.
- AdvGARCH-BestQLIKE average QLIKE rank: 4.71.
- Best neural/hybrid rank by proxy averages: 15.14.
- Raw log-target models remain underpredictors when pred/actual ratios stay materially below one and spike underprediction remains high.
