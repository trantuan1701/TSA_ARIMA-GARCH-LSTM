# VaR Backtesting Report

## VaR Construction

- One-day Normal VaR is computed as `VaR_alpha_t = z_alpha * sqrt(pred_var_t)`.
- Returns and volatility predictions are both in percent-return units.
- Violations occur when the target-date log return is below the VaR forecast.

## Coverage Results

### 5% VaR
- Hybrid-QLIKE: violations 18/726 (0.025); Kupiec p=0.00058307.
- ComboStack-PoolD_AllStable: violations 18/726 (0.025); Kupiec p=0.00058307.
- Calibrated-Hybrid-LogTarget-Small-isotonic: violations 21/726 (0.029); Kupiec p=0.00480735.
- Refit-BestAsymmetric-rolling_1000_monthly: violations 31/745 (0.042); Kupiec p=0.27985.
- AdvGARCH-BestQLIKE: violations 34/745 (0.046); Kupiec p=0.579475.
- GARCH(1,1): violations 35/745 (0.047); Kupiec p=0.70251.
- ARIMA-GARCH: violations 35/745 (0.047); Kupiec p=0.70251.
- LSTM-LogTarget: violations 87/726 (0.120); Kupiec p=1.54381e-13.

### 1% VaR
- ComboStack-PoolD_AllStable: violations 10/726 (0.014); Kupiec p=0.33368.
- Hybrid-QLIKE: violations 10/726 (0.014); Kupiec p=0.33368.
- Calibrated-Hybrid-LogTarget-Small-isotonic: violations 12/726 (0.017); Kupiec p=0.10606.
- Refit-BestAsymmetric-rolling_1000_monthly: violations 13/745 (0.017); Kupiec p=0.0645287.
- AdvGARCH-BestQLIKE: violations 15/745 (0.020); Kupiec p=0.0145295.
- ARIMA-GARCH: violations 16/745 (0.021); Kupiec p=0.00631029.
- GARCH(1,1): violations 16/745 (0.021); Kupiec p=0.00631029.
- LSTM-LogTarget: violations 56/726 (0.077); Kupiec p=3.81053e-31.

## Interpretation

- Low-MAE models with severe variance underprediction should show excessive VaR violations if their volatility scale is too low.
- Kupiec tests assess unconditional coverage; Christoffersen tests add violation independence and conditional coverage when transition counts are informative.

## Output Tables

- `outputs/advanced/tables/table_var_backtesting_5pct.csv`
- `outputs/advanced/tables/table_var_backtesting_1pct.csv`
