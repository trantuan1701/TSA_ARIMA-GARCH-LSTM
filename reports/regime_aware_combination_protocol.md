# Regime-Aware Combination Protocol

- Econometric expert: `HAR-Parkinson` selected by validation QLIKE from stable experts.
- Neural expert: `Calibrated-Hybrid-LogTarget-Small-isotonic` selected by validation QLIKE from corrected-context neural/calibrated candidates.
- Regime indicator: origin-observable `rolling_vol_20` from the model-ready data.
- Threshold grid: 75th, 85th, and 90th validation-period quantiles of the origin signal.
- Weights: convex low/high-regime weights selected by validation QLIKE only and frozen for test evaluation.
- Leakage control: no test losses, test actual variances, or future target values enter expert, threshold, or weight selection.

## Frozen Test Evidence

- Static combination test QLIKE: 1.077023.
- Regime-aware combination test QLIKE: 1.064346.
- Regime threshold quantile: 0.75; value 1.673120.
- Low-regime econometric weight: 0.45; high-regime econometric weight: 0.00.

## Validation Candidate Grid

- q=0.75: validation QLIKE 1.492900, low/high econ weights 0.45/0.00.
- q=0.85: validation QLIKE 1.494764, low/high econ weights 0.40/0.00.
- q=0.90: validation QLIKE 1.495953, low/high econ weights 0.37/0.00.
