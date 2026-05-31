# Robustness Analysis Report

## Common Window Construction

- Loaded all existing prediction CSV files under `outputs/predictions/`.
- Validated the required prediction schema: `date,target_date,actual_var,pred_var,model,split`.
- Used the exact intersection of test-split `(date, target_date)` keys across all models.
- Common test window: 726 observations, target dates 2023-02-07 to 2025-12-31.

## Yearly Split Sizes

| Period | Observations |
|---|---:|
| 2023 | 229 |
| 2024 | 249 |
| 2025 | 248 |
| full_common_test | 726 |

## Volatility Regime Thresholds

- calm: actual_var <= median = 0.286179365807.
- high-volatility: actual_var > 75th percentile = 0.996433725039.
- extreme-volatility: actual_var > 90th percentile = 2.37571006833.
- High-volatility and extreme-volatility regimes intentionally overlap.

## Underprediction Definition

- `spike_underprediction_rate` is the proportion of extreme-volatility observations where `pred_var < actual_var`.

## Key Findings

- Best full-window QLIKE: GARCH(1,1) (1.052888).
- GARCH(1,1) full-window QLIKE rank: 1 of 14.
- Best extreme-volatility QLIKE: Hybrid-QLIKE (4.570475).
- Log-target neural variants have an average extreme-spike underprediction rate of 1.000; the highest is LSTM-LogTarget at 1.000.

## Input Files Used

- `outputs/predictions/lstm_tuned/pred_hybrid_logtarget.csv`
- `outputs/predictions/lstm_tuned/pred_hybrid_logtarget_small.csv`
- `outputs/predictions/lstm_tuned/pred_hybrid_qlike.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_logtarget.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_logtarget_small.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_qlike.csv`
- `outputs/predictions/pred_arima_garch.csv`
- `outputs/predictions/pred_baseline_mean.csv`
- `outputs/predictions/pred_garch_11.csv`
- `outputs/predictions/pred_lstm_base.csv`
- `outputs/predictions/pred_lstm_hybrid.csv`
- `outputs/predictions/pred_rolling_vol_10.csv`
- `outputs/predictions/pred_rolling_vol_20.csv`
- `outputs/predictions/pred_rolling_vol_5.csv`

## Output Files

- `outputs/tables/table_yearly_results.csv`
- `outputs/tables/table_regime_results.csv`
- `outputs/tables/table_underprediction_analysis.csv`
- `outputs/figures/fig_loss_difference_vs_garch.png`
- `outputs/figures/fig_regime_qlike.png`
