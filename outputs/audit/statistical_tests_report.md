# Statistical Tests Report

## Common Test Window

- Size: 726 observations.
- Target-date range: 2023-02-07 to 2025-12-31.
- Constructed as the exact intersection of `(date, target_date)` keys across all test predictions.

## Benchmark

- Benchmark model: GARCH(1,1).

## Loss Difference Convention

- `d_t = loss_model_t - loss_GARCH(1,1)_t`.
- Positive `mean_diff` means the compared model is worse than GARCH(1,1).
- Negative `mean_diff` means the compared model is better than GARCH(1,1).
- DM statistics use a Bartlett/Newey-West long-run variance estimate with lag floor(n^(1/3)).

## Compared Models

- HistoricalMean
- RollingVol-5
- RollingVol-10
- RollingVol-20
- ARIMA-GARCH
- LSTM
- ARIMA-GARCH-LSTM
- LSTM-LogTarget
- Hybrid-LogTarget
- LSTM-QLIKE
- Hybrid-QLIKE
- LSTM-LogTarget-Small
- Hybrid-LogTarget-Small

## Skipped Models

None.

## QLIKE Significant Comparisons

- HistoricalMean: mean_diff=0.139371, p=0.0833395, winner=GARCH(1,1)
- Hybrid-QLIKE: mean_diff=0.177620, p=0.0805879, winner=GARCH(1,1)
- RollingVol-20: mean_diff=0.227747, p=0.0661459, winner=GARCH(1,1)
- RollingVol-10: mean_diff=0.257076, p=0.0616453, winner=GARCH(1,1)
- RollingVol-5: mean_diff=0.875121, p=8.39546e-06, winner=GARCH(1,1)
- Hybrid-LogTarget: mean_diff=1.033471, p=0.000114607, winner=GARCH(1,1)
- Hybrid-LogTarget-Small: mean_diff=1.180029, p=0.000217555, winner=GARCH(1,1)
- LSTM-LogTarget-Small: mean_diff=1.228610, p=0.00192776, winner=GARCH(1,1)
- LSTM-LogTarget: mean_diff=1.248613, p=1.26798e-05, winner=GARCH(1,1)

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

## Output Tables

- `outputs/tables/table_dm_tests_qlike.csv`
- `outputs/tables/table_dm_tests_squared_error.csv`
- `outputs/tables/table_dm_tests_absolute_error.csv`
