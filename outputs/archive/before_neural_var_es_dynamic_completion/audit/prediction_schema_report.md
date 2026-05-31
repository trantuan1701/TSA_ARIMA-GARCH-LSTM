# Prediction Schema Audit

Status: **PASS**

| File | Rows | Models | Splits |
|---|---:|---|---|
| `outputs/predictions/pred_arima_garch.csv` | 1495 | ARIMA-GARCH | test, validation |
| `outputs/predictions/pred_baseline_mean.csv` | 1495 | HistoricalMean | test, validation |
| `outputs/predictions/pred_garch_11.csv` | 1495 | GARCH(1,1) | test, validation |
| `outputs/predictions/pred_lstm_base.csv` | 1457 | LSTM | test, validation |
| `outputs/predictions/pred_lstm_hybrid.csv` | 1457 | ARIMA-GARCH-LSTM | test, validation |
| `outputs/predictions/pred_rolling_vol_10.csv` | 1495 | RollingVol-10 | test, validation |
| `outputs/predictions/pred_rolling_vol_20.csv` | 1495 | RollingVol-20 | test, validation |
| `outputs/predictions/pred_rolling_vol_5.csv` | 1495 | RollingVol-5 | test, validation |
| `outputs/predictions/lstm_tuned/pred_hybrid_logtarget.csv` | 1457 | Hybrid-LogTarget | test, validation |
| `outputs/predictions/lstm_tuned/pred_hybrid_logtarget_small.csv` | 1457 | Hybrid-LogTarget-Small | test, validation |
| `outputs/predictions/lstm_tuned/pred_hybrid_qlike.csv` | 1457 | Hybrid-QLIKE | test, validation |
| `outputs/predictions/lstm_tuned/pred_lstm_logtarget.csv` | 1457 | LSTM-LogTarget | test, validation |
| `outputs/predictions/lstm_tuned/pred_lstm_logtarget_small.csv` | 1457 | LSTM-LogTarget-Small | test, validation |
| `outputs/predictions/lstm_tuned/pred_lstm_qlike.csv` | 1457 | LSTM-QLIKE | test, validation |

## Errors

None.
