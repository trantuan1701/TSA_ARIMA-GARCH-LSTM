# Forecast Encompassing Report

## Regression Specification

- `actual_var_t = a + b1 * pred_garch_t + b2 * pred_candidate_t + error_t`.
- Robust HC1 standard errors are used when statsmodels is available.
- Because squared return is a noisy volatility proxy, coefficient tests are interpreted cautiously.

## Candidate Coefficients

- ARIMA-GARCH: test candidate coefficient -13.031774, p=0.00308849, R2=0.1708.
- LSTM-QLIKE: test candidate coefficient 3.473865, p=0.00342014, R2=0.0971.
- Hybrid-QLIKE: test candidate coefficient 3.252793, p=0.00355091, R2=0.0975.
- Calibrated-Hybrid-LogTarget-Small-isotonic: test candidate coefficient 1.002104, p=0.0153148, R2=0.1027.
- ComboStack-PoolD_AllStable: test candidate coefficient 1.150786, p=0.00114805, R2=0.1053.

## Incremental QLIKE DM Tests

- ARIMA-GARCH: combo minus GARCH mean QLIKE diff 0.000000, p=1, candidate weight 0.00.
- LSTM-QLIKE: combo minus GARCH mean QLIKE diff -0.010327, p=0.876787, candidate weight 0.35.
- Hybrid-QLIKE: combo minus GARCH mean QLIKE diff 0.005586, p=0.938121, candidate weight 0.35.
- Calibrated-Hybrid-LogTarget-Small-isotonic: combo minus GARCH mean QLIKE diff 0.065782, p=0.393226, candidate weight 1.00.
- ComboStack-PoolD_AllStable: combo minus GARCH mean QLIKE diff 0.060641, p=0.507371, candidate weight 1.00.

## Output Tables

- `outputs/advanced/tables/table_forecast_encompassing.csv`
- `outputs/advanced/tables/table_incremental_information_dm.csv`
