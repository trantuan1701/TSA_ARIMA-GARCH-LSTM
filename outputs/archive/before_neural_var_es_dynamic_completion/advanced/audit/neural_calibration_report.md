# Neural Calibration Report

## Methods

- Multiplicative QLIKE scale: `pred_calibrated = c * pred_var`, with `c` selected by validation QLIKE over `logspace(-2, 2, 401)`.
- Mean-matching scale: `c = mean(actual_var_validation) / mean(pred_var_validation)`.
- Nonnegative affine calibration: `max(epsilon, a + b * pred_var)`, with `a >= 0` and `b >= 0` fitted on validation.
- Isotonic calibration is included only when scikit-learn is available and is flagged as more flexible.

## Best Validation Calibrations

- ARIMA-GARCH-LSTM: best `isotonic` validation QLIKE 1.550506; raw validation QLIKE 1.615283.
- Hybrid-LogTarget: best `isotonic` validation QLIKE 1.524218; raw validation QLIKE 4.333757.
- Hybrid-LogTarget-Small: best `isotonic` validation QLIKE 1.510401; raw validation QLIKE 3.891336.
- Hybrid-QLIKE: best `isotonic` validation QLIKE 1.546045; raw validation QLIKE 1.659582.
- LSTM: best `isotonic` validation QLIKE 1.583359; raw validation QLIKE 1.667317.
- LSTM-LogTarget: best `isotonic` validation QLIKE 1.547466; raw validation QLIKE 4.371889.
- LSTM-LogTarget-Small: best `isotonic` validation QLIKE 1.631877; raw validation QLIKE 4.939853.
- LSTM-QLIKE: best `isotonic` validation QLIKE 1.588594; raw validation QLIKE 1.679371.

## Log-Target Scale Bias Check

- Hybrid-LogTarget: raw test pred/actual ratio 0.280, spike underprediction 1.000; best calibrated test QLIKE 1.167361 with ratio 1.525.
- Hybrid-LogTarget-Small: raw test pred/actual ratio 0.275, spike underprediction 1.000; best calibrated test QLIKE 1.118671 with ratio 1.326.
- LSTM-LogTarget: raw test pred/actual ratio 0.255, spike underprediction 1.000; best calibrated test QLIKE 1.089682 with ratio 1.383.
- LSTM-LogTarget-Small: raw test pred/actual ratio 0.283, spike underprediction 1.000; best calibrated test QLIKE 1.209152 with ratio 1.749.

## Interpretation

- Calibration is fitted only on validation data and then applied unchanged to test predictions.
- A QLIKE improvement after calibration supports scale bias as one mechanism; persistent spike underprediction after calibration points to shape/timing errors rather than scale alone.

## Output Tables

- `outputs/advanced/tables/table_neural_calibration_validation.csv`
- `outputs/advanced/tables/table_neural_calibration_test.csv`
- `outputs/advanced/tables/table_neural_calibration_underprediction.csv`
