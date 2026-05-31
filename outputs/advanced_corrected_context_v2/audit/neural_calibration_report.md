# Neural Calibration Report

## Methods

- Multiplicative QLIKE scale: `pred_calibrated = c * pred_var`, with `c` selected by validation QLIKE over `logspace(-2, 2, 401)`.
- Mean-matching scale: `c = mean(actual_var_validation) / mean(pred_var_validation)`.
- Nonnegative affine calibration: `max(epsilon, a + b * pred_var)`, with `a >= 0` and `b >= 0` fitted on validation.
- Isotonic calibration is included only when scikit-learn is available and is flagged as more flexible.

## Best Validation Calibrations

- ARIMA-GARCH-LSTM: best `isotonic` validation QLIKE 1.545051; raw validation QLIKE 1.618412.
- Hybrid-LogTarget: best `isotonic` validation QLIKE 1.523850; raw validation QLIKE 4.477199.
- Hybrid-LogTarget-Small: best `isotonic` validation QLIKE 1.508107; raw validation QLIKE 4.014371.
- Hybrid-QLIKE: best `isotonic` validation QLIKE 1.555025; raw validation QLIKE 1.664674.
- LSTM: best `isotonic` validation QLIKE 1.577700; raw validation QLIKE 1.665904.
- LSTM-LogTarget: best `isotonic` validation QLIKE 1.542935; raw validation QLIKE 4.440892.
- LSTM-LogTarget-Small: best `isotonic` validation QLIKE 1.623101; raw validation QLIKE 5.035210.
- LSTM-QLIKE: best `isotonic` validation QLIKE 1.586111; raw validation QLIKE 1.682560.

## Log-Target Scale Bias Check

- Hybrid-LogTarget: raw test pred/actual ratio 0.280, spike underprediction 1.000; best calibrated test QLIKE 1.173805 with ratio 1.595.
- Hybrid-LogTarget-Small: raw test pred/actual ratio 0.278, spike underprediction 1.000; best calibrated test QLIKE 1.126931 with ratio 1.340.
- LSTM-LogTarget: raw test pred/actual ratio 0.254, spike underprediction 1.000; best calibrated test QLIKE 1.096090 with ratio 1.377.
- LSTM-LogTarget-Small: raw test pred/actual ratio 0.283, spike underprediction 1.000; best calibrated test QLIKE 1.216443 with ratio 1.755.

## Interpretation

- Calibration is fitted only on validation data and then applied unchanged to test predictions.
- A QLIKE improvement after calibration supports scale bias as one mechanism; persistent spike underprediction after calibration points to shape/timing errors rather than scale alone.

## Output Tables

- `outputs/advanced/tables/table_neural_calibration_validation.csv`
- `outputs/advanced/tables/table_neural_calibration_test.csv`
- `outputs/advanced/tables/table_neural_calibration_underprediction.csv`
