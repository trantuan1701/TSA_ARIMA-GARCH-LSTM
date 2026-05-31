# Corrected-Context Neural Retraining Plan

- Generated at: 2026-05-31T16:20:48.302092+00:00
- Legacy neural/neural-dependent artifacts audited: 103
- Legacy artifacts are copied under `outputs/neural_legacy_pre_context_fix/` and `models/neural_legacy_pre_context_fix/`.
- Corrected retraining will write under `outputs/neural_corrected_context_v2/` and recalibrated neural/combination outputs under `outputs/advanced_corrected_context_v2/`.
- Required corrected-context retraining set: `LSTM`, `ARIMA-GARCH-LSTM`, `LSTM-QLIKE`, `Hybrid-QLIKE`, log-target base/hybrid variants, and small log-target variants.
- Training uses fixed seed 42, `PYTHONHASHSEED=42`, TensorFlow deterministic options where available, sequence length 20, training-only imputation/scaling, validation-only early stopping, and no test data in model fitting or calibration.
- Calibration and forecast-combination weights will be refit from validation data only after corrected neural predictions are regenerated.
- Multi-seed robustness will be attempted only if runtime leaves enough margin; otherwise the deterministic single-seed regeneration is the authoritative corrected artifact set.
