# Experiment Contract

## 1. Research Objective

This project forecasts one-step-ahead VN-Index volatility, not the VN-Index price level.

## 2. Target Definition

`target_var_next = squared_return.shift(-1)`

In every prediction file, `actual_var` must equal `target_var_next`.

## 3. Date Semantics

- `date` = prediction origin date, meaning the day whose information is available.
- `target_date` = date being forecasted.
- `target_date` is usually the next trading day.
- For row `t`, features must only use information available up to `date t`.

## 4. Forbidden Predictors

The following columns must never be used as model input features:

- `target_var_next`
- `target_date`

They can be used only for labels, alignment, and evaluation.

## 5. Chronological Split

Use the existing Stage 1 files:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

Do not reshuffle.

Do not create random train/test splits.

## 6. No Leakage Rules

- Do not use test data for model selection.
- Do not fit scalers on validation or test data.
- Rolling features must be trailing-only.
- For econometric models, parameters should be estimated using training data only.
- Validation and test forecasts may use past observed returns recursively, but must not refit parameters on validation/test data unless explicitly documented as a rolling refit experiment.
- For Stage 2, use fixed-parameter forecasting as the default.

## 7. Prediction File Schema

Every prediction CSV must use this schema:

```text
date,target_date,actual_var,pred_var,model,split
```

Where:

- `actual_var = target_var_next`
- `pred_var` = predicted next-day variance
- `model` = canonical model name
- `split` = `validation` or `test`

## 8. Canonical Model Names

Use exactly these model names:

- `HistoricalMean`
- `RollingVol-5`
- `RollingVol-10`
- `RollingVol-20`
- `GARCH(1,1)`
- `ARIMA-GARCH`
- `LSTM`
- `ARIMA-GARCH-LSTM`

## 9. Evaluation Metrics

Compute:

```text
RMSE = sqrt(mean((actual_var - pred_var)^2))
MAE = mean(abs(actual_var - pred_var))
QLIKE = mean(log(pred_var) + actual_var / pred_var)
```

Before computing QLIKE, clip `pred_var` using `epsilon = 1e-8`.

## 10. Output Directories

Use:

- `outputs/predictions/`
- `outputs/metrics/`
- `outputs/models/`
- `data/processed/`

Create directories automatically.

## 11. Econometric Feature Bridge for Stage 3

Stage 2 must also save a bridge file for LSTM/hybrid modeling:

```text
data/processed/vnindex_with_econometric_features.csv
```

This file should preserve all rows from train, validation, and test and include econometric volatility features generated without using future information.
