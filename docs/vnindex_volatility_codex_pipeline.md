# VN-Index Volatility Forecasting Experiment Plan

## Project Title

**Modeling and Forecasting VN-Index Volatility: An ARIMA–GARCH–LSTM Hybrid Approach**

## Purpose

This project builds an end-to-end empirical pipeline for forecasting VN-Index volatility using traditional econometric models, deep learning models, and a hybrid ARIMA–GARCH–LSTM approach.

The project is not intended to claim a new model architecture. It is an applied empirical study on VN-Index data from CafeF, using a reproducible experimental pipeline.

---

# 1. High-Level Research Framing

## Main task

The goal is not to forecast the VN-Index closing price directly.

The main task is:

> Use historical VN-Index daily data to forecast one-step-ahead volatility.

The core transformation is:

```text
VN-Index close price
→ log return
→ volatility proxy
→ forecast next-day volatility
```

The main volatility proxy is:

```text
target_var_next = squared_return.shift(-1)
```

where:

```text
squared_return = log_return_pct ** 2
```

and:

```text
log_return_pct = 100 * log(close_t / close_{t-1})
```

So the model uses information available up to day `t` to forecast the volatility proxy for day `t+1`.

---

# 2. Data Description

## Source

CafeF historical price data for VNINDEX.

Historical page:

```text
https://s.cafef.vn/lich-su-giao-dich-vnindex-1.chn
```

Ajax endpoint:

```text
https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol=VNINDEX&StartDate=01/01/2010&EndDate=31/12/2025&PageIndex=1&PageSize=100000
```

## Input files

Expected existing input:

```text
data/processed/vnindex_cafef_2010_2025_clean.csv
```

Optional existing input:

```text
data/processed/vnindex_cafef_2010_2025_clean.xlsx
```

## Clean variables

The clean dataset should contain:

```text
date
open
high
low
close
volume
trading_value
log_return_pct
squared_return
abs_return
rolling_vol_5
rolling_vol_10
rolling_vol_20
```

## Modeling target

The target for one-step-ahead volatility forecasting is:

```text
target_var_next = squared_return.shift(-1)
```

---

# 3. Final Project Outputs

The project is considered successful when the following outputs exist:

```text
outputs/metrics/final_model_comparison.csv
outputs/tables/table_model_comparison.md

outputs/figures/fig_close_price.png
outputs/figures/fig_log_return.png
outputs/figures/fig_squared_return.png
outputs/figures/fig_rolling_volatility.png
outputs/figures/fig_actual_vs_predicted_test.png
outputs/figures/fig_rmse_comparison.png
outputs/figures/fig_qlike_comparison.png

outputs/predictions/pred_garch_11.csv
outputs/predictions/pred_arima_garch.csv
outputs/predictions/pred_lstm_base.csv
outputs/predictions/pred_lstm_hybrid.csv
```

The most important output is:

```text
outputs/metrics/final_model_comparison.csv
```

This file will drive the empirical results section of the paper.

---

# 4. Reduced Pipeline for Codex

The pipeline is divided into four major stages:

```text
Stage 1: Data preparation, EDA, and split
Stage 2: Baseline, GARCH, and ARIMA-GARCH
Stage 3: LSTM and ARIMA-GARCH-LSTM hybrid
Stage 4: Final evaluation and paper-ready outputs
```

Additionally, create a runner file:

```text
run_experiment.py
```

which executes all stages sequentially.

---

# Stage 1 — Data Preparation, EDA, and Split

## Goal

Convert the current cleaned CafeF dataset into a model-ready dataset, generate basic EDA figures, and create chronological train/validation/test splits.

## Input

```text
data/processed/vnindex_cafef_2010_2025_clean.csv
```

## Script to implement

```text
src/prepare_data.py
```

## Required tasks

The script must:

```text
1. Load CSV with pandas.
2. Parse date as datetime.
3. Sort data by date ascending.
4. Check duplicate dates.
5. Check missing close values.
6. Recompute log_return_pct if missing:
   log_return_pct = 100 * log(close / close.shift(1))
7. Recompute squared_return if missing:
   squared_return = log_return_pct ** 2
8. Recompute abs_return if missing:
   abs_return = abs(log_return_pct)
9. Recompute rolling volatility features if missing:
   rolling_vol_5 = rolling std of log_return_pct over 5 sessions
   rolling_vol_10 = rolling std of log_return_pct over 10 sessions
   rolling_vol_20 = rolling std of log_return_pct over 20 sessions
10. Create target:
    target_var_next = squared_return.shift(-1)
11. Drop rows with missing essential values.
12. Split chronologically:
    train:      date <= 2019-12-31
    validation: 2020-01-01 to 2022-12-31
    test:       date >= 2023-01-01
13. Save model-ready files.
14. Generate EDA figures.
15. Save a data summary table.
```

## Expected outputs

```text
data/processed/vnindex_model_ready.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv

outputs/figures/fig_close_price.png
outputs/figures/fig_log_return.png
outputs/figures/fig_squared_return.png
outputs/figures/fig_rolling_volatility.png

outputs/metrics/data_summary.csv
```

## Acceptance criteria

```text
- Data is sorted by date.
- No time-series shuffling.
- target_var_next exists.
- Train, validation, and test sets do not overlap.
- close has no missing values.
- Required columns exist in every split file.
- Folders are created automatically.
```

## Codex prompt for Stage 1

```text
Implement src/prepare_data.py for a VN-Index volatility forecasting project.

Input:
data/processed/vnindex_cafef_2010_2025_clean.csv

Requirements:
- Load data with pandas.
- Parse date as datetime and sort ascending.
- Validate duplicate dates, missing close, and required columns.
- Recompute these columns if missing:
  log_return_pct = 100 * log(close / close.shift(1))
  squared_return = log_return_pct ** 2
  abs_return = abs(log_return_pct)
  rolling_vol_5, rolling_vol_10, rolling_vol_20 as rolling std of log_return_pct.
- Create target_var_next = squared_return.shift(-1).
- Drop rows with missing target_var_next and missing essential features.
- Split chronologically:
  train <= 2019-12-31
  validation 2020-01-01 to 2022-12-31
  test >= 2023-01-01
- Save:
  data/processed/vnindex_model_ready.csv
  data/processed/train.csv
  data/processed/validation.csv
  data/processed/test.csv
  outputs/metrics/data_summary.csv
- Generate figures:
  close price, log returns, squared returns, rolling volatility 5/10/20.
- Add a CLI entrypoint so I can run:
  python src/prepare_data.py

Do not shuffle time series data.
Make the script robust and create folders automatically.
```

---

# Stage 2 — Baseline, GARCH, and ARIMA-GARCH

## Goal

Train the core econometric models and simple baselines.

This stage ensures the paper has a strong traditional volatility forecasting benchmark before adding LSTM.

## Inputs

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

## Script to implement

```text
src/train_econometric.py
```

## Models to train

```text
1. Historical mean variance baseline
2. RollingVol-5 baseline
3. RollingVol-10 baseline
4. RollingVol-20 baseline
5. GARCH(1,1)
6. ARIMA-GARCH
```

## Baseline definitions

### Historical mean variance

```text
pred_var = mean squared_return from train
```

### RollingVol-5

```text
pred_var = rolling_vol_5 ** 2
```

### RollingVol-10

```text
pred_var = rolling_vol_10 ** 2
```

### RollingVol-20

```text
pred_var = rolling_vol_20 ** 2
```

## GARCH(1,1)

Use `arch.arch_model` on `log_return_pct`.

The output is a one-step-ahead conditional variance forecast.

## ARIMA-GARCH

Pipeline:

```text
log_return_pct
→ ARIMA(p,0,q), select p and q by AIC on train
→ residual
→ GARCH(1,1) on residual
→ forecast variance
```

Use a small grid for ARIMA:

```text
p = 0, 1, 2, 3
d = 0
q = 0, 1, 2, 3
```

For speed, use a fixed-fit approach first. Rolling refit is optional and can be added later.

## Metrics

Use the same metrics for every model:

```text
RMSE
MAE
QLIKE
```

QLIKE formula:

```text
QLIKE = mean(log(pred_var) + actual_var / pred_var)
```

Clip predictions before metric calculation:

```text
pred_var = max(pred_var, 1e-8)
```

## Expected outputs

```text
outputs/predictions/pred_baseline_mean.csv
outputs/predictions/pred_rolling_vol_5.csv
outputs/predictions/pred_rolling_vol_10.csv
outputs/predictions/pred_rolling_vol_20.csv
outputs/predictions/pred_garch_11.csv
outputs/predictions/pred_arima_garch.csv

outputs/metrics/econometric_metrics.csv
outputs/metrics/arima_order_selection.csv
outputs/metrics/garch_summary.txt
outputs/metrics/arima_garch_summary.txt

outputs/models/garch_11.pkl
outputs/models/arima_model.pkl
```

## Required prediction schema

Every prediction CSV must use this schema:

```text
date
actual_var
pred_var
model
split
```

## Acceptance criteria

```text
- pred_var is always positive after clipping.
- actual_var equals target_var_next.
- Metrics are calculated separately for validation and test.
- Test set is not used for ARIMA order selection.
- If a model fails to converge, skip that model and log a clear warning.
- All output folders are created automatically.
```

## Codex prompt for Stage 2

```text
Implement src/train_econometric.py.

Goal:
Train baseline, GARCH(1,1), and ARIMA-GARCH models for one-step-ahead VN-Index volatility forecasting.

Inputs:
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv

Target:
target_var_next, defined as next-day squared return.

Models:
1. HistoricalMean:
   pred_var = mean squared_return from train.
2. RollingVol-5:
   pred_var = rolling_vol_5 ** 2.
3. RollingVol-10:
   pred_var = rolling_vol_10 ** 2.
4. RollingVol-20:
   pred_var = rolling_vol_20 ** 2.
5. GARCH(1,1):
   Fit on train log_return_pct using arch.arch_model.
   Forecast variance for validation and test.
   Use a simple fixed-fit approach first.
6. ARIMA-GARCH:
   Grid search ARIMA(p,0,q) with p,q in 0..3 on train log_return_pct.
   Select by AIC.
   Fit GARCH(1,1) on ARIMA residuals.
   Forecast variance for validation and test.

Evaluation metrics:
RMSE, MAE, QLIKE.
QLIKE = mean(log(pred_var) + actual_var / pred_var)
Clip pred_var with epsilon = 1e-8 before metrics.

Save prediction files to outputs/predictions/.
Save metrics to outputs/metrics/econometric_metrics.csv.
Save ARIMA order selection to outputs/metrics/arima_order_selection.csv.
Save summaries as txt files.
Save fitted models to outputs/models/.

Prediction file schema:
date, actual_var, pred_var, model, split

Add CLI:
python src/train_econometric.py

Make it robust, reproducible, and create folders automatically.
```

---

# Stage 3 — LSTM and ARIMA-GARCH-LSTM Hybrid

## Goal

Train the deep learning baseline and the proposed hybrid model.

The hybrid model is the main model matching the paper title.

## Inputs

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
outputs/predictions/pred_arima_garch.csv
```

## Script to implement

```text
src/train_lstm_hybrid.py
```

## Models to train

```text
1. LSTM base
2. ARIMA-GARCH-LSTM hybrid
```

## Base LSTM features

```text
log_return_pct
squared_return
abs_return
rolling_vol_5
rolling_vol_10
rolling_vol_20
volume
trading_value
```

## Hybrid features

Use all base features plus:

```text
garch_pred_var
garch_pred_vol
```

where:

```text
garch_pred_vol = sqrt(garch_pred_var)
```

If `garch_pred_var` is not available for train, use in-sample fitted conditional variance from GARCH. If that is not available in the current implementation, use a simple fallback and log it clearly.

## Target

```text
target_var_next
```

## Sequence length

```text
seq_len = 20
```

This means the model uses the previous 20 trading sessions to forecast next-day volatility.

## Preprocessing rules

```text
- Fit StandardScaler only on train features.
- Transform validation and test using the train-fitted scaler.
- Do not fit scaler on validation or test.
- Do not shuffle time-series data.
```

## LSTM architecture

```text
Input shape = (seq_len, num_features)
LSTM(64)
Dropout(0.2)
Dense(32, activation="relu")
Dense(1, activation="softplus")
```

Use `softplus` so predicted variance stays non-negative.

## Training setup

```text
optimizer = Adam(learning_rate=0.001)
loss = mse
epochs = 100
batch_size = 32
EarlyStopping:
    monitor = val_loss
    patience = 10
    restore_best_weights = True
shuffle = False
```

## Metrics

```text
RMSE
MAE
QLIKE
```

## Expected outputs

```text
outputs/models/lstm_base.keras
outputs/models/lstm_hybrid.keras

outputs/predictions/pred_lstm_base.csv
outputs/predictions/pred_lstm_hybrid.csv

outputs/metrics/lstm_metrics.csv
outputs/metrics/lstm_training_history.csv
outputs/metrics/hybrid_training_history.csv

outputs/figures/fig_lstm_base_loss.png
outputs/figures/fig_lstm_hybrid_loss.png
```

## Required prediction schema

```text
date
actual_var
pred_var
model
split
```

## Acceptance criteria

```text
- StandardScaler is fitted only on train.
- shuffle=False during training.
- Output variance is non-negative.
- Prediction files use the same schema as econometric models.
- Early stopping is used.
- Script can run without GPU.
- Random seeds are set.
- Missing GARCH features are handled gracefully with a clear warning.
```

## Codex prompt for Stage 3

```text
Implement src/train_lstm_hybrid.py.

Goal:
Train two neural volatility forecasting models:
1. LSTM base
2. ARIMA-GARCH-LSTM hybrid

Inputs:
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
outputs/predictions/pred_arima_garch.csv if available

Target:
target_var_next

Base LSTM features:
log_return_pct
squared_return
abs_return
rolling_vol_5
rolling_vol_10
rolling_vol_20
volume
trading_value

Hybrid features:
All base features plus:
garch_pred_var
garch_pred_vol

If garch_pred_var is not available for all splits, create it by merging predictions where possible. For train, use in-sample fitted conditional variance if available; otherwise implement a simple fallback and clearly log it.

Preprocessing:
- Fit StandardScaler only on train features.
- Transform train, validation, test.
- Create sequences with seq_len=20.
- Do not shuffle time series data.

Model architecture:
Input shape = (seq_len, num_features)
LSTM(64)
Dropout(0.2)
Dense(32, activation="relu")
Dense(1, activation="softplus")

Training:
optimizer Adam learning_rate=0.001
loss mse
epochs=100
batch_size=32
EarlyStopping monitor val_loss, patience=10, restore_best_weights=True
shuffle=False

Evaluation:
RMSE, MAE, QLIKE on validation and test.
Clip predictions with epsilon=1e-8.

Outputs:
outputs/models/lstm_base.keras
outputs/models/lstm_hybrid.keras
outputs/predictions/pred_lstm_base.csv
outputs/predictions/pred_lstm_hybrid.csv
outputs/metrics/lstm_metrics.csv
outputs/metrics/lstm_training_history.csv
outputs/metrics/hybrid_training_history.csv
outputs/figures/fig_lstm_base_loss.png
outputs/figures/fig_lstm_hybrid_loss.png

Prediction file schema:
date, actual_var, pred_var, model, split

Add CLI:
python src/train_lstm_hybrid.py

Make the script robust and reproducible.
Set random seeds.
Create folders automatically.
```

---

# Stage 4 — Final Evaluation and Paper-Ready Outputs

## Goal

Combine all prediction files, compute final metrics, rank models, and create figures/tables ready for the paper.

## Input

```text
outputs/predictions/*.csv
```

## Script to implement

```text
src/evaluate_all.py
```

## Required tasks

The script must:

```text
1. Load all CSV files in outputs/predictions/.
2. Validate required schema.
3. Concatenate predictions.
4. Clip pred_var with epsilon = 1e-8.
5. Compute RMSE, MAE, QLIKE for each model and split.
6. Rank models on the test split by QLIKE and RMSE.
7. Save final comparison tables.
8. Plot actual vs predicted volatility on the test set.
9. Plot RMSE, MAE, and QLIKE comparison bar charts.
10. Export markdown and LaTeX tables for paper writing.
```

## Expected outputs

```text
outputs/metrics/final_model_comparison.csv
outputs/metrics/final_model_comparison_test_only.csv

outputs/figures/fig_actual_vs_predicted_test.png
outputs/figures/fig_rmse_comparison.png
outputs/figures/fig_mae_comparison.png
outputs/figures/fig_qlike_comparison.png

outputs/tables/table_model_comparison.md
outputs/tables/table_model_comparison.tex
```

## Actual vs predicted plot

Use test split only.

Include major models if available:

```text
RollingVol-20
GARCH(1,1)
ARIMA-GARCH
LSTM
ARIMA-GARCH-LSTM
```

## Acceptance criteria

```text
- All available models appear in the final comparison table.
- Models are ranked on the test set.
- The script does not crash if one prediction file is missing.
- Missing files or invalid files generate warnings.
- Markdown table is easy to copy into the paper.
- Figures are readable and saved to outputs/figures/.
```

## Codex prompt for Stage 4

```text
Implement src/evaluate_all.py.

Goal:
Create final paper-ready evaluation outputs for the VN-Index volatility forecasting project.

Input:
All CSV files in outputs/predictions/ with schema:
date, actual_var, pred_var, model, split

Tasks:
- Load and concatenate all prediction files.
- Validate required columns.
- Clip pred_var with epsilon=1e-8.
- Compute RMSE, MAE, QLIKE for each model and split.
- Rank models on the test split by QLIKE and RMSE.
- Save:
  outputs/metrics/final_model_comparison.csv
  outputs/metrics/final_model_comparison_test_only.csv
  outputs/tables/table_model_comparison.md
  outputs/tables/table_model_comparison.tex
- Plot:
  outputs/figures/fig_actual_vs_predicted_test.png
  outputs/figures/fig_rmse_comparison.png
  outputs/figures/fig_mae_comparison.png
  outputs/figures/fig_qlike_comparison.png

For actual vs predicted plot:
- Use test split only.
- Plot actual_var and predictions from major models:
  RollingVol-20, GARCH(1,1), ARIMA-GARCH, LSTM, ARIMA-GARCH-LSTM if available.
- Make the plot readable.

Add CLI:
python src/evaluate_all.py

Do not fail if some prediction files are missing; print a warning and continue.
```

---

# 5. Runner Script

## Goal

Create a single command to run the full experiment.

## Script to implement

```text
run_experiment.py
```

## Required behavior

The script runs:

```text
python src/prepare_data.py
python src/train_econometric.py
python src/train_lstm_hybrid.py
python src/evaluate_all.py
```

It should:

```text
- Print clear stage headers.
- Stop if a stage fails.
- Show the failing command.
- Use subprocess.run(..., check=True).
```

## Expected command

```bash
python run_experiment.py
```

## Codex prompt for runner

```text
Create run_experiment.py that sequentially runs:
1. src/prepare_data.py
2. src/train_econometric.py
3. src/train_lstm_hybrid.py
4. src/evaluate_all.py

Use subprocess.run with check=True.
Print clear stage headers.
If a stage fails, stop and show the error.
```

---

# 6. Suggested Project Structure

```text
vnindex-volatility/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── vnindex_cafef_2010_2025_clean.csv
│       ├── vnindex_model_ready.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── src/
│   ├── prepare_data.py
│   ├── train_econometric.py
│   ├── train_lstm_hybrid.py
│   └── evaluate_all.py
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── paper/
│
└── run_experiment.py
```

---

# 7. Recommended Dependencies

Install:

```bash
python -m pip install pandas numpy matplotlib scikit-learn statsmodels arch tensorflow openpyxl joblib
```

Optional:

```bash
python -m pip install tabulate
```

---

# 8. Evaluation Metrics

## RMSE

```text
RMSE = sqrt(mean((actual_var - pred_var)^2))
```

## MAE

```text
MAE = mean(abs(actual_var - pred_var))
```

## QLIKE

```text
QLIKE = mean(log(pred_var) + actual_var / pred_var)
```

Use:

```text
pred_var = max(pred_var, 1e-8)
```

before calculating QLIKE.

---

# 9. Final Model List

Minimum required models:

```text
1. HistoricalMean
2. RollingVol-5
3. RollingVol-10
4. RollingVol-20
5. GARCH(1,1)
6. ARIMA-GARCH
7. LSTM
8. ARIMA-GARCH-LSTM
```

If time is limited, the minimum acceptable set is:

```text
1. RollingVol-20
2. GARCH(1,1)
3. ARIMA-GARCH
4. LSTM
5. ARIMA-GARCH-LSTM
```

---

# 10. Definition of Done

The experiment pipeline is complete when the following command runs successfully:

```bash
python run_experiment.py
```

and produces:

```text
outputs/metrics/final_model_comparison.csv
outputs/tables/table_model_comparison.md
outputs/figures/fig_actual_vs_predicted_test.png
```

The final comparison table should contain at least:

```text
model
split
RMSE
MAE
QLIKE
rank
```

---

# 11. Paper Writing Notes

## Positioning

Do not write:

```text
This paper proposes a novel ARIMA-GARCH-LSTM model.
```

Prefer:

```text
This study applies and evaluates an ARIMA-GARCH-LSTM hybrid framework for forecasting VN-Index volatility using daily data from 2010 to 2025.
```

or:

```text
Rather than proposing a new forecasting architecture, this study provides an empirical comparison of traditional econometric, deep learning, and hybrid approaches for VN-Index volatility forecasting.
```

## Main contribution

The contribution is applied and empirical:

```text
1. Updated empirical evidence on VN-Index volatility forecasting using daily data from 2010 to 2025.
2. A fair comparison between rolling volatility baselines, GARCH, ARIMA-GARCH, LSTM, and hybrid models.
3. A hybrid framework where GARCH-derived volatility estimates are used as additional inputs for LSTM.
4. Evaluation using volatility-appropriate metrics such as RMSE, MAE, and QLIKE.
```

## Methodology summary

```text
First, daily closing prices are transformed into percentage log returns.
The one-step-ahead volatility target is proxied by the next-day squared return.
The dataset is split chronologically into training, validation, and test periods.
Several models are estimated, including rolling volatility benchmarks, GARCH, ARIMA-GARCH, LSTM, and the proposed ARIMA-GARCH-LSTM hybrid model.
The ARIMA component captures the conditional mean of returns, while the GARCH component models time-varying conditional variance.
The LSTM model uses lagged return-based, volatility-based, and liquidity-related features to learn nonlinear temporal patterns.
In the hybrid model, conditional volatility estimates from ARIMA-GARCH are added as additional inputs to the LSTM.
All models are evaluated on the same test set using RMSE, MAE, and QLIKE.
```

---

# 12. One-Shot Prompt for Codex

Use this if you want to ask Codex to build the whole project at once.

```text
Build an end-to-end Python experiment pipeline for a paper titled:
"Modeling and Forecasting VN-Index Volatility: An ARIMA-GARCH-LSTM Hybrid Approach".

Data:
data/processed/vnindex_cafef_2010_2025_clean.csv

Target:
One-step-ahead volatility proxied by next-day squared log return:
target_var_next = squared_return.shift(-1)

Split:
Train <= 2019-12-31
Validation 2020-01-01 to 2022-12-31
Test >= 2023-01-01

Implement these scripts:
1. src/prepare_data.py
2. src/train_econometric.py
3. src/train_lstm_hybrid.py
4. src/evaluate_all.py
5. run_experiment.py

Models:
- Historical mean variance
- RollingVol-5
- RollingVol-10
- RollingVol-20
- GARCH(1,1)
- ARIMA-GARCH
- LSTM base
- ARIMA-GARCH-LSTM hybrid

Metrics:
RMSE, MAE, QLIKE

Output:
- predictions in outputs/predictions/
- metrics in outputs/metrics/
- figures in outputs/figures/
- markdown and latex tables in outputs/tables/

Important:
- No time-series shuffling.
- Fit scalers only on train.
- Do not use test set for model selection.
- Prediction CSV schema must be:
  date, actual_var, pred_var, model, split
- Make all scripts runnable from CLI.
- Create folders automatically.
- Make the code robust and reproducible.
```

---

# 13. Practical Execution Order

Recommended order:

```text
1. Ask Codex to implement Stage 1.
2. Run:
   python src/prepare_data.py
3. Check train.csv, validation.csv, test.csv.
4. Ask Codex to implement Stage 2.
5. Run:
   python src/train_econometric.py
6. Check outputs/metrics/econometric_metrics.csv.
7. Ask Codex to implement Stage 3.
8. Run:
   python src/train_lstm_hybrid.py
9. Check LSTM prediction files.
10. Ask Codex to implement Stage 4.
11. Run:
    python src/evaluate_all.py
12. Finally run:
    python run_experiment.py
```

Do not start writing the full paper before generating:

```text
outputs/metrics/final_model_comparison.csv
```

That file is the backbone of the empirical results section.
