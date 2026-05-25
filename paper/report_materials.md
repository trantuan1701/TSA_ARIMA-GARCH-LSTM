# Report Materials: VN-Index Volatility Forecasting

## 1. Project Overview

This project forecasts VN-Index volatility, not the VN-Index price level. The repository title and working paper title are:

**"Modeling and Forecasting VN-Index Volatility: An ARIMA-GARCH-LSTM Hybrid Approach"**

The work should be presented as an applied empirical study. It compares econometric, deep learning, and hybrid approaches for one-step-ahead daily volatility forecasting using VN-Index data. The target is a next-trading-day variance proxy based on squared percentage log returns.

Main research question:

> Can LSTM-based or ARIMA-GARCH-LSTM hybrid models improve one-step-ahead VN-Index volatility forecasts compared with traditional econometric volatility models?

The final report should avoid presenting the hybrid model as a new theoretical contribution. The study is best framed as a reproducible empirical comparison for VN-Index volatility forecasting.

## 2. Research Positioning

The study does not claim to propose a completely new model architecture. The empirical contribution is:

- an updated VN-Index daily dataset covering 2010 to 2025;
- a reproducible forecasting pipeline from data preparation to final evaluation;
- a comparison of simple baselines, GARCH, ARIMA-GARCH, LSTM, and ARIMA-GARCH-LSTM models;
- additional tuned neural variants using log-target and QLIKE-style training objectives;
- a common-window evaluation so models are compared on the same test observations.

The final results show that traditional GARCH remains very competitive. In the main common-window test comparison, `GARCH(1,1)` is the best model by QLIKE and RMSE. Tuned log-target neural variants obtain lower MAE, but their QLIKE values are substantially worse, indicating weaker performance for volatility-spike-sensitive evaluation.

## 3. Data Source

The data source documented in `README.md` and the downloader script is CafeF historical VNINDEX data.

| Item | Repository value |
|---|---|
| Symbol | `VNINDEX` |
| Source | CafeF historical VN-Index price data |
| Historical page | `https://s.cafef.vn/lich-su-giao-dich-vnindex-1.chn` |
| Ajax endpoint | `https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol=VNINDEX&StartDate=01/01/2010&EndDate=31/12/2025&PageIndex=1&PageSize=100000` |
| Documented time range | `01/01/2010` to `31/12/2025` |

The README notes that CafeF's Ajax response may be capped or paginated and that the downloader retrieves the sample month by month through the CafeF data endpoint.

Data files used in the project:

- `data/raw/vnindex_cafef_2010_2025_raw.csv`
- `data/processed/vnindex_cafef_2010_2025_clean.csv`
- `data/processed/vnindex_cafef_2010_2025_clean.xlsx`
- `data/processed/vnindex_model_ready.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/vnindex_with_econometric_features.csv`

## 4. Variables

Main variables in the prepared/model-ready data:

| Variable | Meaning |
|---|---|
| `date` | Prediction origin date. Information up to this date is available. |
| `target_date` | Trading date being forecasted, usually the next trading day. |
| `open` | Opening VN-Index level. |
| `high` | Highest VN-Index level during the session. |
| `low` | Lowest VN-Index level during the session. |
| `close` | Closing VN-Index level. |
| `volume` | Matched trading volume from CafeF when available. |
| `trading_value` | Matched trading value from CafeF when available. |
| `log_return_pct` | Percentage log return. |
| `squared_return` | Squared percentage log return. |
| `abs_return` | Absolute percentage log return. |
| `rolling_vol_5` | Trailing 5-session rolling standard deviation of `log_return_pct`. |
| `rolling_vol_10` | Trailing 10-session rolling standard deviation of `log_return_pct`. |
| `rolling_vol_20` | Trailing 20-session rolling standard deviation of `log_return_pct`. |
| `target_var_next` | Next-day volatility proxy, equal to next row's `squared_return`. |

Formulas:

```text
log_return_pct = 100 * log(close_t / close_{t-1})
squared_return = log_return_pct^2
abs_return = abs(log_return_pct)
rolling_vol_k = trailing rolling standard deviation of log_return_pct over k trading sessions
target_var_next = squared_return.shift(-1)
```

The supervised target is `target_var_next`. It is a one-step-ahead volatility proxy rather than a direct price or return-level forecast.

## 5. Data Preparation and Split

The following values come from `outputs/metrics/data_summary.csv`.

| Dataset | Rows | Feature date range | Target date range |
|---|---:|---|---|
| Prepared full | 3989 | 2010-01-04 to 2025-12-31 | 2010-01-05 to 2025-12-31 |
| Model-ready | 3968 | 2010-02-01 to 2025-12-30 | 2010-02-02 to 2025-12-31 |
| Train | 2471 | 2010-02-01 to 2019-12-30 | 2010-02-02 to 2019-12-31 |
| Validation | 750 | 2020-01-02 to 2022-12-29 | 2020-01-03 to 2022-12-30 |
| Test | 745 | 2023-01-03 to 2025-12-30 | 2023-01-04 to 2025-12-31 |

The model-ready dataset starts later than the raw prepared data because lagged returns, rolling volatility features, and the next-day target require dropping rows with missing essential values. In `model_ready`, all essential modeling variables have zero missing values according to `outputs/metrics/data_summary.csv`.

Split semantics:

- `date` is the prediction origin.
- `target_date` is the forecasted trading day.
- Train and validation rows are filtered so their labels do not cross into later periods.
- The test split starts from feature date `2023-01-03` and can forecast target dates through `2025-12-31`.

## 6. Leakage Prevention Rules

The leakage prevention rules are documented in `docs/experiment_contract.md` and implemented across the training/evaluation scripts:

- No time-series shuffling is used.
- `target_var_next` and `target_date` are not predictors.
- `date` and `split` are not predictors.
- Rolling features are trailing-only.
- Scalers are fit only on training data.
- Test data is not used for model selection.
- Econometric model parameters are estimated using the training split only.
- Validation and test forecasts may use past observed returns recursively, but parameters are not refit unless a separate rolling refit experiment is explicitly documented.
- Prediction CSV files use the schema `date,target_date,actual_var,pred_var,model,split`.
- In prediction files, `actual_var` must equal `target_var_next`.
- The final paper comparison should use common-window test metrics so every model is evaluated on the same `date,target_date` observations.

For the LSTM and hybrid models, validation and test sequences are created separately within each split. This prevents sequences from crossing split boundaries.

## 7. Exploratory Data Analysis

The available EDA artifacts are visual/descriptive. Although the README mentions stationarity, autocorrelation, and ARCH diagnostics as intended analysis areas, no separate statistical test output file was found among the inspected requested artifacts. The report should not invent ADF, Ljung-Box, or ARCH-LM values.

| Figure | What it shows | Why it matters | Suggested paper use |
|---|---|---|---|
| `outputs/figures/fig_close_price.png` | Daily VN-Index closing level from 2010 to 2025. | The level series is visually useful for showing long-run market movement and likely non-stationarity in prices. | Data description / EDA. |
| `outputs/figures/fig_log_return.png` | Daily percentage log returns. | Returns fluctuate around zero and are the basis for volatility modeling. | EDA and motivation for modeling returns rather than price levels. |
| `outputs/figures/fig_squared_return.png` | Squared percentage log returns. | Squared returns reveal volatility spikes and clustering visually. | EDA and volatility proxy explanation. |
| `outputs/figures/fig_rolling_volatility.png` | 5-, 10-, and 20-session rolling volatility. | Rolling volatility highlights changing risk regimes over time. | EDA and motivation for GARCH/LSTM inputs. |

Recommended descriptive interpretation:

- The close-price figure should be used to motivate why the project forecasts volatility rather than the index level.
- The log-return figure should be used to show that returns are centered near zero but vary over time.
- The squared-return and rolling-volatility figures should be used to motivate conditional heteroskedasticity and volatility clustering.
- Formal statistical claims should be avoided unless diagnostic test outputs are generated and reported later.

## 8. Methodology

This section describes implemented experiments, not proposed future methods.

### 8.1 Baselines

Implemented baseline models:

- `HistoricalMean`
- `RollingVol-5`
- `RollingVol-10`
- `RollingVol-20`

Their role is to provide simple benchmarks. `HistoricalMean` predicts the training-sample average variance. Rolling volatility baselines use recent trailing volatility windows and square the rolling volatility value to obtain a variance forecast.

### 8.2 GARCH(1,1)

`GARCH(1,1)` models the conditional variance of daily log returns and is designed to capture volatility clustering.

Implementation details from `src/train_econometric.py` and `outputs/metrics/garch_summary.txt`:

- The model is fitted on training returns only.
- Validation and test forecasts use fixed parameters.
- Recursive volatility states are updated using observed post-training returns available before each later prediction origin.
- Boundary rows not evaluated because their labels cross a split boundary still update the recursive volatility state if their observed returns are known before later prediction origins.
- The GARCH fit converged.

Key fitted parameters from `garch_summary.txt`:

| Parameter | Value |
|---|---:|
| `mu` | 0.0456318069903 |
| `omega` | 0.0374384078967 |
| `alpha[1]` | 0.125658552583 |
| `beta[1]` | 0.847605474506 |

### 8.3 ARIMA-GARCH

The ARIMA-GARCH model separates conditional mean and conditional variance:

- ARIMA models the conditional mean of returns.
- GARCH models the variance of the ARIMA residual process.

ARIMA order selection from `outputs/metrics/arima_order_selection.csv` and `outputs/metrics/arima_summary.txt`:

- Candidate grid: `p in [0,1,2,3]`, `d = 0`, `q in [0,1,2,3]`.
- Selection criterion: lowest training AIC among successfully fitted models.
- Validation and test data are not used for order selection.
- Selected order: `ARIMA(0,0,3)`.
- Selected ARIMA AIC: `7434.25083133121`.
- Selected ARIMA BIC: `7463.312722361153`.

ARIMA-GARCH implementation details from `outputs/metrics/arima_garch_summary.txt`:

- ARIMA is fitted on training log returns.
- GARCH(1,1) is fitted on training ARIMA residuals only.
- Validation/test variance forecasts use fixed ARIMA and GARCH parameters.
- The ARIMA future one-step mean method is `statsmodels ARIMAResults.append(..., refit=False)`.
- The ARIMA-GARCH GARCH fit converged.

ARIMA-GARCH GARCH residual-variance parameters:

| Parameter | Value |
|---|---:|
| `omega` | 0.0376470130115 |
| `alpha[1]` | 0.122555645267 |
| `beta[1]` | 0.84932985947 |

### 8.4 LSTM

The base LSTM uses sequences of lagged features to forecast next-day variance.

Implementation details from `src/train_lstm_hybrid.py` and `outputs/metrics/lstm_feature_summary.csv`:

- Sequence length: `20`.
- Base features: `log_return_pct`, `squared_return`, `abs_return`, `rolling_vol_5`, `rolling_vol_10`, `rolling_vol_20`, `volume`, `trading_value`.
- Target: `target_var_next`.
- Number of base features: `8`.
- Sequence counts: train `2452`, validation `731`, test `726`.
- Scaler is fit only on the training rows.
- Validation/test sequences are created within their own split and do not cross split boundaries.
- Output layer uses a positive variance transformation through `softplus`.

The implemented base LSTM architecture is summarized by the script as:

```text
Input -> LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(1, softplus)
```

### 8.5 ARIMA-GARCH-LSTM Hybrid

The ARIMA-GARCH-LSTM hybrid augments the LSTM feature set with econometric forecast features.

Hybrid features from `outputs/metrics/lstm_feature_summary.csv`:

- Base LSTM features:
  `log_return_pct`, `squared_return`, `abs_return`, `rolling_vol_5`, `rolling_vol_10`, `rolling_vol_20`, `volume`, `trading_value`.
- Econometric features:
  `garch_11_pred_var`, `garch_11_pred_vol`, `arima_garch_pred_var`, `arima_garch_pred_vol`, `arima_mean_pred`, `arima_resid_proxy`.
- Number of hybrid features: `14`.
- Sequence length: `20`.
- Sequence counts: train `2452`, validation `731`, test `726`.

The idea is to combine econometric volatility structure with nonlinear sequence learning. This should be described as an empirical hybrid design, not a novel architecture claim.

### 8.6 Tuned LSTM Variants

Stage 3B implemented validation-selected neural variants in `src/tune_lstm_hybrid.py`.

Implemented tuned variants:

- `LSTM-LogTarget`
- `Hybrid-LogTarget`
- `LSTM-QLIKE`
- `Hybrid-QLIKE`
- `LSTM-LogTarget-Small`
- `Hybrid-LogTarget-Small`

Training design:

- Log-target models train on `log(target_var_next + epsilon)` and transform predictions back to the variance scale.
- QLIKE models train directly on the raw variance target using a QLIKE-style loss.
- Small variants use a smaller LSTM/dense configuration than the default tuned variants.
- Best tuned base and hybrid models are selected by validation QLIKE only.
- Test data is not used for neural model selection.
- `outputs/metrics/lstm_tuned/lstm_tuned_failed_variants.csv` contains no failed variant rows beyond the header.

## 9. Evaluation Metrics

Metrics are defined in `docs/experiment_contract.md` and implemented in `src/metrics.py`.

```text
RMSE = sqrt(mean((actual_var - pred_var)^2))
MAE = mean(abs(actual_var - pred_var))
QLIKE = mean(log(pred_var) + actual_var / pred_var)
```

QLIKE interpretation:

- Lower is better.
- QLIKE is important for volatility forecasting because it evaluates variance forecasts directly.
- It penalizes underprediction of volatility spikes strongly through the `actual_var / pred_var` term.
- Before metric computation, `pred_var` is clipped using `epsilon = 1e-8` for numerical stability.

Because neural sequence models lose initial rows due to the 20-day sequence length, the main paper comparison should use the common-window test sample instead of available-sample metrics.

## 10. Main Results: Common Test Window

Source: `outputs/metrics/final_model_comparison_common_test_only.csv`.

This is the main paper comparison table. All listed models are evaluated on the same common test window: `N = 726`. The common test mean actual variance is `1.2119119537615395` for every model. Table values below are rounded to six decimals for readability from the source CSV, which contains full precision.

| Rank by QLIKE | Model | N | RMSE | MAE | QLIKE | Mean predicted variance |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `GARCH(1,1)` | 726 | 3.648839 | 1.393909 | 1.052888 | 1.241320 |
| 2 | `ARIMA-GARCH` | 726 | 3.671816 | 1.400256 | 1.055679 | 1.236080 |
| 3 | `LSTM` | 726 | 3.769716 | 1.639734 | 1.161001 | 1.568589 |
| 4 | `ARIMA-GARCH-LSTM` | 726 | 3.763253 | 1.720113 | 1.168609 | 1.701791 |
| 5 | `LSTM-QLIKE` | 726 | 3.782485 | 1.714073 | 1.185876 | 1.666980 |
| 6 | `HistoricalMean` | 726 | 3.799706 | 1.449153 | 1.192259 | 1.198773 |
| 7 | `Hybrid-QLIKE` | 726 | 3.807296 | 1.862578 | 1.230508 | 1.877606 |
| 8 | `RollingVol-20` | 726 | 3.901051 | 1.462196 | 1.280635 | 1.208200 |
| 9 | `RollingVol-10` | 726 | 3.974129 | 1.478891 | 1.309964 | 1.218541 |
| 10 | `RollingVol-5` | 726 | 4.011383 | 1.492394 | 1.928009 | 1.201815 |
| 11 | `Hybrid-LogTarget` | 726 | 3.867813 | 1.109300 | 2.086360 | 0.339489 |
| 12 | `Hybrid-LogTarget-Small` | 726 | 3.868068 | 1.108165 | 2.232917 | 0.333670 |
| 13 | `LSTM-LogTarget-Small` | 726 | 3.888176 | 1.123546 | 2.281499 | 0.342765 |
| 14 | `LSTM-LogTarget` | 726 | 3.866882 | 1.108626 | 2.301502 | 0.309520 |

Main common-window findings:

- `GARCH(1,1)` is best by QLIKE (`1.052888`) and RMSE (`3.648839`).
- `Hybrid-LogTarget-Small` is best by MAE (`1.108165`), but its QLIKE is much worse (`2.232917`) than GARCH.
- Among the original core models before tuned neural variants, `GARCH(1,1)` is best by QLIKE, RMSE, and MAE.
- `ARIMA-GARCH` is second by QLIKE and very close to `GARCH(1,1)`.
- `LSTM` is better than the initial `ARIMA-GARCH-LSTM` hybrid by QLIKE and MAE.
- `ARIMA-GARCH-LSTM` improves slightly over `LSTM` in RMSE (`3.763253` vs `3.769716`) but not in QLIKE or MAE.
- Historical mean and rolling baselines are weaker than GARCH and ARIMA-GARCH by QLIKE.
- Among rolling baselines, `RollingVol-5` performs worst by QLIKE (`1.928009`).
- Across all models, log-target neural variants have the highest QLIKE values despite low MAE.

Available-sample metrics also exist in `outputs/metrics/final_model_comparison_available_test_only.csv`, but they are secondary for the paper because row counts differ across model classes (`745` for baselines/econometric models and `726` for neural sequence models).

## 11. Neural Tuning Results

Sources:

- `outputs/metrics/lstm_tuned/lstm_tuned_metrics_test_only.csv`
- `outputs/metrics/lstm_tuned/lstm_tuned_selection_summary.csv`

Best tuned models by validation QLIKE:

| Selection group | Best model | Validation QLIKE | Test QLIKE | Test RMSE | Test MAE |
|---|---|---:|---:|---:|---:|
| Base LSTM tuned variants | `LSTM-QLIKE` | 1.679371 | 1.185876 | 3.782485 | 1.714073 |
| Hybrid tuned variants | `Hybrid-QLIKE` | 1.659582 | 1.230508 | 3.807296 | 1.862578 |

Full tuned neural test results. Table values are rounded to six decimals from the source CSV.

| Model | N | RMSE | MAE | QLIKE | Mean predicted variance |
|---|---:|---:|---:|---:|---:|
| `LSTM-QLIKE` | 726 | 3.782485 | 1.714073 | 1.185876 | 1.666980 |
| `Hybrid-QLIKE` | 726 | 3.807296 | 1.862578 | 1.230508 | 1.877606 |
| `Hybrid-LogTarget` | 726 | 3.867813 | 1.109300 | 2.086360 | 0.339489 |
| `Hybrid-LogTarget-Small` | 726 | 3.868068 | 1.108165 | 2.232917 | 0.333670 |
| `LSTM-LogTarget-Small` | 726 | 3.888176 | 1.123546 | 2.281499 | 0.342765 |
| `LSTM-LogTarget` | 726 | 3.866882 | 1.108626 | 2.301502 | 0.309520 |

Comparison against original Stage 3 LSTM and GARCH:

- Original Stage 3 `LSTM` test QLIKE is `1.161001`, RMSE is `3.769716`, and MAE is `1.639734`.
- `LSTM-QLIKE` did not beat the original Stage 3 `LSTM` on test QLIKE, RMSE, or MAE.
- `Hybrid-QLIKE` did not beat the base `LSTM` on test QLIKE, RMSE, or MAE.
- No tuned neural model beat `GARCH(1,1)` on test QLIKE or RMSE.
- `Hybrid-LogTarget-Small` beat `GARCH(1,1)` on MAE only (`1.108165` vs `1.393909`), but its QLIKE was much worse (`2.232917` vs `1.052888`).
- Log-target models improved MAE relative to the original LSTM, but they worsened QLIKE and RMSE.

Why log-target models can show low MAE but high QLIKE:

- They predict much lower average variance than the common test mean actual variance. For example, `LSTM-LogTarget` has mean predicted variance `0.309520`, while the common test mean actual variance is `1.211912`.
- Most daily variance observations are relatively small, so MAE can reward low smooth forecasts.
- During volatility spikes, underprediction is heavily penalized by QLIKE through `actual_var / pred_var`.
- Therefore, MAE and QLIKE can lead to different rankings in volatility forecasting.

## 12. Final Interpretation

Main finding:

`GARCH(1,1)` remains the strongest overall model for daily VN-Index volatility forecasting under the common test window, especially when judged by QLIKE and RMSE. These are the most relevant final metrics for the main volatility-forecasting story.

Secondary findings:

- `ARIMA-GARCH` is close to `GARCH(1,1)` but does not improve on it.
- The ARIMA mean component adds limited incremental forecasting value in this experiment.
- The base `LSTM` underperforms GARCH, likely because the dataset has a limited number of daily observations, the squared-return volatility target is noisy, and the signal-to-noise ratio is weak.
- The initial `ARIMA-GARCH-LSTM` hybrid does not consistently improve over the base `LSTM`.
- Tuned neural models show that target and loss design matter, but they do not overturn the main conclusion.
- Log-target training improves MAE but can underpredict volatility spikes, resulting in poor QLIKE.

The final report should avoid claiming that the hybrid model is best. A more accurate interpretation is that the hybrid design is empirically tested, but traditional GARCH remains difficult to beat in this applied daily-data setting.

## 13. How to Tell the Paper Story

Recommended narrative:

1. Volatility forecasting is important for risk management, asset allocation, and market monitoring.
2. VN-Index daily returns exhibit time-varying volatility in visual EDA.
3. Traditional GARCH models are designed for volatility clustering and are natural benchmarks.
4. LSTM and hybrid models are evaluated to test whether nonlinear sequence learning improves one-step-ahead volatility forecasts.
5. Empirical results show that `GARCH(1,1)` remains the best model by QLIKE and RMSE on the common test window.
6. Neural and hybrid models provide useful robustness evidence but do not consistently outperform econometric models.
7. The study contributes an applied, reproducible comparison for VN-Index volatility forecasting.

A more neutral title could be:

**"Modeling and Forecasting VN-Index Volatility: Evidence from GARCH, LSTM, and Hybrid Approaches"**

Do not change the repository title automatically. The neutral title is only a possible paper-title refinement.

## 14. Discussion Points

- Why GARCH works well: it is directly designed for conditional heteroskedasticity and volatility persistence.
- Why ARIMA-GARCH does not beat GARCH: daily mean return dynamics appear to add limited incremental information for variance forecasting.
- Why LSTM struggles: daily data provide a relatively small sample for neural sequence learning, the squared-return target is noisy, and overfitting risk is material.
- Why hybrid struggles: econometric features may help on validation, but the added features do not necessarily generalize to the test period.
- Why MAE and QLIKE can disagree: MAE rewards low average absolute error, while QLIKE strongly penalizes underprediction of volatility spikes.
- Why common-window evaluation matters: neural models lose initial rows due to sequence construction, so available-sample comparisons use different observations.
- Why the result is still useful: a negative or mixed finding for the hybrid model is valid empirical evidence and helps define realistic model expectations for daily VN-Index volatility forecasting.

## 15. Limitations

- Volatility is proxied using squared daily percentage log returns, which are noisy.
- The dataset uses daily frequency only, not intraday realized volatility.
- The LSTM is trained on a relatively small number of daily observations.
- Hyperparameter tuning is limited.
- The test period may represent a specific market regime.
- Results are specific to VN-Index and this CafeF dataset.
- The study evaluates one-step-ahead forecasts only.
- The current inspected artifacts do not include formal diagnostic test result tables for stationarity, autocorrelation, or ARCH effects.

## 16. Future Work

Possible extensions:

- Try additional GARCH-family models: GARCH-t, EGARCH, GJR-GARCH, and TARCH.
- Try range-based volatility proxies using OHLC data: Parkinson, Garman-Klass, and Rogers-Satchell.
- Try multi-GARCH-feature LSTM models using several econometric volatility specifications as inputs.
- Try a residual or multiplicative correction hybrid, for example `final_pred_var = GARCH_pred_var * exp(LSTM_adjustment)`.
- Try regime-specific evaluation for calm, crisis, and post-crisis market periods.
- Try rolling-window or expanding-window refit experiments.
- Try simpler feedforward neural models as a benchmark against LSTM.
- Try more robust neural architectures only after establishing stronger non-neural and simple-neural baselines.
- Add formal diagnostic-test outputs if the final paper needs numerical stationarity, autocorrelation, or ARCH evidence.

## 17. Figures and Tables Inventory

### Paper-ready figures

| Figure path | Suggested caption | Suggested paper section |
|---|---|---|
| `outputs/figures/fig_close_price.png` | Daily closing level of the VN-Index from 2010 to 2025. | Data Description / EDA |
| `outputs/figures/fig_log_return.png` | Daily percentage log returns of the VN-Index from 2010 to 2025. | Data Description / EDA |
| `outputs/figures/fig_squared_return.png` | Squared daily percentage log returns used as the volatility proxy. | Data Description / EDA |
| `outputs/figures/fig_rolling_volatility.png` | Rolling volatility of VN-Index log returns over 5-, 10-, and 20-session windows. | Data Description / EDA |
| `outputs/figures/fig_qlike_comparison_common_test.png` | Common-window test QLIKE by model. Lower values indicate better volatility forecasts. | Empirical Results |
| `outputs/figures/fig_rmse_comparison_common_test.png` | Common-window test RMSE by model. | Empirical Results |
| `outputs/figures/fig_mae_comparison_common_test.png` | Common-window test MAE by model. | Empirical Results |
| `outputs/figures/fig_actual_vs_predicted_test_common.png` | Actual and predicted variance on the common test window. | Empirical Results / Model Comparison |
| `outputs/figures/fig_actual_vs_predicted_test_common_zoom.png` | Zoomed actual and predicted variance on the common test window. | Empirical Results / Robustness |
| `outputs/figures/lstm_tuned/fig_lstm_tuned_qlike_comparison.png` | Validation and test QLIKE for tuned LSTM and hybrid variants. | Neural Tuning / Robustness |
| `outputs/figures/fig_lstm_base_loss.png` | Training and validation loss for the base LSTM model. | Appendix / Neural Training Diagnostics |
| `outputs/figures/fig_lstm_hybrid_loss.png` | Training and validation loss for the ARIMA-GARCH-LSTM model. | Appendix / Neural Training Diagnostics |
| `outputs/figures/lstm_tuned/fig_loss_lstm_logtarget.png` | Training history for `LSTM-LogTarget`. | Appendix |
| `outputs/figures/lstm_tuned/fig_loss_hybrid_logtarget.png` | Training history for `Hybrid-LogTarget`. | Appendix |
| `outputs/figures/lstm_tuned/fig_loss_lstm_qlike.png` | Training history for `LSTM-QLIKE`. | Appendix |
| `outputs/figures/lstm_tuned/fig_loss_hybrid_qlike.png` | Training history for `Hybrid-QLIKE`. | Appendix |
| `outputs/figures/lstm_tuned/fig_loss_lstm_logtarget_small.png` | Training history for `LSTM-LogTarget-Small`. | Appendix |
| `outputs/figures/lstm_tuned/fig_loss_hybrid_logtarget_small.png` | Training history for `Hybrid-LogTarget-Small`. | Appendix |

### Paper-ready tables

| Table path | Suggested caption | Suggested paper section |
|---|---|---|
| `outputs/tables/table_model_comparison_common.md` | Common-window test comparison of volatility forecasting models. | Empirical Results |
| `outputs/tables/table_model_comparison_common.tex` | LaTeX version of the common-window test comparison. | Empirical Results |
| `outputs/tables/table_model_comparison_available.md` | Available-sample test comparison of volatility forecasting models. | Appendix / Robustness |
| `outputs/tables/table_model_comparison_available.tex` | LaTeX version of the available-sample test comparison. | Appendix / Robustness |
| `outputs/metrics/data_summary.csv` | Data summary and chronological split counts. | Data |
| `outputs/metrics/arima_order_selection.csv` | ARIMA order selection by training AIC. | Methodology / Appendix |
| `outputs/metrics/econometric_metrics_test_only.csv` | Test metrics for baselines, GARCH, and ARIMA-GARCH. | Results / Appendix |
| `outputs/metrics/lstm_metrics_test_only.csv` | Test metrics for base LSTM and ARIMA-GARCH-LSTM. | Results / Appendix |
| `outputs/metrics/lstm_tuned/lstm_tuned_metrics_test_only.csv` | Test metrics for tuned neural variants. | Neural Tuning / Appendix |
| `outputs/metrics/lstm_tuned/lstm_tuned_selection_summary.csv` | Validation-selected tuned neural model summary. | Neural Tuning / Appendix |

### Prediction artifacts

Main prediction files inspected:

- `outputs/predictions/pred_baseline_mean.csv`
- `outputs/predictions/pred_rolling_vol_5.csv`
- `outputs/predictions/pred_rolling_vol_10.csv`
- `outputs/predictions/pred_rolling_vol_20.csv`
- `outputs/predictions/pred_garch_11.csv`
- `outputs/predictions/pred_arima_garch.csv`
- `outputs/predictions/pred_lstm_base.csv`
- `outputs/predictions/pred_lstm_hybrid.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_logtarget.csv`
- `outputs/predictions/lstm_tuned/pred_hybrid_logtarget.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_qlike.csv`
- `outputs/predictions/lstm_tuned/pred_hybrid_qlike.csv`
- `outputs/predictions/lstm_tuned/pred_lstm_logtarget_small.csv`
- `outputs/predictions/lstm_tuned/pred_hybrid_logtarget_small.csv`

## 18. Paper Section Drafting Plan

Suggested final paper structure:

1. **Introduction**
   Use Sections 1, 2, 12, and 13. Emphasize volatility forecasting, VN-Index relevance, and the empirical comparison objective.

2. **Literature Review**
   Use Section 2 as positioning. Discuss traditional volatility models, neural sequence models, and hybrid forecasting approaches from external literature. Keep the repository evidence separate from literature claims.

3. **Data**
   Use Sections 3, 4, 5, and 7. Describe CafeF data, variable construction, split design, and visual EDA.

4. **Methodology**
   Use Sections 6, 8, and 9. Explain leakage controls, model groups, target definition, and metrics.

5. **Empirical Results**
   Use Sections 10 and 11. Lead with common-window test results, then discuss tuned neural variants and available-sample results only as secondary evidence.

6. **Discussion**
   Use Sections 12 and 14. Interpret why GARCH remains strong and why neural/hybrid models do not consistently improve forecasts.

7. **Conclusion**
   Use Sections 12, 15, and 16. Summarize the applied empirical contribution, state limitations, and propose future extensions.

## 19. Claims Allowed and Claims to Avoid

### Allowed claims

- The project evaluates econometric, LSTM, and hybrid volatility forecasting models.
- The target is one-step-ahead VN-Index volatility, proxied by next-day squared percentage log return.
- `GARCH(1,1)` performs best on the common test window by QLIKE and RMSE.
- Among the original core models, `GARCH(1,1)` also performs best by MAE.
- `ARIMA-GARCH` is close to `GARCH(1,1)` but does not improve it.
- Neural models are sensitive to target and loss design.
- Log-target neural models can improve MAE while worsening QLIKE.
- Hybrid models do not consistently outperform traditional GARCH in this experiment.
- Common-window evaluation is preferred because all models are evaluated on the same test observations.

### Claims to avoid

- The hybrid model is universally superior.
- The paper proposes a completely novel architecture.
- LSTM outperforms GARCH in the final common-window comparison.
- Results generalize to all stock markets.
- The model predicts the VN-Index price level.
- Low MAE alone proves superior volatility forecasting performance.
- Statistical test results exist unless their output files are generated and cited.

## Missing or Unavailable Artifacts

The following requested or referenced artifacts were not found during inspection:

- `outputs/metrics/missing_values.csv`

Notes:

- Missingness information is still available in `outputs/metrics/data_summary.csv`; all essential `model_ready`, train, validation, and test variables have zero missing values there.
- `README.md` lists some older generic output names, but the current final evaluation artifacts use common-window names such as `fig_qlike_comparison_common_test.png` and `final_model_comparison_common_test_only.csv`.
