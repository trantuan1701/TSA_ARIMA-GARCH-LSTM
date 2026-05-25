# IEEE Paper Plan: VN-Index Volatility Forecasting

Scope: empirical IEEE-style paper using only repository artifacts for data, methods, metrics, figures, and claims.

User request: complete the paper end to end in this turn. The approval gate from the review-paper skill is treated as satisfied by the explicit request to inspect, write, backfill, compile, and report.

## Repository Evidence Snapshot

- README and project plan describe one-step-ahead VN-Index volatility forecasting using CafeF data.
- `docs/experiment_contract.md` defines the target, split semantics, forbidden predictors, prediction schema, canonical model names, and metrics.
- `src/` contains data preparation, econometric training, LSTM/hybrid training, tuned neural variants, metrics, and final evaluation.
- No `notebooks/`, `configs/`, `reports/`, or `latex/` directories were found.
- `paper/report_materials.md` contains a detailed evidence summary and recommended paper story.
- Processed data, predictions, metrics, and figures are present under `data/processed/` and `outputs/`.

## Main Story

The paper is an applied empirical comparison, not a new architecture claim. GARCH(1,1) is strongest overall on the common test window by QLIKE and RMSE. Tuned log-target neural variants can reduce MAE but underpredict volatility spikes and perform poorly by QLIKE.

## Section Plan

1. Introduction: motivation, VN-Index task, empirical question, main result.
2. Related Work: ARIMA, ARCH/GARCH, LSTM, volatility forecast evaluation.
3. Methodology: target definition, models, hybrid features, metrics.
4. Experimental Setup: CafeF data, chronological splits, leakage controls, implementation evidence.
5. Results and Discussion: common-window table, figures, tuned neural variants, key findings.
6. Conclusion: findings, limitations, future work.

## Planned Visuals

- Rolling volatility time series.
- Common-window QLIKE comparison.
- Actual vs predicted variance zoom.
- Tuned LSTM/hybrid validation and test QLIKE comparison.

## Completed Follow-Up

- Formal ADF, Ljung-Box, ARCH-LM, advanced-search, VaR, and OHLC proxy robustness results have been incorporated into the rewritten manuscript.
