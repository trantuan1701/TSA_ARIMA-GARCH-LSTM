# Econometric Diagnostics Report

## Computed

- ADF stationarity test on full-sample percentage log returns.
- Ljung-Box tests on percentage log returns and squared returns.
- ARCH-LM test on demeaned percentage log returns.
- GARCH parameter, p-value, persistence, and convergence diagnostics from existing pickles.
- Residual Ljung-Box and ARCH-LM diagnostics from stored standardized residuals where available.

## Key Results

- ADF on returns: statistic=-41.084280, p-value=0.
- GARCH(1,1): persistence=0.9732640270890623, convergence=converged
- ARIMA-GARCH: persistence=0.9718855047372229, convergence=converged

## Input Files Used

- `data/processed/vnindex_model_ready.csv`
- `outputs/metrics/arima_garch_summary.txt`
- `outputs/metrics/garch_summary.txt`
- `outputs/models/arima_garch_garch.pkl`
- `outputs/models/garch_11.pkl`

## Assumptions

- Stationarity and dependence tests use `data/processed/vnindex_model_ready.csv`.
- Ljung-Box tests are reported at lags [10, 20].
- ARCH-LM tests use 10 lags when enough observations are available.
- ARCH-LM inputs are demeaned before testing for conditional heteroskedasticity.
- GARCH residual diagnostics use training-sample standardized residuals stored in fitted artifacts.
- The ARIMA-GARCH GARCH leg is a zero-mean residual model, so `mu` and `mu_p_value` are marked `not_available` for that row.
- Ljung-Box rows report p-values directly; ADF-only critical-value columns are marked `not_available` for Ljung-Box rows.

## Residual Sources

- GARCH(1,1): std_resid attribute, n=2471.
- ARIMA-GARCH: std_resid attribute, n=2471.

## Unavailable Diagnostics

None.

## Output Tables

- `outputs/tables/table_stationarity_tests.csv`
- `outputs/tables/table_arch_lm_tests.csv`
- `outputs/tables/table_garch_diagnostics.csv`
