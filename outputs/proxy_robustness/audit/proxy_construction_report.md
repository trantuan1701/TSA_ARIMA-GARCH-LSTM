# Proxy Construction Report

- Price source: `processed_clean`.
- All one-day range proxies use log price ratios and are multiplied by `100^2` to match percentage-return variance forecasts.
- Yang-Zhang proxies are rolling 5-, 10-, and 20-day variance estimates, not one-day proxies.
- Tiny negative numerical values are set to zero; materially negative formula values are treated as invalid.

## Valid Observations

- close_to_close_squared_return: 3976/3989 valid observations; mean=1.391167.
- parkinson: 3977/3989 valid observations; mean=0.924217.
- garman_klass: 3977/3989 valid observations; mean=0.858668.
- rogers_satchell: 3977/3989 valid observations; mean=0.859888.
- yang_zhang_5: 3984/3989 valid observations; mean=1.214807.
- yang_zhang_10: 3979/3989 valid observations; mean=1.221719.
- yang_zhang_20: 3969/3989 valid observations; mean=1.225692.

## Invalid Rows

- all_ohlc_proxies / high_below_open_or_close: 8 rows.
- all_ohlc_proxies / low_above_open_or_close: 4 rows.
- close_to_close_squared_return / nonfinite_formula_value: 1 rows.
- garman_klass / negative_formula_value: 4 rows.
- rogers_satchell / negative_formula_value: 2 rows.
- yang_zhang_5 / rolling_window_unavailable_or_nonfinite: 5 rows.
- yang_zhang_10 / rolling_window_unavailable_or_nonfinite: 10 rows.
- yang_zhang_20 / rolling_window_unavailable_or_nonfinite: 20 rows.
