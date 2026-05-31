# Refit Protocol Report

## Protocols

- Fixed train-only: parameters are estimated on the training split and post-training state is updated recursively.
- Refit at validation start: validation uses the training split; the analogous test policy is `refit_train_val_before_test`.
- Expanding monthly/quarterly: refit on all observations available before the first trading day of each period.
- Rolling monthly/quarterly: refit on the latest 1000, 1500, or 2000 observations before each refit date.

## Validation Selection

- BestAsymmetric: best validation protocol `rolling_1000_monthly` with QLIKE 1.599196; delta vs fixed train-only = -0.013076.
- BestGARCHFamily: best validation protocol `fixed_train_only` with QLIKE 1.611304; delta vs fixed train-only = 0.000000.
- OriginalARIMAGARCH: best validation protocol `expanding_quarterly` with QLIKE 1.629847; delta vs fixed train-only = -0.006637.
- OriginalGARCH: best validation protocol `expanding_quarterly` with QLIKE 1.623572; delta vs fixed train-only = -0.005860.

## Test Evaluation

- BestGARCHFamily: selected `fixed_train_only`; test QLIKE 1.032548, RMSE 3.561391, MAE 1.324605.
- OriginalGARCH: selected `expanding_quarterly`; test QLIKE 1.059784, RMSE 3.637215, MAE 1.432106.
- OriginalARIMAGARCH: selected `expanding_quarterly`; test QLIKE 1.060979, RMSE 3.650333, MAE 1.434812.
- BestAsymmetric: selected `rolling_1000_monthly`; test QLIKE 1.067672, RMSE 3.554380, MAE 1.421574.

## Output Tables

- `outputs/advanced/tables/table_refit_protocol_validation.csv`
- `outputs/advanced/tables/table_refit_protocol_test.csv`
