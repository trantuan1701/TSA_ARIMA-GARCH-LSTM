# Proxy Robustness Summary

## Stable Top Models

- AdvGARCH-BestAsymmetric: average rank 3.29, best 1, worst 13, classification `robust_top_model`.
- AdvGARCH-BestHeavyTail: average rank 4.43, best 2, worst 14, classification `robust_top_model`.
- AdvGARCH-BestARIMA: average rank 4.71, best 3, worst 10, classification `middle_ranked_model`.
- AdvGARCH-BestQLIKE: average rank 4.71, best 3, worst 10, classification `middle_ranked_model`.
- Refit-BestGARCHFamily-fixed_train_only: average rank 4.71, best 3, worst 10, classification `middle_ranked_model`.
- ComboPair-GARCH11_ARIMAGARCH: average rank 7.00, best 1, worst 15, classification `middle_ranked_model`.
- GARCH(1,1): average rank 7.00, best 1, worst 15, classification `middle_ranked_model`.
- AdvGARCH-BestSymmetric: average rank 7.14, best 2, worst 17, classification `middle_ranked_model`.
- ComboMedian-PoolA_Econometric: average rank 8.00, best 6, worst 10, classification `middle_ranked_model`.
- ARIMA-GARCH: average rank 9.71, best 3, worst 19, classification `middle_ranked_model`.

## Conclusion Safety

- GARCH(1,1) is a strong benchmark. Status: `partially_supported`. GARCH(1,1) average QLIKE rank 7.00; top-5 under 2/7 proxies.
- Advanced GARCH-family models can improve QLIKE modestly. Status: `supported`. AdvGARCH-BestQLIKE average QLIKE rank 4.71; raw DM improvement over GARCH under 4 proxies; Holm under 1 proxies.
- ARIMA mean dynamics add limited value. Status: `supported`. ARIMA-GARCH average QLIKE rank 9.71 versus GARCH 7.00.
- Neural/hybrid models do not robustly dominate econometric models. Status: `supported`. Average best neural/hybrid proxy rank 15.14; average best econometric proxy rank 1.14.
- Log-target neural variants can reduce MAE but underpredict risk. Status: `supported`. Raw log-target average pred/actual ratio 0.356; average spike underprediction 1.000.
- Results are sensitive or insensitive to the volatility proxy. Status: `partially_supported`. Average rank standard deviation across models is 3.59; top-ranked model changes across proxy definitions.

## Existing Output Integrity

- No existing files under `outputs/predictions/` or `outputs/metrics/` were modified.
