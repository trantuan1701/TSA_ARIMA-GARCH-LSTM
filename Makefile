PYTHON ?= python

.PHONY: data econometric lstm tune evaluate diagnostics statistical-tests robustness empirical-rigor advanced-model-search advanced-combinations advanced-hypothesis-tests advanced-var advanced-experiments proxy-robustness paper test all clean-cache

data:
	$(PYTHON) src/prepare_data.py

econometric:
	$(PYTHON) src/train_econometric.py

lstm:
	$(PYTHON) src/train_lstm_hybrid.py

tune:
	$(PYTHON) src/tune_lstm_hybrid.py

evaluate:
	$(PYTHON) src/evaluate_all.py

diagnostics:
	PYTHONDONTWRITEBYTECODE=1 python src/econometric_diagnostics.py

statistical-tests:
	PYTHONDONTWRITEBYTECODE=1 python src/statistical_tests.py

robustness:
	PYTHONDONTWRITEBYTECODE=1 python src/robustness_analysis.py

empirical-rigor:
	make diagnostics
	make statistical-tests
	make robustness

advanced-model-search:
	PYTHONDONTWRITEBYTECODE=1 python src/advanced_model_search.py

advanced-combinations:
	PYTHONDONTWRITEBYTECODE=1 python src/advanced_forecast_combinations.py

advanced-hypothesis-tests:
	PYTHONDONTWRITEBYTECODE=1 python src/advanced_hypothesis_tests.py

advanced-var:
	PYTHONDONTWRITEBYTECODE=1 python src/advanced_var_backtesting.py

advanced-experiments:
	PYTHONDONTWRITEBYTECODE=1 python src/run_advanced_experiments.py

proxy-robustness:
	PYTHONDONTWRITEBYTECODE=1 python src/proxy_robustness.py

paper:
	$(MAKE) -C paper all

test:
	PYTHONDONTWRITEBYTECODE=1 pytest

all:
	$(PYTHON) run_experiment.py

clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
