PYTHON ?= python

.PHONY: data econometric lstm tune evaluate diagnostics statistical-tests robustness empirical-rigor advanced-model-search advanced-combinations advanced-hypothesis-tests advanced-var advanced-experiments proxy-robustness neural-corrected advanced-corrected-combinations completion-artifacts artifact-manifest final-artifacts final-audit stage1-audit stage1-baselines stage1-risk stage1-diagnostics stage1-robustness stage1-test stage1-all paper test all clean-cache

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

neural-corrected:
	PYTHONDONTWRITEBYTECODE=1 VNINDEX_NEURAL_OUTPUT_ROOT=outputs/neural_corrected_context_v2 VNINDEX_RANDOM_SEED=42 python src/train_lstm_hybrid.py
	PYTHONDONTWRITEBYTECODE=1 VNINDEX_NEURAL_OUTPUT_ROOT=outputs/neural_corrected_context_v2 VNINDEX_RANDOM_SEED=42 python src/tune_lstm_hybrid.py

advanced-corrected-combinations:
	mkdir -p outputs/advanced_corrected_context_v2/predictions
	rsync -a outputs/advanced/predictions/garch_family outputs/advanced/predictions/refit_protocols outputs/advanced_corrected_context_v2/predictions/
	PYTHONDONTWRITEBYTECODE=1 VNINDEX_NEURAL_OUTPUT_ROOT=outputs/neural_corrected_context_v2 VNINDEX_ADVANCED_DIR=outputs/advanced_corrected_context_v2 python src/advanced_forecast_combinations.py --force

completion-artifacts:
	PYTHONDONTWRITEBYTECODE=1 python src/completion_extensions.py all

artifact-manifest:
	PYTHONDONTWRITEBYTECODE=1 python src/completion_extensions.py manifest

final-artifacts:
	PYTHONDONTWRITEBYTECODE=1 python src/final_financial_econometrics.py all
	PYTHONDONTWRITEBYTECODE=1 python src/completion_extensions.py all

final-audit:
	PYTHONDONTWRITEBYTECODE=1 python src/final_financial_econometrics.py audit

stage1-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py audit

stage1-baselines:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py baselines

stage1-risk:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py risk

stage1-diagnostics:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py diagnostics

stage1-robustness:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py robustness

stage1-test:
	PYTHONDONTWRITEBYTECODE=1 pytest

stage1-all:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/stage1_extensions.py all

paper:
	$(MAKE) -C paper all

test:
	PYTHONDONTWRITEBYTECODE=1 pytest

all:
	$(PYTHON) run_experiment.py

clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
