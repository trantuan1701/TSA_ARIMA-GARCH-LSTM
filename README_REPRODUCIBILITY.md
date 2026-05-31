# Reproducibility Notes

This project forecasts one-step-ahead VN-Index volatility using daily CafeF OHLC data. The primary target is next-day squared percentage log return:

```text
log_return_pct = 100 * log(close_t / close_{t-1})
squared_return = log_return_pct ** 2
target_var_next = squared_return.shift(-1)
```

## Repository And Data Availability

- Git repository: yes.
- Verified branch: `master`.
- Verified remote: `https://github.com/trantuan1701/TSA_ARIMA-GARCH-LSTM.git`.
- Remote reachability was checked with `git ls-remote origin master`.
- Release snapshot: use the commit or tag that contains the packaged root `paper.pdf`, the rewritten root `README.md`, and `outputs/artifact_manifest.csv`.
- Raw CafeF data are present locally under `data/raw/`, but no redistribution license was found in this repository. If this work is published publicly, add a data-license statement or replace raw redistribution with download/preprocessing instructions.

## Environment

Use Python 3.13.11 or a compatible Python 3.10+ environment. The verified package set is pinned in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Verified versions include pandas 3.0.2, numpy 2.4.4, statsmodels 0.14.6, arch 8.0.0, TensorFlow 2.21.0, scipy 1.17.1, pyarrow 24.0.0, scikit-learn 1.8.0, and pytest 9.0.3.

## Seeds And Determinism

- Base and tuned neural scripts use seed `42`.
- Set through `VNINDEX_RANDOM_SEED=42`; scripts also set Python, NumPy, and TensorFlow seeds.
- TensorFlow deterministic options are enabled where available: `TF_DETERMINISTIC_OPS=1` and `TF_ENABLE_ONEDNN_OPTS=0`.
- CPU TensorFlow may still emit hardware warnings; regenerated predictions are versioned as artifacts.

## Reproduction Commands

Baseline data/model pipeline:

```bash
PYTHONDONTWRITEBYTECODE=1 python run_experiment.py --skip-paper
```

Advanced and final econometric artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 python src/stage1_extensions.py all
PYTHONDONTWRITEBYTECODE=1 python src/final_financial_econometrics.py all
```

Corrected-context neural regeneration:

```bash
make neural-corrected
make advanced-corrected-combinations
```

Completion-pass VaR/ES, regime, dynamic-combination, and manifest artifacts:

```bash
make completion-artifacts
```

Full tests and manuscript build:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
PYTHONDONTWRITEBYTECODE=1 make final-artifacts
make -C paper all
```

## Versioned Artifacts

- Baseline archive: `outputs/archive/before_neural_var_es_dynamic_completion/`, `models/archive/before_neural_var_es_dynamic_completion/`, `paper/archive/before_neural_var_es_dynamic_completion/`.
- Legacy neural archive: `outputs/neural_legacy_pre_context_fix/`, `models/neural_legacy_pre_context_fix/`.
- Corrected neural outputs: `outputs/neural_corrected_context_v2/`.
- Corrected neural-dependent calibration/combinations: `outputs/advanced_corrected_context_v2/`.
- Final completion outputs: `outputs/final/`.

## Primary Evidence Files

- `outputs/final/primary_results_corrected_neural.csv`
- `outputs/final/neural_retraining_comparison.csv`
- `outputs/final/regime_spike_results_corrected_neural.csv`
- `outputs/final/common_distribution_var_es_all_models.csv`
- `outputs/final/model_specific_var_es_garch.csv`
- `outputs/final/regime_aware_combination_results.csv`
- `outputs/audit/refit_failure_taxonomy.csv`
- `outputs/artifact_manifest.csv`
- `outputs/artifact_manifest.json`
- `paper.pdf`

Use `outputs/final/primary_results_corrected_neural.csv` for the corrected-context primary leaderboard. The earlier `N=726` baseline remains archived and is used only as a comparability check.
