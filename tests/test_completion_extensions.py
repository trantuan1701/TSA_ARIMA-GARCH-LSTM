from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.completion_extensions import (
    normal_var_es,
    quantile_loss,
    student_t_var_es,
)


def test_normal_var_es_lower_tail_formula() -> None:
    alpha = 0.05
    sigma = np.array([2.0])
    q, es = normal_var_es(alpha, sigma)
    z = NormalDist().inv_cdf(alpha)

    assert q[0] == z * 2.0
    assert es[0] == -2.0 * norm.pdf(z) / alpha
    assert es[0] < q[0] < 0.0


def test_student_t_var_es_uses_variance_standardization() -> None:
    alpha = 0.025
    df = 8.0
    sigma = np.array([1.5])
    q, es = student_t_var_es(alpha, sigma, df)
    raw_q = float(t.ppf(alpha, df))
    scale = math.sqrt((df - 2.0) / df)
    expected_q = 1.5 * scale * raw_q
    expected_es = -1.5 * scale * (df + raw_q**2) * float(t.pdf(raw_q, df)) / ((df - 1.0) * alpha)

    assert q[0] == expected_q
    assert es[0] == expected_es
    assert es[0] < q[0] < 0.0


def test_var_violation_convention_and_quantile_loss() -> None:
    returns = np.array([-3.0, -1.0, 0.5])
    var = np.array([-2.0, -2.0, -2.0])
    violations = returns < var

    assert violations.tolist() == [True, False, False]
    assert quantile_loss(returns, var, alpha=0.05) >= 0.0


def test_corrected_neural_outputs_preserve_legacy_namespace() -> None:
    legacy = Path("outputs/neural_legacy_pre_context_fix/predictions/pred_lstm_base.csv")
    corrected = Path("outputs/neural_corrected_context_v2/predictions/pred_lstm_base.csv")

    assert legacy.exists()
    assert corrected.exists()
    assert len(pd.read_csv(legacy).query("split == 'test'")) == 726
    assert len(pd.read_csv(corrected).query("split == 'test'")) == 745


def test_regime_aware_weights_are_validation_frozen() -> None:
    results = pd.read_csv("outputs/final/regime_aware_combination_results.csv")
    assert set(results["split"]) == {"validation", "test"}
    validation = results[results["split"] == "validation"].iloc[0]
    test = results[results["split"] == "test"].iloc[0]

    for column in ["threshold_quantile", "threshold_value", "low_econ_weight", "high_econ_weight"]:
        assert validation[column] == test[column]


def test_artifact_manifest_contains_corrected_namespace() -> None:
    manifest = pd.read_csv("outputs/artifact_manifest.csv")

    assert (manifest["relative_path"] == "outputs/neural_corrected_context_v2/predictions/pred_lstm_base.csv").any()
    assert (manifest["relative_path"] == "outputs/final/common_distribution_var_es_all_models.csv").any()
