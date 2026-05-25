"""Reusable volatility forecast metrics."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-8


def evaluate_volatility_predictions(
    actual_var: Sequence[float] | pd.Series | np.ndarray,
    pred_var: Sequence[float] | pd.Series | np.ndarray,
    *,
    epsilon: float = EPSILON,
    label: str | None = None,
) -> dict[str, float | int]:
    """Evaluate one-step-ahead variance predictions.

    Non-numeric, NaN, and infinite rows are excluded from metric computation.
    Predictions are clipped to ``epsilon`` before metrics are computed so QLIKE
    remains well-defined and all evaluated variances are strictly positive.
    """

    actual = pd.to_numeric(pd.Series(actual_var), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred_var), errors="coerce").to_numpy(dtype=float)

    if actual.shape[0] != pred.shape[0]:
        raise ValueError(
            f"actual_var and pred_var must have equal length; got {actual.shape[0]} "
            f"and {pred.shape[0]}."
        )

    finite_mask = np.isfinite(actual) & np.isfinite(pred)
    dropped = int((~finite_mask).sum())
    if dropped:
        metric_label = f" for {label}" if label else ""
        logging.warning(
            "Dropping %s non-finite volatility prediction rows%s before metric computation.",
            dropped,
            metric_label,
        )

    actual = actual[finite_mask]
    pred = pred[finite_mask]
    if actual.size == 0:
        raise ValueError("No finite observations remain for volatility metric computation.")

    pred_clipped = np.clip(pred, epsilon, None)
    errors = actual - pred_clipped
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    qlike = float(np.mean(np.log(pred_clipped) + actual / pred_clipped))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
        "n_obs": int(actual.size),
        "mean_actual_var": float(np.mean(actual)),
        "mean_pred_var": float(np.mean(pred_clipped)),
    }
