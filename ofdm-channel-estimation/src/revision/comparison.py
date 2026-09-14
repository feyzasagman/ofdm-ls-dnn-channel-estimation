"""Revision experiment comparison utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow import keras

from src.core.metrics import mse, nmse
from src.dnn.evaluate import _features_to_complex


def compare_estimators_on_test(
    raw_data: dict[str, np.ndarray],
    test_indices: np.ndarray,
    dnn_pred: np.ndarray,
) -> dict[str, Any]:
    """Compare LS, simplified MMSE, LMMSE flat, and LS+DNN on test split."""
    test_idx = np.asarray(test_indices, dtype=int)
    true_test = raw_data["true_channel"][test_idx]
    ls_test = raw_data["ls_estimate"][test_idx]
    simp_test = raw_data["simplified_mmse_estimate"][test_idx]
    lmmse_test = raw_data["lmmse_flat_estimate"][test_idx]
    snr_test = raw_data["snr_db"][test_idx]

    overall = {
        "ls_mse": float(mse(true_test, ls_test)),
        "ls_nmse": float(nmse(true_test, ls_test)),
        "simplified_mmse_mse": float(mse(true_test, simp_test)),
        "simplified_mmse_nmse": float(nmse(true_test, simp_test)),
        "lmmse_flat_mse": float(mse(true_test, lmmse_test)),
        "lmmse_flat_nmse": float(nmse(true_test, lmmse_test)),
        "ls_dnn_mse": float(mse(true_test, dnn_pred)),
        "ls_dnn_nmse": float(nmse(true_test, dnn_pred)),
    }

    per_snr_rows: list[dict[str, float]] = []
    for snr_val in np.sort(np.unique(snr_test)):
        mask = snr_test == snr_val
        row = {
            "snr_db": float(snr_val),
            "ls_mse": float(mse(true_test[mask], ls_test[mask])),
            "ls_nmse": float(nmse(true_test[mask], ls_test[mask])),
            "simplified_mmse_mse": float(mse(true_test[mask], simp_test[mask])),
            "simplified_mmse_nmse": float(nmse(true_test[mask], simp_test[mask])),
            "lmmse_flat_mse": float(mse(true_test[mask], lmmse_test[mask])),
            "lmmse_flat_nmse": float(nmse(true_test[mask], lmmse_test[mask])),
            "ls_dnn_mse": float(mse(true_test[mask], dnn_pred[mask])),
            "ls_dnn_nmse": float(nmse(true_test[mask], dnn_pred[mask])),
        }
        per_snr_rows.append(row)

    return {"overall": overall, "per_snr": per_snr_rows}


def predict_dnn_test_channels(
    model: keras.Model,
    processed_data: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict complex channel estimates on processed test split."""
    x_test = np.asarray(processed_data["x_test"], dtype=np.float32)
    y_test = np.asarray(processed_data["y_test"], dtype=np.float32)
    x_mean = np.asarray(processed_data["x_mean"], dtype=np.float64)
    x_std = np.asarray(processed_data["x_std"], dtype=np.float64)
    y_mean = np.asarray(processed_data["y_mean"], dtype=np.float64)
    y_std = np.asarray(processed_data["y_std"], dtype=np.float64)
    normalize = bool(np.asarray(processed_data["normalize"]).item())
    residual_mode = bool(np.asarray(processed_data["residual_mode"]).item())

    y_pred = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64)
    if normalize:
        x_den = x_test.astype(np.float64) * x_std + x_mean
        y_den = y_pred * y_std + y_mean
    else:
        x_den = x_test.astype(np.float64)
        y_den = y_pred

    if residual_mode:
        feat = x_den + y_den
    else:
        feat = y_den
    return _features_to_complex(feat)
