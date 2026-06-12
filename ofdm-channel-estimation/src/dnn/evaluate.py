"""Evaluation utilities for trained DNN channel regressors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.core.metrics import mse, nmse


def evaluate_dnn(
    processed_dataset_path: str | Path,
    model_path: str | Path,
    residual_mode: bool | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float | Path]:
    """Evaluate trained model on test split and save predictions/metrics.

    If residual mode is active:
        final_prediction = ls_input + residual_pred
    """
    data = np.load(Path(processed_dataset_path), allow_pickle=False)
    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    x_mean = np.asarray(data["x_mean"], dtype=np.float64)
    x_std = np.asarray(data["x_std"], dtype=np.float64)
    y_mean = np.asarray(data["y_mean"], dtype=np.float64)
    y_std = np.asarray(data["y_std"], dtype=np.float64)
    normalize = bool(np.asarray(data["normalize"]).item())
    residual_from_data = bool(np.asarray(data["residual_mode"]).item())
    snr_test = np.asarray(data["snr_test"], dtype=np.float64)

    residual_flag = residual_from_data if residual_mode is None else bool(residual_mode)

    model = keras.models.load_model(Path(model_path))
    y_pred = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64)

    if normalize:
        x_test_denorm = x_test.astype(np.float64) * x_std + x_mean
        y_test_denorm = y_test.astype(np.float64) * y_std + y_mean
        y_pred_denorm = y_pred * y_std + y_mean
    else:
        x_test_denorm = x_test.astype(np.float64)
        y_test_denorm = y_test.astype(np.float64)
        y_pred_denorm = y_pred

    if residual_flag:
        # y_* are residuals, convert predictions back to channel estimate.
        final_pred_feat = x_test_denorm + y_pred_denorm
        final_true_feat = x_test_denorm + y_test_denorm
    else:
        final_pred_feat = y_pred_denorm
        final_true_feat = y_test_denorm

    pred_complex = _features_to_complex(final_pred_feat)
    true_complex = _features_to_complex(final_true_feat)

    test_mse = mse(true_complex, pred_complex)
    test_nmse = nmse(true_complex, pred_complex)

    per_snr_nmse = _compute_nmse_per_snr(true_complex, pred_complex, snr_test)
    save_path = _resolve_eval_output_path(processed_dataset_path, output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        pred_channel=pred_complex,
        true_channel=true_complex,
        snr_test=snr_test,
        mse=float(test_mse),
        nmse=float(test_nmse),
    )

    metrics_path = save_path.with_suffix(".json")
    metrics_payload = {
        "mse": float(test_mse),
        "nmse": float(test_nmse),
        "residual_mode": residual_flag,
        "per_snr_nmse": per_snr_nmse,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "mse": float(test_mse),
        "nmse": float(test_nmse),
        "results_path": save_path,
        "metrics_path": metrics_path,
    }


def _features_to_complex(features: np.ndarray) -> np.ndarray:
    """Convert [Re, Im] feature matrix back to complex matrix."""
    if features.ndim != 2:
        raise ValueError("features must be 2D.")
    if features.shape[1] % 2 != 0:
        raise ValueError("features second dimension must be even ([Re, Im]).")
    half = features.shape[1] // 2
    return features[:, :half] + 1j * features[:, half:]


def _compute_nmse_per_snr(
    true_complex: np.ndarray,
    pred_complex: np.ndarray,
    snr_test: np.ndarray,
) -> dict[str, float]:
    """Compute NMSE separately for each SNR value in the test set."""
    out: dict[str, float] = {}
    unique_snr = np.unique(snr_test)
    for snr in unique_snr:
        mask = snr_test == snr
        out[str(float(snr))] = float(nmse(true_complex[mask], pred_complex[mask]))
    return out


def _resolve_eval_output_path(
    processed_dataset_path: str | Path,
    output_path: str | Path | None,
) -> Path:
    """Resolve evaluation output path."""
    if output_path is not None:
        return Path(output_path)
    project_root = Path(__file__).resolve().parents[2]
    stem = Path(processed_dataset_path).stem
    return project_root / "data" / "results" / "logs" / f"{stem}_eval.npz"
