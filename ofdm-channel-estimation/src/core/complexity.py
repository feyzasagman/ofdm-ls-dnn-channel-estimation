"""Computational complexity measurements for channel estimators."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

from src.estimators.lmmse_flat_estimator import estimate_lmmse_flat_channel
from src.estimators.ls_estimator import estimate_ls_channel
from src.estimators.mmse_estimator import estimate_simplified_mmse_channel


def measure_estimator_inference_time(
    estimator_fn,
    n_warmup: int = 10,
    n_reps: int = 100,
) -> dict[str, float]:
    """Measure average/std inference time in seconds."""
    for _ in range(n_warmup):
        estimator_fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        estimator_fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_reps": int(n_reps),
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr)),
    }


def measure_dnn_inference(
    model: keras.Model,
    x_sample: np.ndarray,
    n_warmup: int = 10,
    n_reps: int = 100,
) -> dict[str, float]:
    """Measure DNN inference time with warm-up."""
    x = np.asarray(x_sample, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    for _ in range(n_warmup):
        model.predict(x, verbose=0)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_reps": int(n_reps),
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr)),
    }


def build_complexity_report(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
    model: keras.Model | None = None,
    x_sample: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build complexity report for LS, simplified MMSE, LMMSE flat, and DNN."""
    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "unknown",
        },
        "estimators": {},
    }

    report["estimators"]["ls"] = measure_estimator_inference_time(
        lambda: estimate_ls_channel(rx_freq, pilot_indices, pilot_symbols, n_subcarriers)
    )
    report["estimators"]["simplified_mmse"] = measure_estimator_inference_time(
        lambda: estimate_simplified_mmse_channel(
            rx_freq, pilot_indices, pilot_symbols, n_subcarriers, noise_variance
        )
    )
    report["estimators"]["lmmse_flat"] = measure_estimator_inference_time(
        lambda: estimate_lmmse_flat_channel(
            rx_freq, pilot_indices, pilot_symbols, n_subcarriers, noise_variance
        )
    )

    if model is not None and x_sample is not None:
        report["estimators"]["ls_dnn"] = measure_dnn_inference(model, x_sample)
        report["ls_dnn_model"] = {
            "total_params": int(model.count_params()),
            "trainable_params": int(
                sum(int(np.prod(w.shape)) for w in model.trainable_weights)
            ),
        }
    return report


def save_complexity_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
