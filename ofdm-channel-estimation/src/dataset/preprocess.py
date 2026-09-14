"""Preprocessing utilities for DNN-ready channel estimation datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.split import (
    build_split_report,
    save_split_report,
    stratified_split_indices,
    validate_split_disjoint,
)


def preprocess_for_dnn(
    raw_dataset_path: str | Path,
    residual_mode: bool = False,
    normalize: bool = True,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    output_path: str | Path | None = None,
    split_report_path: str | Path | None = None,
) -> Path:
    """Load raw NPZ, prepare DNN tensors, split, optionally normalize, save NPZ.

    Inputs:
        X = ls_estimate
    Targets:
        Y = true_channel                  (default)
        Y = true_channel - ls_estimate    (residual_mode=True)
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")

    raw_path = Path(raw_dataset_path)
    data = np.load(raw_path, allow_pickle=False)
    ls_est = np.asarray(data["ls_estimate"], dtype=np.complex128)
    true_channel = np.asarray(data["true_channel"], dtype=np.complex128)
    snr_db = np.asarray(data["snr_db"], dtype=np.float64)

    if ls_est.shape != true_channel.shape:
        raise ValueError("ls_estimate and true_channel must have the same shape.")
    if ls_est.ndim != 2:
        raise ValueError("Expected 2D arrays with shape (n_samples, n_subcarriers).")

    x_complex = ls_est
    if residual_mode:
        y_complex = true_channel - ls_est
    else:
        y_complex = true_channel

    x = _complex_to_features(x_complex)
    y = _complex_to_features(y_complex)

    split = stratified_split_indices(
        snr_db=snr_db,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    validate_split_disjoint(split)
    split_report = build_split_report(snr_db, split, random_seed)
    if split_report_path is not None:
        save_split_report(split_report, split_report_path)

    x_train = x[split["train"]]
    y_train = y[split["train"]]
    x_val = x[split["val"]]
    y_val = y[split["val"]]
    x_test = x[split["test"]]
    y_test = y[split["test"]]

    snr_train = snr_db[split["train"]]
    snr_val = snr_db[split["val"]]
    snr_test = snr_db[split["test"]]

    if normalize:
        x_mean, x_std = _fit_normalizer(x_train)
        y_mean, y_std = _fit_normalizer(y_train)
        x_train = (x_train - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        x_mean = np.zeros(x.shape[1], dtype=np.float64)
        x_std = np.ones(x.shape[1], dtype=np.float64)
        y_mean = np.zeros(y.shape[1], dtype=np.float64)
        y_std = np.ones(y.shape[1], dtype=np.float64)

    save_path = _resolve_processed_output_path(raw_path, output_path, residual_mode)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        snr_train=snr_train,
        snr_val=snr_val,
        snr_test=snr_test,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        residual_mode=bool(residual_mode),
        normalize=bool(normalize),
        n_features=x.shape[1],
        train_indices=split["train"],
        val_indices=split["val"],
        test_indices=split["test"],
        split_random_seed=int(random_seed),
        split_strategy="snr_stratified",
    )
    return save_path


def _complex_to_features(x_complex: np.ndarray) -> np.ndarray:
    """Convert complex matrix to real-valued feature matrix [Re, Im]."""
    return np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1).astype(np.float64)


def _fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise standardization parameters."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _split_indices(
    n_samples: int,
    test_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Deprecated random split kept for backward compatibility."""
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_samples)

    n_test = max(1, int(round(n_samples * test_ratio)))
    remaining = n_samples - n_test
    n_val = max(1, int(round(remaining * val_ratio))) if val_ratio > 0 else 0
    n_train = remaining - n_val

    if n_train <= 0:
        raise ValueError("Split ratios leave no training samples.")

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _resolve_processed_output_path(
    raw_path: Path,
    output_path: str | Path | None,
    residual_mode: bool,
) -> Path:
    """Resolve output NPZ path under data/processed when not provided."""
    if output_path is not None:
        return Path(output_path)
    project_root = Path(__file__).resolve().parents[2]
    suffix = "residual" if residual_mode else "direct"
    filename = f"{raw_path.stem}_{suffix}_processed.npz"
    return project_root / "data" / "processed" / filename
