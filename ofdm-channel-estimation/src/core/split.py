"""Train/validation/test splitting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 999)


def stratified_split_indices(
    snr_db: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split indices with per-SNR stratification."""
    snr_arr = np.asarray(snr_db, dtype=np.float64)
    if snr_arr.ndim != 1:
        raise ValueError("snr_db must be 1D.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")

    rng = np.random.default_rng(random_seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for snr_val in np.sort(np.unique(snr_arr)):
        idx = np.flatnonzero(snr_arr == snr_val)
        if idx.size < 3:
            raise ValueError(f"Need at least 3 samples per SNR for split; SNR={snr_val} has {idx.size}.")
        perm = rng.permutation(idx)

        n_test = max(1, int(round(idx.size * test_ratio)))
        remaining = idx.size - n_test
        n_val = max(1, int(round(remaining * val_ratio))) if val_ratio > 0 else 0
        n_train = remaining - n_val
        if n_train <= 0:
            raise ValueError(f"Split ratios leave no training samples at SNR={snr_val}.")

        test_parts.append(perm[:n_test])
        val_parts.append(perm[n_test : n_test + n_val])
        train_parts.append(perm[n_test + n_val :])

    split = {
        "train": np.concatenate(train_parts) if train_parts else np.array([], dtype=int),
        "val": np.concatenate(val_parts) if val_parts else np.array([], dtype=int),
        "test": np.concatenate(test_parts) if test_parts else np.array([], dtype=int),
    }
    validate_split_disjoint(split)
    return split


def validate_split_disjoint(split: dict[str, np.ndarray]) -> None:
    """Assert no index appears in more than one split."""
    train = set(np.asarray(split["train"], dtype=int).tolist())
    val = set(np.asarray(split["val"], dtype=int).tolist())
    test = set(np.asarray(split["test"], dtype=int).tolist())
    if train & val or train & test or val & test:
        raise AssertionError("Train/val/test splits must be disjoint.")


def build_split_report(
    snr_db: np.ndarray,
    split: dict[str, np.ndarray],
    random_seed: int,
) -> dict[str, Any]:
    """Build split metadata including per-SNR counts."""
    snr_arr = np.asarray(snr_db, dtype=np.float64)
    report: dict[str, Any] = {
        "random_seed": int(random_seed),
        "split_strategy": "snr_stratified",
        "total_samples": int(snr_arr.size),
        "training_samples": int(split["train"].size),
        "validation_samples": int(split["val"].size),
        "test_samples": int(split["test"].size),
        "samples_per_snr": {},
    }
    for snr_val in np.sort(np.unique(snr_arr)):
        key = str(float(snr_val))
        report["samples_per_snr"][key] = {
            "total": int(np.sum(snr_arr == snr_val)),
            "train": int(np.sum(snr_arr[split["train"]] == snr_val)),
            "val": int(np.sum(snr_arr[split["val"]] == snr_val)),
            "test": int(np.sum(snr_arr[split["test"]] == snr_val)),
        }
    return report


def save_split_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save split metadata JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
