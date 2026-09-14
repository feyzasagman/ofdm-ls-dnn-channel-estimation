"""Tests for major-revision infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.metrics import ber
from src.core.seed import set_global_seed
from src.core.snr_gain import compute_snr_gain_at_target_ber
from src.core.split import stratified_split_indices, validate_split_disjoint
from src.estimators.lmmse_flat_estimator import estimate_lmmse_flat_channel
from src.estimators.ls_estimator import estimate_ls_channel
from src.estimators.mmse_estimator import estimate_simplified_mmse_channel


def test_set_global_seed_reproducible_numpy() -> None:
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_stratified_split_disjoint_and_balanced() -> None:
    snr = np.array([-10, -10, -10, 0, 0, 0, 10, 10, 10], dtype=float)
    split = stratified_split_indices(snr, test_ratio=0.33, val_ratio=0.33, random_seed=7)
    validate_split_disjoint(split)
    for part in split.values():
        for snr_val in np.unique(snr):
            assert np.sum(snr[part] == snr_val) >= 1


def test_ls_pilot_identity() -> None:
    n = 64
    p_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56])
    p_sym = np.ones(p_idx.size, dtype=np.complex128)
    h = 0.7 + 0.2j
    rx = h * np.ones(n, dtype=np.complex128)
    h_hat, _ = estimate_ls_channel(rx, p_idx, p_sym, n)
    assert np.allclose(h_hat, h, atol=1e-10)


def test_simplified_mmse_shrinkage_high_snr() -> None:
    n = 64
    p_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56])
    p_sym = np.ones(p_idx.size, dtype=np.complex128)
    h = 0.9 + 0.1j
    rx = h * np.ones(n, dtype=np.complex128)
    h_hat, _ = estimate_simplified_mmse_channel(rx, p_idx, p_sym, n, noise_variance=1e-6)
    assert np.allclose(h_hat, h, atol=1e-2)


def test_lmmse_flat_matches_scalar() -> None:
    n = 64
    p_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56])
    p_sym = np.ones(p_idx.size, dtype=np.complex128)
    h = 0.5 - 0.3j
    rx = h * np.ones(n, dtype=np.complex128)
    h_full, h_scalar = estimate_lmmse_flat_channel(rx, p_idx, p_sym, n, noise_variance=0.01)
    assert np.allclose(h_full, h_scalar)
    assert np.all(h_full == h_full[0])


def test_ber_zero_and_nonzero() -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert ber(bits, bits) == 0.0
    assert ber(bits, 1 - bits) == 1.0


def test_snr_gain_interpolation() -> None:
    snr = np.array([0, 5, 10, 15, 20], dtype=float)
    ber_ls = np.array([0.2, 0.05, 0.01, 0.001, 0.0001])
    ber_dnn = np.array([0.1, 0.02, 0.005, 0.0005, 0.00005])
    out = compute_snr_gain_at_target_ber(snr, ber_ls, ber_dnn)
    assert out["target_ber_0.01"]["status"] == "estimable"
    assert out["target_ber_0.01"]["snr_gain_db"] is not None


def test_snr_gain_not_estimable() -> None:
    snr = np.array([0, 5, 10], dtype=float)
    ber_ls = np.array([1e-5, 1e-6, 1e-7])
    ber_dnn = np.array([1e-5, 1e-6, 1e-7])
    out = compute_snr_gain_at_target_ber(snr, ber_ls, ber_dnn, target_bers=(1e-1,))
    assert out["target_ber_0.1"]["status"] == "not estimable within simulated SNR range"
