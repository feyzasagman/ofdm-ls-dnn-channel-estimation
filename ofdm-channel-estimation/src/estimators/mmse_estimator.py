"""Simplified (Wiener-shrinkage) channel estimation from OFDM pilots.

This is NOT full covariance-based pilot-domain LMMSE. See
``estimate_lmmse_flat_channel`` in ``lmmse_flat_estimator.py`` for the joint
scalar LMMSE baseline under single-tap flat fading.
"""

from __future__ import annotations

import numpy as np


def estimate_simplified_mmse_channel(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pilot Wiener shrinkage + linear interpolation (simplified MMSE).

    Assumptions:
        - Oracle noise variance ``noise_variance`` is available at the receiver.
        - Each pilot LS estimate is treated independently for shrinkage.
        - Real/imag parts are linearly interpolated across subcarriers.

    Steps:
      1) Pilot LS: H_ls,p = Y_p / X_p
      2) Per-pilot shrinkage: alpha_p = |H_ls,p|^2 / (|H_ls,p|^2 + sigma_n^2/|X_p|^2)
      3) H_shrunk,p = alpha_p * H_ls,p
      4) Interpolate shrunk pilots to all subcarriers.
    """
    y = np.asarray(rx_freq, dtype=np.complex128).ravel()
    p_idx = np.asarray(pilot_indices, dtype=int).ravel()
    p_sym = np.asarray(pilot_symbols, dtype=np.complex128).ravel()

    _validate_inputs(y, p_idx, p_sym, n_subcarriers, noise_variance)

    order = np.argsort(p_idx)
    p_idx = p_idx[order]
    p_sym = p_sym[order]

    y_pilots = y[p_idx]
    h_ls_pilots = y_pilots / p_sym

    eps = 1e-12
    p_h = np.abs(h_ls_pilots) ** 2
    p_n = float(noise_variance) / (np.abs(p_sym) ** 2 + eps)
    alpha = p_h / (p_h + p_n + eps)
    h_mmse_pilots = alpha * h_ls_pilots

    h_mmse_full = _interpolate_complex(h_mmse_pilots, p_idx, n_subcarriers)
    return h_mmse_full, h_mmse_pilots


def _interpolate_complex(
    pilot_values: np.ndarray,
    pilot_indices: np.ndarray,
    n_subcarriers: int,
) -> np.ndarray:
    """Linearly interpolate real/imag parts separately over all carriers."""
    x_all = np.arange(n_subcarriers, dtype=float)
    x_p = pilot_indices.astype(float)

    if pilot_values.size == 1:
        return np.full(n_subcarriers, pilot_values[0], dtype=np.complex128)

    real_interp = np.interp(x_all, x_p, np.real(pilot_values))
    imag_interp = np.interp(x_all, x_p, np.imag(pilot_values))
    return real_interp + 1j * imag_interp


def _validate_inputs(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
) -> None:
    """Validate MMSE-like estimator inputs."""
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")
    if rx_freq.size != n_subcarriers:
        raise ValueError("rx_freq length must match n_subcarriers.")
    if pilot_indices.size == 0:
        raise ValueError("pilot_indices must be non-empty.")
    if pilot_indices.size != pilot_symbols.size:
        raise ValueError("pilot_indices and pilot_symbols must have same length.")
    if np.any((pilot_indices < 0) | (pilot_indices >= n_subcarriers)):
        raise ValueError("pilot_indices must be within [0, n_subcarriers).")
    if np.unique(pilot_indices).size != pilot_indices.size:
        raise ValueError("pilot_indices must be unique.")
    if np.any(np.abs(pilot_symbols) == 0):
        raise ValueError("pilot_symbols must be non-zero.")
    if not np.isfinite(noise_variance) or noise_variance < 0.0:
        raise ValueError("noise_variance must be a finite non-negative float.")


def estimate_mmse_channel(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for ``estimate_simplified_mmse_channel``."""
    return estimate_simplified_mmse_channel(
        rx_freq=rx_freq,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=n_subcarriers,
        noise_variance=noise_variance,
    )
