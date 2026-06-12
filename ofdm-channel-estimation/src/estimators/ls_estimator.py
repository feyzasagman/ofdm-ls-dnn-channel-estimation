"""Least-squares channel estimation from OFDM pilots."""

from __future__ import annotations

import numpy as np


def estimate_ls_channel(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate channel in frequency domain using pilot LS + interpolation."""
    y = np.asarray(rx_freq, dtype=np.complex128).ravel()
    p_idx = np.asarray(pilot_indices, dtype=int).ravel()
    p_sym = np.asarray(pilot_symbols, dtype=np.complex128).ravel()

    _validate_inputs(y, p_idx, p_sym, n_subcarriers)

    order = np.argsort(p_idx)
    p_idx = p_idx[order]
    p_sym = p_sym[order]

    y_pilots = y[p_idx]
    h_ls_pilots = y_pilots / p_sym
    h_ls_full = _interpolate_complex(h_ls_pilots, p_idx, n_subcarriers)
    return h_ls_full, h_ls_pilots


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
) -> None:
    """Validate LS estimator inputs."""
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
        raise ValueError("pilot_symbols must be non-zero for LS division.")
