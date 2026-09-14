"""Joint scalar LMMSE estimator for single-tap flat-fading channels."""

from __future__ import annotations

import numpy as np


def estimate_lmmse_flat_channel(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
    channel_prior_variance: float = 1.0,
) -> tuple[np.ndarray, complex]:
    """Estimate flat-fading channel using joint pilot-domain LMMSE.

    Model assumptions:
        - One scalar channel coefficient h is common to all subcarriers.
        - Prior: E[h]=0, E[|h|^2]=channel_prior_variance (Rayleigh-like prior).
        - y_p = h * x_p + n_p, with n_p ~ CN(0, noise_variance).

    Joint MMSE estimate:
        h_hat = sum_p (x_p^* y_p / sigma_n^2) / (1/sigma_h^2 + sum_p |x_p|^2 / sigma_n^2)

    For AWGN (deterministic h=1), this reduces to a noise-regularized average of
    LS pilot estimates; the prior is an approximation in that case.

    Returns full-band estimate by broadcasting scalar h_hat to all subcarriers.
    """
    y = np.asarray(rx_freq, dtype=np.complex128).ravel()
    p_idx = np.asarray(pilot_indices, dtype=int).ravel()
    p_sym = np.asarray(pilot_symbols, dtype=np.complex128).ravel()

    _validate_inputs(y, p_idx, p_sym, n_subcarriers, noise_variance, channel_prior_variance)

    y_pilots = y[p_idx]
    eps = 1e-12
    sigma_n2 = float(noise_variance)
    sigma_h2 = float(channel_prior_variance)

    numerator = np.sum(np.conj(p_sym) * y_pilots) / (sigma_n2 + eps)
    denominator = (1.0 / (sigma_h2 + eps)) + np.sum(np.abs(p_sym) ** 2) / (sigma_n2 + eps)
    h_hat = numerator / (denominator + eps)

    h_full = np.full(n_subcarriers, h_hat, dtype=np.complex128)
    return h_full, complex(h_hat)


def _validate_inputs(
    rx_freq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    noise_variance: float,
    channel_prior_variance: float,
) -> None:
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")
    if rx_freq.size != n_subcarriers:
        raise ValueError("rx_freq length must match n_subcarriers.")
    if pilot_indices.size == 0:
        raise ValueError("pilot_indices must be non-empty.")
    if pilot_indices.size != pilot_symbols.size:
        raise ValueError("pilot_indices and pilot_symbols must have same length.")
    if not np.isfinite(noise_variance) or noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative.")
    if not np.isfinite(channel_prior_variance) or channel_prior_variance <= 0.0:
        raise ValueError("channel_prior_variance must be positive.")
