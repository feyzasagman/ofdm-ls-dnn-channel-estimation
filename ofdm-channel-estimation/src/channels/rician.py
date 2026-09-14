"""Flat Rician fading channel model."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_rician_flat_fading(
    tx_signal: np.ndarray,
    snr_db: float,
    k_factor_db: float = 6.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, complex, float]:
    """Apply single-tap Rician flat fading and AWGN.

    Channel model (single-tap flat fading):
        h = h_los + h_scatter

    Rician K-factor (linear): K = P_los / P_scatter, with K_dB = 10*log10(K).

    Component scaling for E[|h|^2] ≈ 1:
        |h_los| = sqrt(K / (K + 1))
        h_scatter ~ CN(0, 1 / (2*(K + 1)))  (circularly symmetric)
    """
    x = np.asarray(tx_signal, dtype=np.complex128)
    if x.size == 0:
        raise ValueError("tx_signal must be non-empty.")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    if not np.isfinite(k_factor_db):
        raise ValueError("k_factor_db must be finite.")

    generator = rng if rng is not None else np.random.default_rng()

    k_linear = 10.0 ** (k_factor_db / 10.0)
    los_amp = np.sqrt(k_linear / (k_linear + 1.0))
    scatter_std = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))

    # Fixed-phase LOS + circularly symmetric scattered component.
    h_los = los_amp + 0.0j
    h_scatter = scatter_std * (
        generator.standard_normal() + 1j * generator.standard_normal()
    )
    channel_coeff = h_los + h_scatter

    faded = channel_coeff * x
    faded_power = float(np.mean(np.abs(faded) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = faded_power / snr_linear

    noise = np.sqrt(noise_variance / 2.0) * (
        generator.standard_normal(size=x.shape) + 1j * generator.standard_normal(size=x.shape)
    )
    rx_signal = faded + noise
    return rx_signal, complex(channel_coeff), float(noise_variance)
