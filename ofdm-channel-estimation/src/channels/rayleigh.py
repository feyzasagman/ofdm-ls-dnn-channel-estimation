"""Flat Rayleigh fading channel model."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_rayleigh_flat_fading(
    tx_signal: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, complex, float]:
    """Apply single-tap Rayleigh fading and AWGN.

    A single complex coefficient h is used for the whole input vector:
        y = h * x + n
    where h ~ CN(0, 1). Noise variance is set from faded signal power so the
    receive-side SNR matches snr_db.
    """
    x = np.asarray(tx_signal, dtype=np.complex128)
    if x.size == 0:
        raise ValueError("tx_signal must be non-empty.")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    generator = rng if rng is not None else np.random.default_rng()

    channel_coeff = (
        generator.standard_normal() + 1j * generator.standard_normal()
    ) / np.sqrt(2.0)
    faded = channel_coeff * x

    faded_power = float(np.mean(np.abs(faded) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = faded_power / snr_linear

    noise = np.sqrt(noise_variance / 2.0) * (
        generator.standard_normal(size=x.shape) + 1j * generator.standard_normal(size=x.shape)
    )
    rx_signal = faded + noise
    return rx_signal, complex(channel_coeff), float(noise_variance)
