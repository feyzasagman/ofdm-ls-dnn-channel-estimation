"""AWGN channel utilities for complex baseband signals."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_awgn(
    tx_signal: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """Add complex AWGN to a transmitted baseband signal.

    SNR definition used in this project:
        snr_linear = signal_power / noise_power
    where signal_power = mean(|tx_signal|^2) and noise_power is complex
    noise variance E[|n|^2].
    """
    x = np.asarray(tx_signal, dtype=np.complex128)
    if x.size == 0:
        raise ValueError("tx_signal must be non-empty.")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    generator = rng if rng is not None else np.random.default_rng()
    signal_power = float(np.mean(np.abs(x) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear

    noise = np.sqrt(noise_variance / 2.0) * (
        generator.standard_normal(size=x.shape) + 1j * generator.standard_normal(size=x.shape)
    )
    rx_signal = x + noise
    return rx_signal, float(noise_variance)
