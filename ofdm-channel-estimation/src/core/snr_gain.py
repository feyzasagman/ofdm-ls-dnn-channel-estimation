"""SNR gain estimation at target BER levels."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_snr_gain_at_target_ber(
    snr_db: np.ndarray,
    ber_ls: np.ndarray,
    ber_dnn: np.ndarray,
    target_bers: tuple[float, ...] = (1e-1, 1e-2, 1e-3),
) -> dict[str, Any]:
    """Compute SNR gain = SNR_LS - SNR_DNN at target BER via interpolation.

    Returns 'not estimable within simulated SNR range' when the target BER is
    not bracketed by the simulated BER curves.
    """
    snr = np.asarray(snr_db, dtype=np.float64)
    ls = np.asarray(ber_ls, dtype=np.float64)
    dnn = np.asarray(ber_dnn, dtype=np.float64)

    if snr.shape != ls.shape or snr.shape != dnn.shape:
        raise ValueError("snr_db, ber_ls, and ber_dnn must have the same shape.")

    order = np.argsort(snr)
    snr = snr[order]
    ls = ls[order]
    dnn = dnn[order]

    results: dict[str, Any] = {}
    for target in target_bers:
        key = f"target_ber_{target:g}"
        snr_ls = _snr_at_target_ber(snr, ls, target)
        snr_dnn = _snr_at_target_ber(snr, dnn, target)
        if snr_ls is None or snr_dnn is None:
            results[key] = {
                "target_ber": float(target),
                "status": "not estimable within simulated SNR range",
                "snr_ls_db": None,
                "snr_dnn_db": None,
                "snr_gain_db": None,
            }
        else:
            results[key] = {
                "target_ber": float(target),
                "status": "estimable",
                "snr_ls_db": float(snr_ls),
                "snr_dnn_db": float(snr_dnn),
                "snr_gain_db": float(snr_ls - snr_dnn),
            }
    return results


def _snr_at_target_ber(snr_db: np.ndarray, ber: np.ndarray, target: float) -> float | None:
    """Interpolate SNR at which BER curve crosses target (log domain for BER)."""
    if np.all(ber <= 0):
        return None

    # Use log10 for interpolation; replace exact zeros with NaN for bracket search.
    log_ber = np.full_like(ber, np.nan, dtype=np.float64)
    positive = ber > 0
    if not np.any(positive):
        return None
    log_ber[positive] = np.log10(ber[positive])
    log_target = np.log10(target)

    # Find crossing on log scale using piecewise linear interpolation in SNR.
    for i in range(len(snr_db) - 1):
        y0, y1 = log_ber[i], log_ber[i + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if (y0 - log_target) * (y1 - log_target) <= 0 and y0 != y1:
            frac = (log_target - y0) / (y1 - y0)
            return float(snr_db[i] + frac * (snr_db[i + 1] - snr_db[i]))

    # If target is above entire curve, extrapolation is not allowed.
    return None
