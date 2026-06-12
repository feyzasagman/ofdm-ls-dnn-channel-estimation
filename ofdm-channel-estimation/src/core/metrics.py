"""Evaluation metrics for channel estimation experiments."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error for real or complex arrays."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if yt.size == 0:
        raise ValueError("Inputs must be non-empty.")
    err = yt - yp
    return float(np.mean(np.abs(err) ** 2))


def nmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized MSE for real or complex arrays."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if yt.size == 0:
        raise ValueError("Inputs must be non-empty.")

    numerator = np.sum(np.abs(yt - yp) ** 2)
    denominator = np.sum(np.abs(yt) ** 2)
    return float(numerator / (denominator + eps))


def ber(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    """Compute bit error rate.

    Inputs can be either:
    - Bit arrays containing values in {0, 1}, or
    - Complex QPSK symbols, which are hard-decoded by sign.
    """
    bt = np.asarray(bits_true).ravel()
    bp = np.asarray(bits_pred).ravel()
    if bt.size == 0 or bp.size == 0:
        raise ValueError("Inputs must be non-empty.")

    bt_bits = _to_bits(bt)
    bp_bits = _to_bits(bp)
    if bt_bits.shape != bp_bits.shape:
        raise ValueError("Converted bit arrays must have the same shape.")

    return float(np.mean(bt_bits != bp_bits))


def _to_bits(arr: np.ndarray) -> np.ndarray:
    """Convert bit arrays or complex QPSK symbols to bit representation."""
    if np.iscomplexobj(arr):
        b0 = (np.imag(arr) < 0).astype(np.uint8)
        b1 = (np.real(arr) < 0).astype(np.uint8)
        out = np.empty(2 * arr.size, dtype=np.uint8)
        out[0::2] = b0
        out[1::2] = b1
        return out

    bits = arr.astype(np.uint8)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bit arrays must contain only 0 and 1.")
    return bits
