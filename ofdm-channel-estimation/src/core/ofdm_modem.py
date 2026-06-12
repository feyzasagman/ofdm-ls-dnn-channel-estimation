"""Minimal OFDM modem utilities with QPSK support."""

from __future__ import annotations

import numpy as np


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map bits to Gray-coded QPSK symbols.

    Mapping uses bit pairs [b0, b1]:
        00 -> ( +1 +1j ) / sqrt(2)
        01 -> ( -1 +1j ) / sqrt(2)
        11 -> ( -1 -1j ) / sqrt(2)
        10 -> ( +1 -1j ) / sqrt(2)

    Args:
        bits: 1D array-like of bits (0/1) with even length.

    Returns:
        1D complex QPSK symbol array.
    """
    b = np.asarray(bits).astype(np.uint8).ravel()
    if b.size == 0:
        return np.array([], dtype=np.complex128)
    if b.size % 2 != 0:
        raise ValueError("QPSK requires an even number of bits.")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("Input bits must contain only 0 and 1.")

    pairs = b.reshape(-1, 2)
    b0 = pairs[:, 0]  # controls imag sign
    b1 = pairs[:, 1]  # controls real sign
    real = 1.0 - 2.0 * b1
    imag = 1.0 - 2.0 * b0
    return (real + 1j * imag) / np.sqrt(2.0)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision demodulation for QPSK symbols.

    Args:
        symbols: 1D complex QPSK symbol array.

    Returns:
        1D uint8 bit array (length = 2 * number of symbols).
    """
    s = np.asarray(symbols, dtype=np.complex128).ravel()
    if s.size == 0:
        return np.array([], dtype=np.uint8)

    b0 = (np.imag(s) < 0).astype(np.uint8)
    b1 = (np.real(s) < 0).astype(np.uint8)
    bits = np.empty(2 * s.size, dtype=np.uint8)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits


def place_pilots(
    data_symbols: np.ndarray,
    n_subcarriers: int,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
) -> np.ndarray:
    """Insert pilot symbols into frequency-domain OFDM vector.

    Args:
        data_symbols: 1D complex data symbols for non-pilot carriers.
        n_subcarriers: Total number of subcarriers.
        pilot_indices: Pilot index positions.
        pilot_symbols: Pilot symbols corresponding to pilot_indices.

    Returns:
        1D complex frequency-domain OFDM symbol of size n_subcarriers.
    """
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")

    pilots = np.asarray(pilot_indices, dtype=int).ravel()
    pvals = np.asarray(pilot_symbols, dtype=np.complex128).ravel()
    data = np.asarray(data_symbols, dtype=np.complex128).ravel()

    if pilots.size != pvals.size:
        raise ValueError("pilot_indices and pilot_symbols must have the same length.")
    if np.any((pilots < 0) | (pilots >= n_subcarriers)):
        raise ValueError("pilot_indices must be within [0, n_subcarriers).")

    freq = np.zeros(n_subcarriers, dtype=np.complex128)
    freq[pilots] = pvals

    data_indices = np.setdiff1d(np.arange(n_subcarriers, dtype=int), np.unique(pilots))
    if data.size != data_indices.size:
        raise ValueError("data_symbols length must match number of non-pilot carriers.")
    freq[data_indices] = data
    return freq


def ofdm_modulate(freq_symbols: np.ndarray, cp_len: int) -> np.ndarray:
    """Perform OFDM modulation (IFFT + cyclic prefix).

    Args:
        freq_symbols: 1D or 2D complex frequency-domain symbols.
            - 1D shape: (n_subcarriers,)
            - 2D shape: (n_symbols, n_subcarriers)
        cp_len: Cyclic prefix length.

    Returns:
        Time-domain OFDM symbols with CP:
            - 1D input -> 1D output of size n_subcarriers + cp_len
            - 2D input -> 2D output of shape (n_symbols, n_subcarriers + cp_len)
    """
    if cp_len < 0:
        raise ValueError("cp_len must be non-negative.")

    x = np.asarray(freq_symbols, dtype=np.complex128)
    if x.ndim == 1:
        td = np.fft.ifft(x)
        cp = td[-cp_len:] if cp_len > 0 else np.array([], dtype=np.complex128)
        return np.concatenate([cp, td])
    if x.ndim == 2:
        td = np.fft.ifft(x, axis=-1)
        if cp_len > 0:
            cp = td[:, -cp_len:]
            return np.concatenate([cp, td], axis=-1)
        return td
    raise ValueError("freq_symbols must be a 1D or 2D array.")


def ofdm_demodulate(rx_time: np.ndarray, n_subcarriers: int, cp_len: int) -> np.ndarray:
    """Perform OFDM demodulation (CP removal + FFT).

    Args:
        rx_time: 1D or 2D complex OFDM time-domain symbols with CP.
            - 1D shape: (n_subcarriers + cp_len,)
            - 2D shape: (n_symbols, n_subcarriers + cp_len)
        n_subcarriers: Number of subcarriers expected after CP removal.
        cp_len: Cyclic prefix length.

    Returns:
        Frequency-domain OFDM symbols:
            - 1D input -> 1D output shape (n_subcarriers,)
            - 2D input -> 2D output shape (n_symbols, n_subcarriers)
    """
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")
    if cp_len < 0:
        raise ValueError("cp_len must be non-negative.")

    y = np.asarray(rx_time, dtype=np.complex128)
    expected_len = n_subcarriers + cp_len

    if y.ndim == 1:
        if y.size != expected_len:
            raise ValueError("rx_time length does not match n_subcarriers + cp_len.")
        td = y[cp_len:]
        return np.fft.fft(td)
    if y.ndim == 2:
        if y.shape[1] != expected_len:
            raise ValueError("rx_time second dimension must be n_subcarriers + cp_len.")
        td = y[:, cp_len:]
        return np.fft.fft(td, axis=-1)
    raise ValueError("rx_time must be a 1D or 2D array.")
