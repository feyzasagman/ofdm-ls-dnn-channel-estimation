"""Pilot placement utilities for OFDM symbols."""

from __future__ import annotations

import numpy as np


def generate_pilot_indices(n_subcarriers: int, pilot_spacing: int) -> np.ndarray:
    """Return pilot subcarrier indices with fixed spacing.

    Args:
        n_subcarriers: Total number of subcarriers.
        pilot_spacing: Step between neighboring pilots.

    Returns:
        1D integer array of pilot indices.
    """
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")
    if pilot_spacing <= 0:
        raise ValueError("pilot_spacing must be positive.")

    return np.arange(0, n_subcarriers, pilot_spacing, dtype=int)


def generate_pilot_symbols(pilot_indices: np.ndarray, pilot_value: complex = 1.0 + 0.0j) -> np.ndarray:
    """Create pilot symbols for the given pilot locations.

    Args:
        pilot_indices: Pilot index array.
        pilot_value: Complex pilot value to place on all pilot tones.

    Returns:
        Complex-valued pilot symbol array.
    """
    indices = np.asarray(pilot_indices, dtype=int).ravel()
    if indices.size == 0:
        return np.array([], dtype=np.complex128)
    return np.full(indices.size, pilot_value, dtype=np.complex128)


def get_data_indices(n_subcarriers: int, pilot_indices: np.ndarray) -> np.ndarray:
    """Return non-pilot subcarrier indices.

    Args:
        n_subcarriers: Total number of subcarriers.
        pilot_indices: Pilot index array.

    Returns:
        1D integer array containing data-carrying indices.
    """
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive.")

    all_indices = np.arange(n_subcarriers, dtype=int)
    pilots = np.unique(np.asarray(pilot_indices, dtype=int).ravel())
    if np.any((pilots < 0) | (pilots >= n_subcarriers)):
        raise ValueError("pilot_indices must be within [0, n_subcarriers).")

    return np.setdiff1d(all_indices, pilots, assume_unique=True)
