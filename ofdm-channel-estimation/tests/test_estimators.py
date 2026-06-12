"""Integration tests for OFDM channel estimators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.channels.awgn import apply_awgn
from src.channels.rayleigh import apply_rayleigh_flat_fading
from src.core.metrics import mse, nmse
from src.core.ofdm_modem import (
    ofdm_demodulate,
    ofdm_modulate,
    place_pilots,
    qpsk_modulate,
)
from src.core.ofdm_params import OFDMParams
from src.core.pilots import generate_pilot_indices, generate_pilot_symbols, get_data_indices
from src.estimators.ls_estimator import estimate_ls_channel
from src.estimators.mmse_estimator import estimate_mmse_channel


def _build_tx_ofdm_symbol(params: OFDMParams, rng: np.random.Generator) -> np.ndarray:
    """Create one frequency-domain OFDM symbol with pilots and QPSK data."""
    pilot_indices = generate_pilot_indices(params.n_subcarriers, params.pilot_spacing)
    pilot_symbols = generate_pilot_symbols(pilot_indices, params.pilot_value)
    data_indices = get_data_indices(params.n_subcarriers, pilot_indices)

    bits = rng.integers(0, 2, size=2 * data_indices.size, dtype=np.uint8)
    data_symbols = qpsk_modulate(bits)
    freq_symbol = place_pilots(
        data_symbols=data_symbols,
        n_subcarriers=params.n_subcarriers,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
    )
    return freq_symbol


def test_estimators_integration_awgn() -> None:
    """End-to-end AWGN integration test for OFDM + LS/MMSE estimators."""
    params = OFDMParams(
        n_subcarriers=64,
        cp_len=16,
        pilot_spacing=4,
        modulation_order=4,
        pilot_value=1.0 + 0.0j,
        random_seed=123,
    )
    rng = np.random.default_rng(params.random_seed)
    snr_db = 10.0

    tx_freq = _build_tx_ofdm_symbol(params, rng)
    tx_time = ofdm_modulate(tx_freq, cp_len=params.cp_len)

    rx_time, noise_variance = apply_awgn(tx_time, snr_db=snr_db, rng=rng)
    rx_freq = ofdm_demodulate(rx_time, n_subcarriers=params.n_subcarriers, cp_len=params.cp_len)

    pilot_indices = generate_pilot_indices(params.n_subcarriers, params.pilot_spacing)
    pilot_symbols = generate_pilot_symbols(pilot_indices, params.pilot_value)

    h_ls_full, h_ls_pilots = estimate_ls_channel(
        rx_freq=rx_freq,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=params.n_subcarriers,
    )
    h_mmse_full, h_mmse_pilots = estimate_mmse_channel(
        rx_freq=rx_freq,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=params.n_subcarriers,
        noise_variance=noise_variance,
    )

    h_true = np.ones(params.n_subcarriers, dtype=np.complex128)
    ls_mse = mse(h_true, h_ls_full)
    mmse_mse = mse(h_true, h_mmse_full)
    ls_nmse = nmse(h_true, h_ls_full)
    mmse_nmse = nmse(h_true, h_mmse_full)

    assert h_ls_full.shape == (params.n_subcarriers,)
    assert h_mmse_full.shape == (params.n_subcarriers,)
    assert h_ls_pilots.shape == pilot_indices.shape
    assert h_mmse_pilots.shape == pilot_indices.shape
    assert np.iscomplexobj(h_ls_full)
    assert np.iscomplexobj(h_mmse_full)
    assert np.isfinite(ls_mse) and ls_mse >= 0.0
    assert np.isfinite(mmse_mse) and mmse_mse >= 0.0
    assert np.isfinite(ls_nmse) and ls_nmse >= 0.0
    assert np.isfinite(mmse_nmse) and mmse_nmse >= 0.0

    print(
        f"[AWGN] SNR={snr_db:.1f} dB | "
        f"LS MSE={ls_mse:.6f} | MMSE MSE={mmse_mse:.6f}"
    )


def test_estimators_integration_rayleigh() -> None:
    """End-to-end flat Rayleigh integration test for OFDM + LS/MMSE estimators."""
    params = OFDMParams(
        n_subcarriers=64,
        cp_len=16,
        pilot_spacing=4,
        modulation_order=4,
        pilot_value=1.0 + 0.0j,
        random_seed=456,
    )
    rng = np.random.default_rng(params.random_seed)
    snr_db = 10.0

    tx_freq = _build_tx_ofdm_symbol(params, rng)
    tx_time = ofdm_modulate(tx_freq, cp_len=params.cp_len)

    rx_time, channel_coeff, noise_variance = apply_rayleigh_flat_fading(
        tx_time, snr_db=snr_db, rng=rng
    )
    rx_freq = ofdm_demodulate(rx_time, n_subcarriers=params.n_subcarriers, cp_len=params.cp_len)

    pilot_indices = generate_pilot_indices(params.n_subcarriers, params.pilot_spacing)
    pilot_symbols = generate_pilot_symbols(pilot_indices, params.pilot_value)

    h_ls_full, _ = estimate_ls_channel(
        rx_freq=rx_freq,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=params.n_subcarriers,
    )
    h_mmse_full, _ = estimate_mmse_channel(
        rx_freq=rx_freq,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=params.n_subcarriers,
        noise_variance=noise_variance,
    )

    h_true = np.full(params.n_subcarriers, channel_coeff, dtype=np.complex128)
    ls_mse = mse(h_true, h_ls_full)
    mmse_mse = mse(h_true, h_mmse_full)
    ls_nmse = nmse(h_true, h_ls_full)
    mmse_nmse = nmse(h_true, h_mmse_full)

    assert h_ls_full.shape == (params.n_subcarriers,)
    assert h_mmse_full.shape == (params.n_subcarriers,)
    assert np.iscomplexobj(h_ls_full)
    assert np.iscomplexobj(h_mmse_full)
    assert np.isfinite(ls_mse) and ls_mse >= 0.0
    assert np.isfinite(mmse_mse) and mmse_mse >= 0.0
    assert np.isfinite(ls_nmse) and ls_nmse >= 0.0
    assert np.isfinite(mmse_nmse) and mmse_nmse >= 0.0

    print(
        f"[Rayleigh] SNR={snr_db:.1f} dB | "
        f"LS MSE={ls_mse:.6f} | MMSE MSE={mmse_mse:.6f}"
    )
