"""Raw dataset generation for OFDM channel estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.channels.awgn import apply_awgn
from src.channels.rayleigh import apply_rayleigh_flat_fading
from src.channels.rician import apply_rician_flat_fading
from src.core.ofdm_modem import ofdm_demodulate, ofdm_modulate, place_pilots, qpsk_modulate
from src.core.ofdm_params import OFDMParams
from src.core.pilots import generate_pilot_indices, generate_pilot_symbols, get_data_indices
from src.core.channel_models import channel_metadata
from src.estimators.lmmse_flat_estimator import estimate_lmmse_flat_channel
from src.estimators.ls_estimator import estimate_ls_channel
from src.estimators.mmse_estimator import estimate_simplified_mmse_channel

DEFAULT_SNR_LIST: tuple[float, ...] = (-10, -5, 0, 5, 10, 15, 20, 25, 30)


def generate_raw_dataset(
    channel_type: str,
    snr_db_list: Sequence[float] = DEFAULT_SNR_LIST,
    n_samples_per_snr: int = 100,
    ofdm_params: OFDMParams | None = None,
    random_seed: int = 42,
    k_factor_db: float = 6.0,
    output_path: str | Path | None = None,
) -> Path:
    """Generate raw OFDM channel-estimation dataset and save as NPZ.

    For each SNR in snr_db_list, this function generates n_samples_per_snr
    examples and stores:
        - true_channel
        - ls_estimate
        - mmse_estimate
        - snr_db
        - channel_type
    """
    params = ofdm_params if ofdm_params is not None else OFDMParams()
    channel_name = channel_type.strip().lower()
    if channel_name not in {"awgn", "rayleigh", "rician"}:
        raise ValueError("channel_type must be one of: awgn, rayleigh, rician.")
    if n_samples_per_snr <= 0:
        raise ValueError("n_samples_per_snr must be positive.")
    if len(snr_db_list) == 0:
        raise ValueError("snr_db_list must be non-empty.")

    rng = np.random.default_rng(random_seed)
    pilot_indices = generate_pilot_indices(params.n_subcarriers, params.pilot_spacing)
    pilot_symbols = generate_pilot_symbols(pilot_indices, params.pilot_value)
    data_indices = get_data_indices(params.n_subcarriers, pilot_indices)

    n_total = len(snr_db_list) * n_samples_per_snr
    true_channel = np.zeros((n_total, params.n_subcarriers), dtype=np.complex128)
    ls_estimate = np.zeros((n_total, params.n_subcarriers), dtype=np.complex128)
    simplified_mmse_estimate = np.zeros((n_total, params.n_subcarriers), dtype=np.complex128)
    lmmse_flat_estimate = np.zeros((n_total, params.n_subcarriers), dtype=np.complex128)
    rx_freq_all = np.zeros((n_total, params.n_subcarriers), dtype=np.complex128)
    n_bits = 2 * data_indices.size
    bits_all = np.zeros((n_total, n_bits), dtype=np.uint8)
    snr_all = np.zeros(n_total, dtype=np.float64)
    noise_variance_all = np.zeros(n_total, dtype=np.float64)
    channel_type_all = np.full(n_total, channel_name, dtype="<U16")

    idx = 0
    for snr_db in snr_db_list:
        for _ in range(n_samples_per_snr):
            bits = rng.integers(0, 2, size=2 * data_indices.size, dtype=np.uint8)
            data_symbols = qpsk_modulate(bits)
            tx_freq = place_pilots(
                data_symbols=data_symbols,
                n_subcarriers=params.n_subcarriers,
                pilot_indices=pilot_indices,
                pilot_symbols=pilot_symbols,
            )
            tx_time = ofdm_modulate(tx_freq, cp_len=params.cp_len)

            if channel_name == "awgn":
                rx_time, noise_var = apply_awgn(tx_time, snr_db=float(snr_db), rng=rng)
                h_true_coeff = 1.0 + 0.0j
            elif channel_name == "rayleigh":
                rx_time, h_true_coeff, noise_var = apply_rayleigh_flat_fading(
                    tx_time, snr_db=float(snr_db), rng=rng
                )
            else:
                rx_time, h_true_coeff, noise_var = apply_rician_flat_fading(
                    tx_time, snr_db=float(snr_db), k_factor_db=k_factor_db, rng=rng
                )

            rx_freq = ofdm_demodulate(
                rx_time=rx_time,
                n_subcarriers=params.n_subcarriers,
                cp_len=params.cp_len,
            )
            h_ls_full, _ = estimate_ls_channel(
                rx_freq=rx_freq,
                pilot_indices=pilot_indices,
                pilot_symbols=pilot_symbols,
                n_subcarriers=params.n_subcarriers,
            )
            h_simp_full, _ = estimate_simplified_mmse_channel(
                rx_freq=rx_freq,
                pilot_indices=pilot_indices,
                pilot_symbols=pilot_symbols,
                n_subcarriers=params.n_subcarriers,
                noise_variance=noise_var,
            )
            h_lmmse_full, _ = estimate_lmmse_flat_channel(
                rx_freq=rx_freq,
                pilot_indices=pilot_indices,
                pilot_symbols=pilot_symbols,
                n_subcarriers=params.n_subcarriers,
                noise_variance=noise_var,
                channel_prior_variance=1.0,
            )

            true_channel[idx] = np.full(params.n_subcarriers, h_true_coeff, dtype=np.complex128)
            ls_estimate[idx] = h_ls_full
            simplified_mmse_estimate[idx] = h_simp_full
            lmmse_flat_estimate[idx] = h_lmmse_full
            bits_all[idx] = bits
            rx_freq_all[idx] = rx_freq
            snr_all[idx] = float(snr_db)
            noise_variance_all[idx] = float(noise_var)
            idx += 1

    save_path = _resolve_raw_output_path(
        channel_name=channel_name,
        n_samples_per_snr=n_samples_per_snr,
        random_seed=random_seed,
        output_path=output_path,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        true_channel=true_channel,
        ls_estimate=ls_estimate,
        simplified_mmse_estimate=simplified_mmse_estimate,
        lmmse_flat_estimate=lmmse_flat_estimate,
        mmse_estimate=simplified_mmse_estimate,
        bits=bits_all,
        rx_freq=rx_freq_all,
        snr_db=snr_all,
        channel_type=channel_type_all,
        noise_variance=noise_variance_all,
        pilot_indices=pilot_indices,
        pilot_symbols=pilot_symbols,
        n_subcarriers=params.n_subcarriers,
        cp_len=params.cp_len,
        pilot_spacing=params.pilot_spacing,
        modulation_order=params.modulation_order,
        random_seed=random_seed,
        k_factor_db=float(k_factor_db),
        channel_model_name=channel_metadata(channel_name, k_factor_db=k_factor_db)["channel_model_name"],
        flat_fading=True,
        frequency_selective=False,
    )
    return save_path


def _resolve_raw_output_path(
    channel_name: str,
    n_samples_per_snr: int,
    random_seed: int,
    output_path: str | Path | None,
) -> Path:
    """Resolve output NPZ path under data/raw when not explicitly provided."""
    if output_path is not None:
        return Path(output_path)
    project_root = Path(__file__).resolve().parents[2]
    filename = f"{channel_name}_ns{n_samples_per_snr}_seed{random_seed}.npz"
    return project_root / "data" / "raw" / filename
