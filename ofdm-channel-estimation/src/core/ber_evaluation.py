"""End-to-end BER evaluation for OFDM channel estimation methods."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from src.core.ofdm_modem import qpsk_demodulate
from src.core.pilots import get_data_indices
from src.estimators.lmmse_flat_estimator import estimate_lmmse_flat_channel
from src.estimators.ls_estimator import estimate_ls_channel
from src.estimators.mmse_estimator import estimate_simplified_mmse_channel

EstimatorName = Literal["ls", "simplified_mmse", "lmmse_flat", "ls_dnn"]


def evaluate_ber_on_test_split(
    raw_npz_path: str,
    test_indices: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    n_subcarriers: int,
    *,
    dnn_pred: np.ndarray | None = None,
    channel_prior_variance: float = 1.0,
) -> dict[str, Any]:
    """Evaluate BER using identical received signals and transmitted bits.

    Pipeline per sample:
        rx_freq -> channel estimate -> ZF equalization on data carriers
        -> QPSK demodulation -> BER vs stored transmitted bits.
    """
    data = np.load(raw_npz_path, allow_pickle=False)
    rx_freq_all = np.asarray(data["rx_freq"], dtype=np.complex128)
    true_channel_all = np.asarray(data["true_channel"], dtype=np.complex128)
    snr_all = np.asarray(data["snr_db"], dtype=np.float64)
    noise_var_all = np.asarray(data["noise_variance"], dtype=np.float64)
    bits_all = np.asarray(data["bits"], dtype=np.uint8)

    test_idx = np.asarray(test_indices, dtype=int)
    data_indices = get_data_indices(n_subcarriers, pilot_indices)

    methods = ["ls", "simplified_mmse", "lmmse_flat"]
    if dnn_pred is not None:
        methods.append("ls_dnn")
        if dnn_pred.shape[0] != test_idx.size:
            raise ValueError("dnn_pred must align with test_indices length.")

    per_snr: dict[str, dict[str, dict[str, float | int]]] = {}
    for snr_val in np.sort(np.unique(snr_all[test_idx])):
        key = str(float(snr_val))
        per_snr[key] = {m: {"total_bits": 0, "erroneous_bits": 0, "ber": 0.0} for m in methods}

    dnn_i = 0
    for global_i in test_idx:
        rx_freq = rx_freq_all[global_i]
        tx_bits = bits_all[global_i]
        noise_var = float(noise_var_all[global_i])
        snr_val = float(snr_all[global_i])
        key = str(snr_val)

        h_estimates = {
            "ls": estimate_ls_channel(rx_freq, pilot_indices, pilot_symbols, n_subcarriers)[0],
            "simplified_mmse": estimate_simplified_mmse_channel(
                rx_freq, pilot_indices, pilot_symbols, n_subcarriers, noise_var
            )[0],
            "lmmse_flat": estimate_lmmse_flat_channel(
                rx_freq,
                pilot_indices,
                pilot_symbols,
                n_subcarriers,
                noise_var,
                channel_prior_variance=channel_prior_variance,
            )[0],
        }
        if dnn_pred is not None:
            h_estimates["ls_dnn"] = dnn_pred[dnn_i]
            dnn_i += 1

        for method, h_hat in h_estimates.items():
            recovered_bits = _recover_bits_zf(rx_freq=rx_freq, h_hat=h_hat, data_indices=data_indices)
            bit_errors = int(np.sum(recovered_bits != tx_bits))
            per_snr[key][method]["total_bits"] += int(tx_bits.size)
            per_snr[key][method]["erroneous_bits"] += bit_errors

    for key in per_snr:
        for method in methods:
            total = int(per_snr[key][method]["total_bits"])
            err = int(per_snr[key][method]["erroneous_bits"])
            per_snr[key][method]["ber"] = float(err / total) if total > 0 else 0.0

    overall = _aggregate_ber(per_snr, methods)
    return {
        "per_snr": per_snr,
        "overall": overall,
        "methods": methods,
    }


def _recover_bits_zf(
    rx_freq: np.ndarray,
    h_hat: np.ndarray,
    data_indices: np.ndarray,
) -> np.ndarray:
    y_data = rx_freq[data_indices]
    h_data = h_hat[data_indices]
    eps = 1e-12
    x_hat = y_data / (h_data + eps * (np.abs(h_data) == 0))
    return qpsk_demodulate(x_hat)


def _aggregate_ber(
    per_snr: dict[str, dict[str, dict[str, float | int]]],
    methods: list[str],
) -> dict[str, dict[str, float | int]]:
    overall: dict[str, dict[str, float | int]] = {}
    for method in methods:
        total_bits = sum(int(per_snr[k][method]["total_bits"]) for k in per_snr)
        err_bits = sum(int(per_snr[k][method]["erroneous_bits"]) for k in per_snr)
        overall[method] = {
            "total_bits": total_bits,
            "erroneous_bits": err_bits,
            "ber": float(err_bits / total_bits) if total_bits > 0 else 0.0,
        }
    return overall


# Log-scale visualization floor only (not stored/measured BER).
BER_LOG_VIS_FLOOR = 1e-6


def ber_for_log_plot(ber_value: float, floor: float = BER_LOG_VIS_FLOOR) -> float:
    """Map BER to a positive y-value for log-scale figures only.

    Stored JSON/CSV tables must keep the exact BER (including 0.0). Use
    ``ber_figure.plot_ber_curves_on_axes`` omits zero mean BER from log-scale figures.
    """
    if ber_value <= 0.0:
        return floor
    return float(ber_value)
