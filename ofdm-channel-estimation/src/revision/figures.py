"""Revision experiment figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.revision.ber_figure import plot_ber_curves_on_axes, style_ber_axes


def plot_mse_nmse_log(
    aggregated: dict[str, Any],
    channel_type: str,
    output_dir: Path,
    metric: str = "mse",
) -> Path:
    """Plot mean metric vs SNR on log y-axis with std bands."""
    rows = aggregated["per_snr"]
    snr = np.asarray([r["snr_db"] for r in rows], dtype=np.float64)
    methods = {
        "ls": f"ls_{metric}_mean",
        "simplified_mmse": f"simplified_mmse_{metric}_mean",
        "lmmse_flat": f"lmmse_flat_{metric}_mean",
        "ls_dnn": f"ls_dnn_{metric}_mean",
    }
    std_keys = {k: v.replace("_mean", "_std") for k, v in methods.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mean_key in methods.items():
        if mean_key not in rows[0]:
            continue
        y = np.asarray([r[mean_key] for r in rows], dtype=np.float64)
        y_std = np.asarray([r.get(std_keys[label], 0.0) for r in rows], dtype=np.float64)
        y = np.maximum(y, 1e-12)
        ax.plot(snr, y, marker="o", label=label)
        ax.fill_between(snr, np.maximum(y - y_std, 1e-12), y + y_std, alpha=0.2)

    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs SNR ({channel_type}, multi-seed mean±std)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = output_dir / f"snr_vs_{metric}_{channel_type}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


_BER_METHOD_ORDER = ("ls", "simplified_mmse", "lmmse_flat", "ls_dnn")
_BER_METHOD_LABELS = {
    "ls": "LS",
    "simplified_mmse": "Simplified-MMSE",
    "lmmse_flat": "LMMSE-flat",
    "ls_dnn": "LS+DNN",
}
_BER_METHOD_STYLE = {
    "ls": {"color": "#1f77b4", "marker": "o"},
    "simplified_mmse": {"color": "#ff7f0e", "marker": "s"},
    "lmmse_flat": {"color": "#2ca02c", "marker": "^"},
    "ls_dnn": {"color": "#d62728", "marker": "D"},
}


def plot_ber_log(
    aggregated: dict[str, Any],
    channel_type: str,
    output_dir: Path,
) -> Path:
    rows = aggregated["per_snr"]
    snr = np.asarray([r["snr_db"] for r in rows], dtype=np.float64)
    style_map = {
        k: {**_BER_METHOD_STYLE[k], "label": _BER_METHOD_LABELS[k]} for k in _BER_METHOD_ORDER
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ber_curves_on_axes(ax, snr, rows, _BER_METHOD_ORDER, style_map)
    style_ber_axes(ax)
    ax.set_title(f"BER vs SNR ({channel_type}, multi-seed mean±std)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out = output_dir / f"snr_vs_ber_{channel_type}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_history(history: dict[str, Any], channel_type: str, seed: int, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.get("loss", []), label="train MSE")
    ax.plot(history.get("val_loss", []), label="val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training history ({channel_type}, seed={seed})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = output_dir / f"training_history_{channel_type}_seed{seed}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sample_channel(
    true_h: np.ndarray,
    ls_h: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot flat-fading channel magnitude (constant across subcarriers)."""
    n = true_h.size
    k = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k, np.abs(true_h), label="|H_true|", linewidth=2)
    ax.plot(k, np.abs(ls_h), "--", label="|H_LS|", alpha=0.8)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("|H|")
    ax.set_title(
        f"Sample channel ({metadata.get('channel_model_name', 'flat fading')}) "
        f"SNR={metadata.get('snr_db')} dB seed={metadata.get('seed')}"
    )
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    tag = f"{metadata.get('channel_type')}_snr{metadata.get('snr_db')}_seed{metadata.get('seed')}"
    out = output_dir / f"sample_channel_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
