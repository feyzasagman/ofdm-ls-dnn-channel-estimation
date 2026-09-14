"""Shared BER log-scale figure helpers (visualization only; stored BER unchanged)."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

MANUSCRIPT_ZERO_BER_CAPTION_NOTE = (
    "Zero BER values, corresponding to no observed bit errors in the finite test set, "
    "are omitted from the logarithmic plot."
)

BER_SNR_XMIN = -10.0
BER_SNR_XMAX = 30.0
BER_SNR_XMARGIN = 1.0  # dB padding so edge markers/error bars are not clipped
BER_SNR_XTICKS = (-10, -5, 0, 5, 10, 15, 20, 25, 30)


def ber_figure_note() -> str:
    """Short manuscript caption note for zero-BER handling on log-scale figures."""
    return MANUSCRIPT_ZERO_BER_CAPTION_NOTE


def _contiguous_index_segments(pos_indices: np.ndarray) -> list[np.ndarray]:
    """Split positive-SN R indices into runs consecutive in the full SNR grid."""
    if pos_indices.size == 0:
        return []
    segments: list[np.ndarray] = []
    start = 0
    for k in range(1, pos_indices.size):
        if pos_indices[k] != pos_indices[k - 1] + 1:
            segments.append(pos_indices[start:k])
            start = k
    segments.append(pos_indices[start:])
    return segments


def plot_ber_curves_on_axes(
    ax: Axes,
    snr: np.ndarray,
    rows: list[dict[str, Any]],
    method_order: tuple[str, ...],
    method_style: dict[str, dict[str, str]],
) -> None:
    """Plot mean±std BER for strictly positive means; omit zero-BER SNR points.

    Curves are not drawn across SNR gaps caused by omitted zero-BER points. Isolated
    positive points use markers with local error bars instead of line segments.
    """
    for key in method_order:
        mean_k = f"{key}_ber_mean"
        std_k = f"{key}_ber_std"
        if mean_k not in rows[0]:
            continue

        style = method_style[key]
        color = style["color"]
        marker = style["marker"]
        label = style["label"]

        y_raw = np.asarray([r[mean_k] for r in rows], dtype=np.float64)
        y_std = np.asarray([r.get(std_k, 0.0) for r in rows], dtype=np.float64)
        pos_mask = y_raw > 0.0
        if not np.any(pos_mask):
            continue

        pos_indices = np.flatnonzero(pos_mask)
        segments = _contiguous_index_segments(pos_indices)

        for seg_i, idx_seg in enumerate(segments):
            seg_label = label if seg_i == 0 else None
            snr_seg = snr[idx_seg]
            y_seg = y_raw[idx_seg]
            std_seg = y_std[idx_seg]
            y_low = np.maximum(y_seg - std_seg, y_seg * 1e-3)
            y_high = y_seg + std_seg

            if idx_seg.size == 1:
                ax.errorbar(
                    snr_seg,
                    y_seg,
                    yerr=np.vstack([y_seg - y_low, y_high - y_seg]),
                    fmt=marker,
                    color=color,
                    label=seg_label,
                    markersize=7,
                    capsize=4,
                    capthick=1.2,
                    elinewidth=1.2,
                    linestyle="none",
                    zorder=3,
                )
            else:
                ax.plot(
                    snr_seg,
                    y_seg,
                    marker=marker,
                    color=color,
                    label=seg_label,
                    linewidth=2,
                    linestyle="-",
                )
                ax.fill_between(snr_seg, y_low, y_high, color=color, alpha=0.18)


def style_ber_axes(ax: Axes) -> None:
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_xlim(BER_SNR_XMIN - BER_SNR_XMARGIN, BER_SNR_XMAX + BER_SNR_XMARGIN)
    ax.set_xticks(list(BER_SNR_XTICKS))
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
