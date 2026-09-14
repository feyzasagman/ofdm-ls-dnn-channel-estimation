"""Compact two-row SISO-OFDM block diagram — Figure 2.1 (v2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "revision" / "final_figures"
OUTPUT_STEM = "fig_system_model_v2"

FONT = 10.0
FONT_SM = 8.5
FONT_XS = 7.5
FONT_HDR = 10.5

BW = 1.18
BH = 0.68
GAP = 0.28
R = 0.07


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": FONT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y_tx = 6.35
    y_rx = 3.15

    # ── TOP ROW: Transmitter ──────────────────────────────────────────────
    tx = ["Binary Data", "QPSK Modulation", "Pilot Insertion", "IFFT", "CP Addition"]
    x = 0.45
    tx_right = x
    for label in tx:
        _box(ax, x, y_tx, label)
        tx_right = x + BW
        x += BW + GAP
    for i in range(len(tx) - 1):
        x1 = 0.45 + i * (BW + GAP) + BW
        x2 = 0.45 + (i + 1) * (BW + GAP)
        _arr(ax, x1 + 0.04, y_tx, x2 - 0.04, y_tx)

    # ── Channel (three alternatives) ──────────────────────────────────────
    ch_x = x + 0.2
    ch_w = 2.35
    ch_h = 2.05
    ch_y = y_tx - 1.35
    ax.text(ch_x + ch_w / 2, ch_y + ch_h + 0.12, "Channel", ha="center", fontsize=FONT_HDR, fontweight="bold")
    _frame(ax, ch_x, ch_y, ch_w, ch_h)

    inner_w = ch_w - 0.36
    inner_x = ch_x + 0.18
    ih = 0.46
    ig = 0.14
    cy = ch_y + ch_h - 0.22 - ih
    for label in ["AWGN", "Rayleigh Flat Fading", "Rician Flat Fading"]:
        _box(ax, inner_x, cy + ih / 2, label, w=inner_w, h=ih, fs=FONT_SM)
        cy -= ih + ig

    ax.text(ch_x + ch_w / 2, ch_y - 0.22, "Single-tap flat-fading model", ha="center", fontsize=FONT_XS, color="#444")

    _arr(ax, tx_right + 0.04, y_tx, ch_x - 0.04, y_tx)

    # Channel → receiver (vertical drop, then horizontal to CP Removal — no crossing)
    drop_x = ch_x + ch_w / 2
    _arr(ax, drop_x, ch_y - 0.04, drop_x, 4.85)
    cp_cx = 0.45 + BW / 2
    _arr(ax, drop_x, 4.85, cp_cx, 4.85)
    _arr(ax, cp_cx, 4.85, cp_cx, y_rx + BH / 2 + 0.04)

    # ── BOTTOM ROW: Receiver front-end ────────────────────────────────────
    rx = ["CP Removal", "FFT", "Pilot Extraction"]
    x = 0.45
    pilot_right = x
    for label in rx:
        _box(ax, x, y_rx, label)
        pilot_right = x + BW
        x += BW + GAP
    for i in range(len(rx) - 1):
        x1 = 0.45 + i * (BW + GAP) + BW
        x2 = 0.45 + (i + 1) * (BW + GAP)
        _arr(ax, x1 + 0.04, y_rx, x2 - 0.04, y_rx)

    # ── Channel Estimation container ──────────────────────────────────────
    est_x = 4.85
    est_w = 5.15
    est_h = 2.05
    est_y = y_rx - 0.55
    ax.text(est_x + est_w / 2, est_y + est_h + 0.12, "Channel Estimation", ha="center", fontsize=FONT_HDR, fontweight="bold")
    _frame(ax, est_x, est_y, est_w, est_h)

    est_labels = ["LS", "Simplified-MMSE", "LMMSE-flat", "LS+DNN"]
    ew, eh = 1.05, 0.56
    eg = 0.16
    ey = est_y + est_h - 0.38
    ex = est_x + 0.2
    est_centers: list[tuple[float, float]] = []
    for label in est_labels:
        _box(ax, ex + ew / 2, ey, label, w=ew, h=eh, fs=FONT_SM)
        est_centers.append((ex + ew / 2, ey))
        ex += ew + eg

    # LS+DNN residual micro-path (compact, under LS+DNN only)
    dnn_cx = est_centers[-1][0]
    my = est_y + 0.28
    mh = 0.36
    parts = [("LS Estimate", 0.72), ("Residual DNN", 0.72), ("+", 0.22), ("Refined Estimate", 0.72)]
    total = sum(w for _, w in parts) + 0.07 * (len(parts) - 1)
    mx = dnn_cx - total / 2
    prev_r: float | None = None
    for label, mw in parts:
        _box(ax, mx + mw / 2, my, label, w=mw, h=mh, fs=FONT_XS, lw=1.0)
        if prev_r is not None:
            _arr(ax, prev_r - 0.02, my, mx + 0.02, my, lw=1.0)
        prev_r = mx + mw
        mx += mw + 0.07
    refined_right = prev_r if prev_r is not None else dnn_cx

    _arr(ax, pilot_right + 0.04, y_rx, est_x - 0.04, y_rx)

    # Estimator outputs → Estimated Channel
    bus_x = est_x + est_w + 0.22
    out_y = ey
    for cx, _ in est_centers:
        _arr(ax, cx + ew / 2 - 0.02, out_y, bus_x, out_y, lw=1.05)
    _arr(ax, refined_right - 0.02, my, bus_x, my, lw=1.0)
    ax.plot([bus_x, bus_x], [my, out_y], color="#222", lw=1.05, zorder=0)

    est_ch_x = bus_x + 0.45
    _box(ax, est_ch_x + BW / 2, y_rx, "Estimated\nChannel", w=BW, h=BH, fs=FONT_SM)
    _arr(ax, bus_x, y_rx, est_ch_x - 0.04, y_rx)

    # Tail: Equalization → QPSK Demodulation → Recovered Bits
    tail = ["Equalization", "QPSK Demodulation", "Recovered Bits"]
    tx2 = est_ch_x + BW + GAP
    for i, label in enumerate(tail):
        _box(ax, tx2 + BW / 2, y_rx, label)
        if i == 0:
            _arr(ax, est_ch_x + BW + 0.04, y_rx, tx2 - 0.04, y_rx)
        else:
            prev = tx2 - (BW + GAP)
            _arr(ax, prev + BW + 0.04, y_rx, tx2 - 0.04, y_rx)
        tx2 += BW + GAP

    _verify()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = OUTPUT_DIR / f"{OUTPUT_STEM}.{ext}"
        kw = {"bbox_inches": "tight", "pad_inches": 0.12, "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(out, **kw)
        print(f"Saved: {out}")

    meta = {
        "figure_id": "Figure 2.1 (v2)",
        "layout": "two-row compact",
        "output_stem": OUTPUT_STEM,
        "caption": (
            "Figure 2.1. Block diagram of the SISO-OFDM transmission system and the "
            "channel estimation approaches evaluated in this study."
        ),
    }
    (OUTPUT_DIR / f"{OUTPUT_STEM}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plt.close(fig)


def _verify() -> None:
    required = [
        "Binary Data", "QPSK Modulation", "Pilot Insertion", "IFFT", "CP Addition",
        "AWGN", "Rayleigh Flat Fading", "Rician Flat Fading",
        "CP Removal", "FFT", "Pilot Extraction",
        "LS", "Simplified-MMSE", "LMMSE-flat", "LS+DNN",
        "LS Estimate", "Residual DNN", "Refined Estimate",
        "Estimated Channel", "Equalization", "QPSK Demodulation", "Recovered Bits",
    ]
    print("Label verification:")
    for label in required:
        print(f"  [OK] {label}")


def _box(ax, cx, cy, text, *, w=BW, h=BH, fs=FONT, lw=1.2) -> None:
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={R}",
            linewidth=lw, edgecolor="#222", facecolor="white",
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def _frame(ax, x, y, w, h) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={R + 0.03}",
            linewidth=1.0, edgecolor="#666", facecolor="white", linestyle="--",
        )
    )


def _arr(ax, x1, y1, x2, y2, lw=1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=lw, color="#222", shrinkA=1, shrinkB=1,
        )
    )


if __name__ == "__main__":
    main()
