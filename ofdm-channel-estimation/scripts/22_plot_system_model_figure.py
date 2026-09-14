"""Publication-ready SISO-OFDM system block diagram (Figure 2.1)."""

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
OUTPUT_STEM = "fig_system_model"

# Publication-oriented typography (vector text, readable when scaled in IEEE/FUJECE layout)
FONT_BLOCK = 9.5
FONT_SMALL = 8.5
FONT_SECTION = 10.5
FONT_SUB = 8.0

CAPTION = (
    "Figure 2.1. Block diagram of the SISO-OFDM transmission system and the "
    "channel estimation approaches evaluated in this study."
)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": FONT_BLOCK,
            "text.usetex": False,
            "svg.fonttype": "none",  # keep text as vectors in SVG
            "pdf.fonttype": 42,  # TrueType in PDF
        }
    )

    fig, ax = plt.subplots(figsize=(24, 9.5))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y_sig = 5.35
    bh = 0.85
    bw = 1.35

    # --- Section regions ---
    _section_box(ax, 0.15, 4.55, 7.05, 1.55, "Transmitter")
    _section_box(ax, 7.35, 3.05, 2.55, 3.55, "Channel")
    _section_box(ax, 10.15, 1.25, 13.55, 6.95, "Receiver")

    # --- Transmitter chain ---
    tx_blocks = [
        (0.35, "Binary\nData"),
        (1.85, "QPSK\nModulation"),
        (3.35, "Pilot\nInsertion"),
        (4.85, "IFFT"),
        (6.35, "Cyclic Prefix\nAddition"),
    ]
    for x, label in tx_blocks:
        _block(ax, x, y_sig - bh / 2, bw, bh, label)
    for i in range(len(tx_blocks) - 1):
        x1 = tx_blocks[i][0] + bw
        x2 = tx_blocks[i + 1][0]
        _arrow(ax, x1 + 0.05, y_sig, x2 - 0.05, y_sig)

    # --- Channel scenarios (single-tap flat fading only) ---
    ch_x = 7.55
    ch_w = 2.15
    ch_h = 0.72
    channels = [
        (6.55, "AWGN"),
        (5.35, "Single-tap Rayleigh\nflat fading"),
        (4.15, "Single-tap Rician\nflat fading"),
    ]
    for y, label in channels:
        _block(ax, ch_x, y - ch_h / 2, ch_w, ch_h, label, fontsize=FONT_SMALL)
    ax.text(ch_x + ch_w / 2, 7.35, "One scenario per realization", ha="center", fontsize=FONT_SUB, color="#444")

    # Modulated signal into channel hub
    hub_x, hub_y = 8.6, y_sig
    _arrow(ax, 6.35 + bw + 0.05, y_sig, ch_x - 0.05, hub_y)
    for y, _ in channels:
        rad = 0.0 if abs(y - y_sig) < 0.01 else 0.12
        _arrow(ax, ch_x + ch_w, y, 9.95, y_sig, connectionstyle=f"arc3,rad={rad}")

    # --- Receiver front-end ---
    rx_front = [
        (10.35, "Cyclic Prefix\nRemoval"),
        (11.85, "FFT"),
        (13.35, "Pilot\nExtraction"),
    ]
    for x, label in rx_front:
        _block(ax, x, y_sig - bh / 2, bw, bh, label)
    _arrow(ax, 9.95, y_sig, rx_front[0][0] - 0.05, y_sig)
    for i in range(len(rx_front) - 1):
        _arrow(ax, rx_front[i][0] + bw + 0.05, y_sig, rx_front[i + 1][0] - 0.05, y_sig)

    pilot_out_x = rx_front[-1][0] + bw

    # --- Four parallel estimation branches ---
    branch_x = 15.05
    merge_x = 19.15
    branches = [
        (7.55, "LS"),
        (6.35, "Simplified-MMSE\n(Wiener shrinkage)"),
        (5.15, "Flat-channel\nLMMSE"),
        (3.15, "LS+DNN"),
    ]
    for y, label in branches:
        _block(ax, branch_x, y - 0.55, 1.55, 1.1, label, fontsize=FONT_SMALL)
        _arrow(ax, pilot_out_x + 0.05, y_sig, branch_x - 0.05, y, style="solid", connectionstyle="arc3,rad=0.12")

    # LS+DNN internal residual flow
    dnn_y = 2.05
    dnn_blocks = [
        (15.0, "LS\nEstimate"),
        (16.35, "Residual\nDNN"),
        (17.7, "Residual\nCorrection"),
        (19.05, "Add to\nLS Estimate"),
    ]
    sub_h, sub_w = 0.62, 1.15
    for x, label in dnn_blocks:
        _block(ax, x, dnn_y - sub_h / 2, sub_w, sub_h, label, fontsize=FONT_SUB, linewidth=1.1)
    for i in range(len(dnn_blocks) - 1):
        _arrow(ax, dnn_blocks[i][0] + sub_w + 0.03, dnn_y, dnn_blocks[i + 1][0] - 0.03, dnn_y, lw=1.1)
    # Connect LS+DNN branch down to sub-flow
    _arrow(ax, branch_x + 0.78, 3.15 - 0.55, 15.0 + sub_w / 2, dnn_y + sub_h / 2 + 0.05, style="solid")
    ax.text(17.0, 1.35, "LS+DNN: residual learning path", ha="center", fontsize=FONT_SUB, color="#444")

    # Branch outputs to merge bus, then to main signal path at y_sig
    bus_x = merge_x - 0.12
    for y, _ in branches[:-1]:
        _arrow(ax, branch_x + 1.55 + 0.03, y, bus_x, y, lw=1.2)
    _arrow(ax, dnn_blocks[-1][0] + sub_w + 0.03, dnn_y, bus_x, dnn_y, lw=1.2)
    ax.plot([bus_x, bus_x], [dnn_y, 7.55], color="#111", linewidth=1.2, zorder=0)
    ax.plot([bus_x, merge_x], [y_sig, y_sig], color="#111", linewidth=1.2, zorder=0)

    # --- Post-estimation chain ---
    post = [
        (19.45, "Channel\nEstimate"),
        (20.85, "Equalization"),
        (22.25, "QPSK\nDemodulation"),
        (23.55, "Recovered\nBits"),
    ]
    for x, label in post:
        _block(ax, x, y_sig - bh / 2, bw, bh, label)
    _arrow(ax, merge_x, y_sig, post[0][0] - 0.05, y_sig)
    for i in range(len(post) - 1):
        _arrow(ax, post[i][0] + bw + 0.05, y_sig, post[i + 1][0] - 0.05, y_sig)

    # Annotation: flat fading note near channel
    ax.text(
        8.6,
        3.55,
        "Single-tap flat fading:\nno multipath / no frequency selectivity",
        ha="center",
        va="top",
        fontsize=FONT_SUB,
        color="#555",
        linespacing=1.25,
    )

    _verify_checklist(
        tx_blocks=tx_blocks,
        channels=channels,
        rx_front=rx_front,
        branches=branches,
        dnn_blocks=dnn_blocks,
        post=post,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = OUTPUT_DIR / f"{OUTPUT_STEM}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.12, "facecolor": "white", "edgecolor": "none"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
        print(f"Saved: {out}")

    metadata = {
        "figure_id": "Figure 2.1",
        "output_stem": OUTPUT_STEM,
        "caption": CAPTION,
        "language": "English",
        "title_on_figure": False,
        "consistency_checklist": {
            "qpsk_modulation": True,
            "pilot_insertion": True,
            "ifft_cp_addition": True,
            "cp_removal_fft": True,
            "awgn_flat": True,
            "rayleigh_single_tap_flat": True,
            "rician_single_tap_flat": True,
            "ls_estimator": True,
            "simplified_mmse_wiener_shrinkage": True,
            "lmmse_flat": True,
            "ls_dnn_residual": True,
            "equalization": True,
            "qpsk_demodulation": True,
            "recovered_bits_for_ber": True,
            "frequency_selective_multipath": False,
        },
        "estimator_labels_match_revision_pipeline": [
            "LS",
            "Simplified-MMSE",
            "LMMSE-flat",
            "LS+DNN",
        ],
    }
    meta_path = OUTPUT_DIR / f"{OUTPUT_STEM}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    caption_path = OUTPUT_DIR / f"{OUTPUT_STEM}_caption.txt"
    caption_path.write_text(CAPTION + "\n", encoding="utf-8")
    print(f"Saved metadata: {meta_path}")
    plt.close(fig)


def _verify_checklist(
    tx_blocks: list,
    channels: list,
    rx_front: list,
    branches: list,
    dnn_blocks: list,
    post: list,
) -> None:
    """Print pre-save consistency check against revision pipeline."""
    all_text = " ".join(
        label for _, label in tx_blocks + channels + rx_front + branches + dnn_blocks + post
    ).lower()
    required = [
        ("QPSK modulation", "qpsk"),
        ("Pilot insertion", "pilot"),
        ("IFFT", "ifft"),
        ("Cyclic prefix addition", "cyclic prefix\naddition"),
        ("Cyclic prefix removal", "cyclic prefix\nremoval"),
        ("AWGN", "awgn"),
        ("Single-tap Rayleigh flat fading", "rayleigh"),
        ("Single-tap Rician flat fading", "rician"),
        ("FFT", "fft"),
        ("Pilot extraction", "pilot\nextraction"),
        ("LS", "ls"),
        ("Simplified-MMSE", "simplified-mmse"),
        ("Flat-channel LMMSE", "flat-channel"),
        ("LS+DNN residual correction", "residual"),
        ("Equalization", "equalization"),
        ("QPSK demodulation", "qpsk\ndemodulation"),
        ("Recovered bits", "recovered"),
    ]
    print("Figure 2.1 consistency check:")
    for name, token in required:
        ok = token.replace("\n", " ") in all_text.replace("\n", " ")
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {name}")
        if not ok:
            raise RuntimeError(f"Diagram missing required element: {name}")


def _section_box(ax, x, y, w, h, label: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.0,
        edgecolor="#888",
        facecolor="white",
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.12,
        label,
        ha="center",
        va="top",
        fontsize=FONT_SECTION,
        fontweight="bold",
        color="#333",
    )


def _block(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fontsize: float = FONT_BLOCK,
    linewidth: float = 1.3,
    facecolor: str = "white",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=linewidth,
        edgecolor="#111",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    *,
    style: str = "solid",
    lw: float = 1.25,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    linestyle = "-" if style == "solid" else "--"
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        linestyle=linestyle,
        color="#111",
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


if __name__ == "__main__":
    main()
