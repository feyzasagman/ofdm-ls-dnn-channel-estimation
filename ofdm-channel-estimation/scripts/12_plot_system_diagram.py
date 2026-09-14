"""Draw a clean OFDM system flowchart (no formulas)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Seminer sunumu için büyük puntolar
FONT_TITLE = 18
FONT_BLOCK = 13
FONT_CIRCLE = 12
FONT_LABEL = 13
FONT_FOOTER = 11


def main() -> None:
    """Save clean flowchart to figures directory."""
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    outer = FancyBboxPatch(
        (0.2, 0.3),
        16.4,
        5.9,
        boxstyle="round,pad=0.06,rounding_size=0.25",
        linewidth=2.0,
        edgecolor="#222",
        facecolor="white",
    )
    ax.add_patch(outer)
    ax.text(
        8.5, 5.95,
        "OFDM Kanal Kestirimi — Akış Şeması",
        ha="center", fontsize=FONT_TITLE, fontweight="bold",
    )

    y_main = 3.2

    # Verici hattı
    _block(ax, 0.5, y_main, 1.3, 0.95, "Data\nBilgi")
    _block(ax, 2.1, y_main, 1.3, 0.95, "QPSK\nModülasyon")
    _block(ax, 3.7, y_main, 1.3, 0.95, "Pilot\nYerleştirme")
    _block(ax, 5.3, y_main, 1.4, 0.95, "OFDM\nModülasyon")

    # Kanal dalları
    _block(ax, 7.2, 4.55, 1.5, 0.78, "AWGN\nKanalı", facecolor="#f5f5f5")
    _block(ax, 7.2, y_main, 1.5, 0.78, "Rayleigh\nKanalı", facecolor="#fffbeb")
    _block(ax, 7.2, 1.85, 1.5, 0.78, "Rician\nKanalı", facecolor="#fff5f5")

    ax.add_patch(FancyBboxPatch(
        (7.05, 1.65), 1.8, 3.75,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor="#666", facecolor="none", linestyle="--",
    ))
    ax.text(7.95, 5.55, "Kanal", ha="center", fontsize=FONT_LABEL, fontweight="bold")

    # Alıcı hattı
    _block(ax, 9.3, y_main, 1.4, 0.95, "OFDM\nDemodülasyon")
    _circle(ax, 11.2, y_main + 0.47, 0.68, "Kanal\nKestirimi")

    _block(ax, 12.3, 4.35, 1.1, 0.72, "LS")
    _block(ax, 12.3, y_main + 0.1, 1.1, 0.72, "MMSE")
    _block(ax, 12.3, 1.95, 1.1, 0.72, "DNN+LS", facecolor="#ebf4ff")

    _block(ax, 12.3, 0.55, 1.1, 0.78, "DNN", facecolor="#ebf4ff")

    _block(ax, 13.7, y_main, 1.2, 0.95, "Eşitleme")
    _block(ax, 15.1, y_main, 1.2, 0.95, "QPSK\nDemodülasyon")

    _circle(ax, 15.75, 1.55, 0.55, "Metrikler")

    # Oklar — verici
    for x1, x2 in [(1.8, 2.1), (3.4, 3.7), (5.0, 5.3), (6.7, 7.2)]:
        _arrow(ax, x1, y_main + 0.47, x2, y_main + 0.47)

    # mod → kanallar
    _arrow(ax, 6.7, y_main + 0.57, 7.2, 4.9)
    _arrow(ax, 6.7, y_main + 0.47, 7.2, y_main + 0.47)
    _arrow(ax, 6.7, y_main + 0.37, 7.2, 2.2)

    # kanallar → demod
    _arrow(ax, 8.7, 4.9, 9.3, y_main + 0.67)
    _arrow(ax, 8.7, y_main + 0.47, 9.3, y_main + 0.47)
    _arrow(ax, 8.7, 2.2, 9.3, y_main + 0.27)

    # demod → kestirim
    _arrow(ax, 10.7, y_main + 0.47, 10.6, y_main + 0.47)

    # kestirim → yöntemler
    _arrow(ax, 11.8, y_main + 0.77, 12.3, 4.65)
    _arrow(ax, 11.8, y_main + 0.47, 12.3, y_main + 0.47)
    _arrow(ax, 11.8, y_main + 0.17, 12.3, 2.28)

    # DNN ↔ DNN+LS
    _arrow(ax, 12.85, 1.95, 12.85, 1.33, style="dashed")
    _arrow(ax, 12.85, 1.33, 12.85, 1.95, style="dashed")

    # yöntemler → eşitleme
    _arrow(ax, 13.4, 4.65, 13.55, 4.65)
    _arrow(ax, 13.55, 4.65, 13.55, y_main + 0.77)
    _arrow(ax, 13.55, y_main + 0.77, 13.7, y_main + 0.67)

    _arrow(ax, 13.4, y_main + 0.47, 13.7, y_main + 0.47)
    _arrow(ax, 13.4, 2.28, 13.55, 2.28)
    _arrow(ax, 13.55, 2.28, 13.55, y_main + 0.27)
    _arrow(ax, 13.55, y_main + 0.27, 13.7, y_main + 0.27)

    # eşitleme → demod → metrikler
    _arrow(ax, 14.9, y_main + 0.47, 15.1, y_main + 0.47)
    _arrow(ax, 15.75, y_main, 15.75, 2.1)

    ax.text(8.5, 0.55, "MSE  ·  NMSE  ·  BER", ha="center", fontsize=FONT_FOOTER, color="#666")

    figures_dir = PROJECT_ROOT / "data" / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "ofdm_system_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()
    print(f"Saved figure: {out_path}")


def _block(ax, x, y, w, h, text, facecolor="white") -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.6,
        edgecolor="#222",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FONT_BLOCK)


def _circle(ax, cx, cy, r, text) -> None:
    patch = Circle((cx, cy), r, linewidth=1.6, edgecolor="#222", facecolor="white")
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FONT_CIRCLE)


def _arrow(ax, x1, y1, x2, y2, style: str = "solid") -> None:
    linestyle = "-" if style == "solid" else "--"
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, linestyle=linestyle, color="#222",
    )
    ax.add_patch(arrow)


if __name__ == "__main__":
    main()
