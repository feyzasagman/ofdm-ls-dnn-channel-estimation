"""Draw a clean DNN architecture flowchart."""

from __future__ import annotations

import argparse
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
FONT_DNN_LABEL = 13
FONT_SKIP = 10
FONT_TRAIN_TITLE = 13
FONT_TRAIN_BODY = 11

LABELS = {
    "tr": {
        "title": "LS + DNN Kanal Kestirimi — Akış Şeması",
        "ls_input": "LS\nTahmini\n(128)",
        "residual_out": "Residual\nÇıkış",
        "sum": "Toplama",
        "channel_est": "Kanal\nTahmini",
        "skip": "Residual bağlantı",
        "train_title": "Eğitim",
        "train_line1": "Giriş: LS tahmini  ·  Hedef: residual",
        "train_line2": "Kayıp: MSE  ·  Optimizer: Adam",
        "output": "dnn_diagram.png",
    },
    "en": {
        "title": "LS + DNN Channel Estimation — Flowchart",
        "ls_input": "LS\nEstimate\n(128)",
        "residual_out": "Residual\nOutput",
        "sum": "Sum",
        "channel_est": "Channel\nEstimate",
        "skip": "Residual connection",
        "train_title": "Training",
        "train_line1": "Input: LS estimate  ·  Target: residual",
        "train_line2": "Loss: MSE  ·  Optimizer: Adam",
        "output": "dnn_diagram_en.png",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for diagram language."""
    parser = argparse.ArgumentParser(description="Draw DNN architecture flowchart.")
    parser.add_argument("--lang", type=str, default="en", choices=["tr", "en"])
    return parser.parse_args()


def main() -> None:
    """Save DNN flowchart to figures directory."""
    args = parse_args()
    text = LABELS[args.lang]
    fig, ax = plt.subplots(figsize=(16.5, 6.5))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    outer = FancyBboxPatch(
        (0.2, 0.35),
        16.0,
        5.8,
        boxstyle="round,pad=0.06,rounding_size=0.25",
        linewidth=2.0,
        edgecolor="#222",
        facecolor="white",
    )
    ax.add_patch(outer)
    ax.text(
        8.1, 5.85,
        text["title"],
        ha="center", fontsize=FONT_TITLE, fontweight="bold",
    )

    y_main = 2.75

    _block(ax, 0.5, y_main, 1.5, 1.05, text["ls_input"], facecolor="#f5f5f5")

    # DNN katmanları
    _block(ax, 2.5, y_main, 1.5, 1.05, "Dense\n(256)", facecolor="#ebf4ff")
    _block(ax, 4.3, y_main, 1.1, 1.05, "ReLU", facecolor="#dbeafe")
    _block(ax, 5.8, y_main, 1.5, 1.05, "Dense\n(256)", facecolor="#ebf4ff")
    _block(ax, 7.6, y_main, 1.1, 1.05, "ReLU", facecolor="#dbeafe")
    _block(ax, 9.1, y_main, 1.5, 1.05, "Dense\n(128)", facecolor="#ebf4ff")

    ax.add_patch(FancyBboxPatch(
        (2.35, 2.65), 8.45, 1.35,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor="#2b6cb0", facecolor="none", linestyle="--",
    ))
    ax.text(6.55, 4.15, "DNN", ha="center", fontsize=FONT_DNN_LABEL, fontweight="bold", color="#2b6cb0")

    _block(ax, 11.0, y_main, 1.4, 1.05, text["residual_out"], facecolor="#ebf4ff")
    _circle(ax, 12.85, y_main + 0.52, 0.5, text["sum"])
    _block(ax, 13.6, y_main, 1.5, 1.05, text["channel_est"], facecolor="#f0fff4")

    # Oklar — ana akış
    for x1, x2 in [
        (2.0, 2.5),
        (4.0, 4.3),
        (5.4, 5.8),
        (7.3, 7.6),
        (8.7, 9.1),
        (10.6, 11.0),
    ]:
        _arrow(ax, x1, y_main + 0.52, x2, y_main + 0.52)

    _arrow(ax, 12.4, y_main + 0.52, 12.35, y_main + 0.52)
    _arrow(ax, 13.3, y_main + 0.52, 13.6, y_main + 0.52)

    # Skip connection: LS → Toplama
    skip = FancyArrowPatch(
        (1.25, y_main + 0.05),
        (12.85, y_main + 0.12),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        linestyle="--",
        color="#555",
        connectionstyle="arc3,rad=-0.18",
    )
    ax.add_patch(skip)
    ax.text(7.2, 2.42, text["skip"], ha="center", fontsize=FONT_SKIP, color="#555", style="italic")

    # Eğitim kutusu (altta)
    ax.add_patch(FancyBboxPatch(
        (1.0, 0.45), 14.0, 1.45,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.3, edgecolor="#888", facecolor="#fafafa",
    ))
    ax.text(8.0, 1.62, text["train_title"], ha="center", fontsize=FONT_TRAIN_TITLE, fontweight="bold", color="#555")
    ax.text(8.0, 1.22, text["train_line1"], ha="center", fontsize=FONT_TRAIN_BODY, color="#666")
    ax.text(8.0, 0.82, text["train_line2"], ha="center", fontsize=FONT_TRAIN_BODY, color="#666")

    figures_dir = PROJECT_ROOT / "data" / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / text["output"]
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()
    print(f"Saved figure: {out_path}")


def _block(ax, x, y, w, h, text, facecolor="white") -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.6, edgecolor="#222", facecolor=facecolor,
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
