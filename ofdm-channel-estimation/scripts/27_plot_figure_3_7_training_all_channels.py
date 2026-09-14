"""Figure 3.7 — Composite LS+DNN training/validation loss (seed 42, all channels)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REVISION_ROOT = PROJECT_ROOT / "data" / "results" / "revision"
OUT_DIR = REVISION_ROOT / "final_figures"
STEM = "fig_training_all_channels"
SEED = 42

CHANNELS: list[tuple[str, str]] = [
    ("awgn", "(a) AWGN"),
    ("rayleigh", "(b) Rayleigh"),
    ("rician", "(c) Rician"),
]

CAPTION = (
    "Figure 3.7. Representative training and validation loss curves of the LS+DNN "
    "models for AWGN, Rayleigh, and Rician scenarios (seed = 42)."
)


def _load_history(channel: str) -> tuple[Path, dict]:
    path = REVISION_ROOT / "training" / f"history_{channel}_seed{SEED}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing revision history: {path}")
    resolved = path.resolve()
    if "archive_pre_revision" in resolved.parts:
        raise ValueError(f"Refusing archived source: {resolved}")
    if REVISION_ROOT.resolve() not in resolved.parents and resolved.parent != REVISION_ROOT / "training":
        raise ValueError(f"Unexpected history path outside revision tree: {resolved}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _verify_existing_panels() -> None:
    """Confirm individual final figures exist under revision/final_figures."""
    for channel, _ in CHANNELS:
        for ext in ("png",):
            p = OUT_DIR / f"fig_training_{channel}.{ext}"
            if not p.exists():
                raise FileNotFoundError(f"Missing reference figure: {p}")
            if "archive_pre_revision" in p.resolve().parts:
                raise ValueError(f"Refusing archived figure: {p}")
    print("Source verification:")
    for channel, _ in CHANNELS:
        hist_path, _ = _load_history(channel)
        print(f"  [OK] {channel}: {hist_path.relative_to(PROJECT_ROOT)}")


def build_composite() -> plt.Figure:
    histories: list[tuple[str, str, dict, Path]] = []
    ymax = 0.0
    for channel, panel_label in CHANNELS:
        path, hist = _load_history(channel)
        loss = np.asarray(hist["loss"], dtype=np.float64)
        val_loss = np.asarray(hist["val_loss"], dtype=np.float64)
        ymax = max(ymax, float(np.max(loss)), float(np.max(val_loss)))
        histories.append((channel, panel_label, hist, path))

    ymax *= 1.05

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    train_line = None
    val_line = None

    for ax, (channel, panel_label, hist, _) in zip(axes, histories, strict=True):
        loss = np.asarray(hist["loss"], dtype=np.float64)
        val_loss = np.asarray(hist["val_loss"], dtype=np.float64)
        epochs = np.arange(1, loss.size + 1)

        train_line, = ax.plot(epochs, loss, color="#1f77b4", linewidth=1.8, label="Training Loss")
        val_line, = ax.plot(epochs, val_loss, color="#d62728", linewidth=1.8, label="Validation Loss")

        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, ymax)
        ax.set_xlim(1, max(epochs[-1], 1))
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(
            0.03,
            0.97,
            panel_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            fontweight="600",
        )
        ax.tick_params(direction="out")

    axes[0].set_ylabel("MSE loss")
    fig.legend(
        [train_line, val_line],
        ["Training Loss", "Validation Loss"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.82, wspace=0.22)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_existing_panels()

    fig = build_composite()
    png_path = OUT_DIR / f"{STEM}.png"
    svg_path = OUT_DIR / f"{STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure_id": "Figure 3.7",
        "seed": SEED,
        "channels": [c for c, _ in CHANNELS],
        "sources": [
            str((REVISION_ROOT / "training" / f"history_{c}_seed{SEED}.json").relative_to(PROJECT_ROOT))
            for c, _ in CHANNELS
        ],
        "reference_figures": [f"final_figures/fig_training_{c}.png" for c, _ in CHANNELS],
        "caption": CAPTION,
        "note": "Loss values read directly from revision training histories; not recomputed.",
    }
    (OUT_DIR / f"{STEM}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT_DIR / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
