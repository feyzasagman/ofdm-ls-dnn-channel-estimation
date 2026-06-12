"""Plot DNN training and validation loss curves from history files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for training history plotting."""
    parser = argparse.ArgumentParser(description="Plot training vs validation loss.")
    parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to training history file (.json or .npz).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix for output figure filename.",
    )
    return parser.parse_args()


def main() -> None:
    """Load history and save training/validation loss figure."""
    args = parse_args()
    history_path = Path(args.history_path)
    tag = _normalize_tag(args.tag)

    loss, val_loss = _load_history(history_path)
    epochs = np.arange(1, len(loss) + 1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / _with_optional_tag("training_vs_validation_loss.png", tag)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, loss, marker="o", markersize=3, linewidth=1.6, label="Training Loss")
    plt.plot(epochs, val_loss, marker="s", markersize=3, linewidth=1.6, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Training history plot saved.")
    print(f"Input file  : {history_path}")
    print(f"Saved figure: {out_path}")


def _load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load loss and val_loss from JSON or NPZ history file."""
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        loss = np.asarray(payload["loss"], dtype=float)
        val_loss = np.asarray(payload["val_loss"], dtype=float)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        loss = np.asarray(data["loss"], dtype=float)
        val_loss = np.asarray(data["val_loss"], dtype=float)
    else:
        raise ValueError("History file must be .json or .npz")

    if loss.size == 0 or val_loss.size == 0:
        raise ValueError("History must contain non-empty 'loss' and 'val_loss'.")
    if loss.shape != val_loss.shape:
        raise ValueError("loss and val_loss must have the same length.")
    return loss, val_loss


def _normalize_tag(tag: str | None) -> str | None:
    """Normalize optional tag string for filenames."""
    if tag is None:
        return None
    normalized = tag.strip()
    return normalized if normalized else None


def _with_optional_tag(filename: str, tag: str | None) -> str:
    """Append tag before file extension when provided."""
    if tag is None:
        return filename
    path = Path(filename)
    return f"{path.stem}_{tag}{path.suffix}"


if __name__ == "__main__":
    main()
