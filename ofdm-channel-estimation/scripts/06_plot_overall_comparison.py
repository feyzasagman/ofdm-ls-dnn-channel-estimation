"""Plot overall MSE/NMSE comparison across AWGN, Rayleigh and Rician."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures"

CHANNEL_FILES = {
    "AWGN": METRICS_DIR / "comparison_overall_awgn.json",
    "Rayleigh": METRICS_DIR / "comparison_overall_rayleigh.json",
    "Rician": METRICS_DIR / "comparison_overall_rician.json",
}


def main() -> None:
    """Load overall metrics and save grouped bar charts."""
    metrics = {name: _load_overall(path) for name, path in CHANNEL_FILES.items()}
    channels = list(CHANNEL_FILES.keys())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mse_path = FIGURES_DIR / "overall_mse_comparison.png"
    nmse_path = FIGURES_DIR / "overall_nmse_comparison.png"

    _plot_grouped_bars(
        channels=channels,
        ls_values=[metrics[c]["ls_mse"] for c in channels],
        mmse_values=[metrics[c]["mmse_mse"] for c in channels],
        dnn_values=[metrics[c]["ls_dnn_mse"] for c in channels],
        ylabel="MSE",
        title="Overall MSE Comparison Across Channel Models",
        out_path=mse_path,
    )
    _plot_grouped_bars(
        channels=channels,
        ls_values=[metrics[c]["ls_nmse"] for c in channels],
        mmse_values=[metrics[c]["mmse_nmse"] for c in channels],
        dnn_values=[metrics[c]["ls_dnn_nmse"] for c in channels],
        ylabel="NMSE",
        title="Overall NMSE Comparison Across Channel Models",
        out_path=nmse_path,
    )

    print("Overall channel comparison plots saved.")
    print("Input files:")
    for name, path in CHANNEL_FILES.items():
        print(f"  {name}: {path}")
    print(f"Saved figure: {mse_path}")
    print(f"Saved figure: {nmse_path}")


def _load_overall(path: Path) -> dict[str, float]:
    """Read overall comparison JSON and return metric fields."""
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ls_mse": float(payload["ls_mse"]),
        "ls_nmse": float(payload["ls_nmse"]),
        "mmse_mse": float(payload["mmse_mse"]),
        "mmse_nmse": float(payload["mmse_nmse"]),
        "ls_dnn_mse": float(payload["ls_dnn_mse"]),
        "ls_dnn_nmse": float(payload["ls_dnn_nmse"]),
    }


def _plot_grouped_bars(
    channels: list[str],
    ls_values: list[float],
    mmse_values: list[float],
    dnn_values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot grouped bars for LS, MMSE and LS+DNN per channel."""
    x = np.arange(len(channels))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, ls_values, width, label="LS")
    plt.bar(x, mmse_values, width, label="MMSE")
    plt.bar(x + width, dnn_values, width, label="LS+DNN")
    plt.xticks(x, channels)
    plt.xlabel("Channel Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
