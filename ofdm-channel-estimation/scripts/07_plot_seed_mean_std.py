"""Plot AWGN 3-seed aggregate overall MSE with mean and std error bars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = PROJECT_ROOT / "data" / "results" / "metrics" / "awgn_seed_aggregate_overall.json"
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures"
OUTPUT_PNG = FIGURES_DIR / "awgn_seed_mean_std_bar.png"

ESTIMATORS = ("LS", "MMSE", "LS+DNN")
MSE_KEYS = ("ls_mse", "mmse_mse", "ls_dnn_mse")
NMSE_KEYS = ("ls_nmse", "mmse_nmse", "ls_dnn_nmse")


def main() -> None:
    """Load aggregate JSON and save MSE bar chart with std error bars."""
    metrics = _load_aggregate(INPUT_JSON)
    means = [metrics[key]["mean"] for key in MSE_KEYS]
    stds = [metrics[key]["std"] for key in MSE_KEYS]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_mean_std_bar(
        estimators=ESTIMATORS,
        means=means,
        stds=stds,
        ylabel="MSE",
        title="AWGN Overall MSE (Mean ± Std over 3 Seeds)",
        out_path=OUTPUT_PNG,
    )

    print("AWGN seed aggregate plot saved.")
    print(f"Input file  : {INPUT_JSON}")
    print(f"Saved figure: {OUTPUT_PNG}")


def _load_aggregate(path: Path) -> dict[str, dict[str, float]]:
    """Read aggregate overall JSON with mean/std per metric."""
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: {"mean": float(payload[key]["mean"]), "std": float(payload[key]["std"])}
        for key in MSE_KEYS + NMSE_KEYS
    }


def _plot_mean_std_bar(
    estimators: tuple[str, ...],
    means: list[float],
    stds: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot bar chart with error bars for estimator comparison."""
    x = np.arange(len(estimators))

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(x, means, yerr=stds, capsize=5, width=0.55, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xticks(x, estimators)
    plt.xlabel("Estimator")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
