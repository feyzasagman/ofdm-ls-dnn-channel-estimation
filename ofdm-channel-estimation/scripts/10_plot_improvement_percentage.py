"""Plot SNR-wise percentage improvement of LS+DNN over LS and MMSE."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for improvement percentage plots."""
    parser = argparse.ArgumentParser(description="Plot LS+DNN improvement percentages vs SNR.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(METRICS_DIR / "comparison_per_snr.csv"),
        help="Path to per-SNR comparison CSV.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional suffix for output filenames.")
    return parser.parse_args()


def main() -> None:
    """Compute improvement percentages and save plots (+ optional CSV)."""
    args = parse_args()
    csv_path = Path(args.input_csv)
    tag = _normalize_tag(args.tag)

    rows = _read_comparison_csv(csv_path)
    rows = sorted(rows, key=lambda r: r["snr_db"])

    for row in rows:
        row["ls_improvement_mse_pct"] = _improvement_pct(row["ls_mse"], row["ls_dnn_mse"])
        row["mmse_improvement_mse_pct"] = _improvement_pct(row["mmse_mse"], row["ls_dnn_mse"])
        row["ls_improvement_nmse_pct"] = _improvement_pct(row["ls_nmse"], row["ls_dnn_nmse"])
        row["mmse_improvement_nmse_pct"] = _improvement_pct(row["mmse_nmse"], row["ls_dnn_nmse"])

    snr = np.array([r["snr_db"] for r in rows], dtype=float)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mse_fig = FIGURES_DIR / _with_optional_tag("improvement_percentage_mse.png", tag)
    nmse_fig = FIGURES_DIR / _with_optional_tag("improvement_percentage_nmse.png", tag)
    out_csv = METRICS_DIR / _with_optional_tag("improvement_percentage.csv", tag)

    _plot_improvement(
        snr=snr,
        ls_vals=np.array([r["ls_improvement_mse_pct"] for r in rows]),
        mmse_vals=np.array([r["mmse_improvement_mse_pct"] for r in rows]),
        title="LS+DNN Improvement over LS and MMSE (MSE)",
        out_path=mse_fig,
    )
    _plot_improvement(
        snr=snr,
        ls_vals=np.array([r["ls_improvement_nmse_pct"] for r in rows]),
        mmse_vals=np.array([r["mmse_improvement_nmse_pct"] for r in rows]),
        title="LS+DNN Improvement over LS and MMSE (NMSE)",
        out_path=nmse_fig,
    )
    _write_improvement_csv(out_csv, rows)

    print("Improvement percentage plots saved.")
    print(f"Input CSV     : {csv_path}")
    print(f"Saved figure  : {mse_fig}")
    print(f"Saved figure  : {nmse_fig}")
    print(f"Saved CSV     : {out_csv}")


def _read_comparison_csv(path: Path) -> list[dict[str, float]]:
    """Read required columns from comparison per-SNR CSV."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    required = (
        "snr_db",
        "ls_mse",
        "ls_nmse",
        "mmse_mse",
        "mmse_nmse",
        "ls_dnn_mse",
        "ls_dnn_nmse",
    )
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = [col for col in required if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            rows.append({col: float(row[col]) for col in required})
    if not rows:
        raise ValueError("CSV contains no data rows.")
    return rows


def _improvement_pct(baseline: float, improved: float) -> float:
    """Compute positive improvement percentage: (baseline - improved) / baseline * 100."""
    if baseline == 0.0:
        return 0.0
    return ((baseline - improved) / baseline) * 100.0


def _plot_improvement(
    snr: np.ndarray,
    ls_vals: np.ndarray,
    mmse_vals: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Plot improvement curves for LS and MMSE baselines."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(snr, ls_vals, marker="o", linewidth=1.8, label="vs LS")
    plt.plot(snr, mmse_vals, marker="s", linewidth=1.8, label="vs MMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Improvement (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_improvement_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Save computed improvement percentages to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "ls_improvement_mse_pct",
        "mmse_improvement_mse_pct",
        "ls_improvement_nmse_pct",
        "mmse_improvement_nmse_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _normalize_tag(tag: str | None) -> str | None:
    """Normalize optional tag string."""
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
