"""Plot SNR-based MSE/NMSE comparison figures from metrics files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse CLI options for plotting comparison results."""
    parser = argparse.ArgumentParser(description="Plot SNR vs MSE/NMSE curves.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "results" / "metrics" / "comparison_per_snr.csv"),
        help="Path to comparison_per_snr.csv",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "results" / "metrics" / "comparison_per_snr.json"),
        help="Fallback path to comparison_per_snr.json",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix for output figure filenames (e.g. awgn, rayleigh).",
    )
    return parser.parse_args()


def main() -> None:
    """Load per-SNR metrics and save MSE/NMSE figures."""
    args = parse_args()
    csv_path = Path(args.input_csv)
    json_path = Path(args.input_json)

    rows, used_input = _load_rows(csv_path, json_path)
    rows = sorted(rows, key=lambda r: r["snr_db"])

    snr = np.array([row["snr_db"] for row in rows], dtype=float)
    ls_mse = np.array([row["ls_mse"] for row in rows], dtype=float)
    mmse_mse = np.array([row["mmse_mse"] for row in rows], dtype=float)
    dnn_mse = np.array([row["ls_dnn_mse"] for row in rows], dtype=float)
    ls_nmse = np.array([row["ls_nmse"] for row in rows], dtype=float)
    mmse_nmse = np.array([row["mmse_nmse"] for row in rows], dtype=float)
    dnn_nmse = np.array([row["ls_dnn_nmse"] for row in rows], dtype=float)

    figures_dir = PROJECT_ROOT / "data" / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tag = _normalize_tag(args.tag)
    mse_path = figures_dir / _with_optional_tag("snr_vs_mse.png", tag)
    nmse_path = figures_dir / _with_optional_tag("snr_vs_nmse.png", tag)

    _plot_three_curves(
        x=snr,
        y1=ls_mse,
        y2=mmse_mse,
        y3=dnn_mse,
        ylabel="MSE",
        title="SNR vs MSE for Channel Estimation",
        out_path=mse_path,
    )
    _plot_three_curves(
        x=snr,
        y1=ls_nmse,
        y2=mmse_nmse,
        y3=dnn_nmse,
        ylabel="NMSE",
        title="SNR vs NMSE for Channel Estimation",
        out_path=nmse_path,
    )

    print(f"Input metrics file : {used_input}")
    print(f"Saved figure       : {mse_path}")
    print(f"Saved figure       : {nmse_path}")


def _load_rows(csv_path: Path, json_path: Path) -> tuple[list[dict[str, float]], Path]:
    """Load per-SNR rows from CSV first, otherwise JSON."""
    if csv_path.exists():
        rows: list[dict[str, float]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "snr_db": float(row["snr_db"]),
                        "ls_mse": float(row["ls_mse"]),
                        "ls_nmse": float(row["ls_nmse"]),
                        "mmse_mse": float(row["mmse_mse"]),
                        "mmse_nmse": float(row["mmse_nmse"]),
                        "ls_dnn_mse": float(row["ls_dnn_mse"]),
                        "ls_dnn_nmse": float(row["ls_dnn_nmse"]),
                    }
                )
        return rows, csv_path

    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = [
            {
                "snr_db": float(item["snr_db"]),
                "ls_mse": float(item["ls_mse"]),
                "ls_nmse": float(item["ls_nmse"]),
                "mmse_mse": float(item["mmse_mse"]),
                "mmse_nmse": float(item["mmse_nmse"]),
                "ls_dnn_mse": float(item["ls_dnn_mse"]),
                "ls_dnn_nmse": float(item["ls_dnn_nmse"]),
            }
            for item in payload
        ]
        return rows, json_path

    raise FileNotFoundError(
        f"Neither CSV nor JSON metrics file found:\n- {csv_path}\n- {json_path}"
    )


def _plot_three_curves(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot LS/MMSE/LS+DNN curves on a single figure and save."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y1, marker="o", linewidth=1.8, label="LS")
    plt.plot(x, y2, marker="s", linewidth=1.8, label="MMSE")
    plt.plot(x, y3, marker="^", linewidth=1.8, label="LS+DNN")
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
