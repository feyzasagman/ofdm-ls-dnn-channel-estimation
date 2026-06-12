"""Aggregate multi-seed comparison metrics for AWGN experiments."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEEDS = (42, 123, 999)
METRICS_KEYS = (
    "ls_mse",
    "ls_nmse",
    "mmse_mse",
    "mmse_nmse",
    "ls_dnn_mse",
    "ls_dnn_nmse",
)


def main() -> None:
    """Aggregate overall and per-SNR metrics across predefined seeds."""
    metrics_dir = PROJECT_ROOT / "data" / "results" / "metrics"

    overall_files = [metrics_dir / f"comparison_overall_awgn_seed{seed}.json" for seed in SEEDS]
    per_snr_files = [metrics_dir / f"comparison_per_snr_awgn_seed{seed}.csv" for seed in SEEDS]

    overall_data = [_read_overall_json(path) for path in overall_files]
    per_snr_data = [_read_per_snr_csv(path) for path in per_snr_files]

    overall_agg = _aggregate_overall(overall_data)
    per_snr_agg_rows = _aggregate_per_snr(per_snr_data)

    overall_out = metrics_dir / "awgn_seed_aggregate_overall.json"
    per_snr_out = metrics_dir / "awgn_seed_aggregate_per_snr.csv"

    overall_out.write_text(json.dumps(overall_agg, indent=2), encoding="utf-8")
    _write_per_snr_csv(per_snr_out, per_snr_agg_rows)

    print("AWGN seed aggregation completed.")
    print(f"Seeds: {SEEDS}")
    print(f"Saved overall aggregate: {overall_out}")
    print(f"Saved per-SNR aggregate: {per_snr_out}")
    print(
        "Overall NMSE (mean±std) | "
        f"LS={overall_agg['ls_nmse']['mean']:.6f}±{overall_agg['ls_nmse']['std']:.6f}, "
        f"MMSE={overall_agg['mmse_nmse']['mean']:.6f}±{overall_agg['mmse_nmse']['std']:.6f}, "
        f"LS+DNN={overall_agg['ls_dnn_nmse']['mean']:.6f}±{overall_agg['ls_dnn_nmse']['std']:.6f}"
    )


def _read_overall_json(path: Path) -> dict[str, float]:
    """Read one overall metrics JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing overall metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(payload[key]) for key in METRICS_KEYS}


def _read_per_snr_csv(path: Path) -> dict[float, dict[str, float]]:
    """Read one per-SNR CSV file into snr -> metrics mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Missing per-SNR metrics file: {path}")

    out: dict[float, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = float(row["snr_db"])
            out[snr] = {
                "ls_mse": float(row["ls_mse"]),
                "ls_nmse": float(row["ls_nmse"]),
                "mmse_mse": float(row["mmse_mse"]),
                "mmse_nmse": float(row["mmse_nmse"]),
                "ls_dnn_mse": float(row["ls_dnn_mse"]),
                "ls_dnn_nmse": float(row["ls_dnn_nmse"]),
            }
    return out


def _aggregate_overall(overall_data: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean/std for overall metrics across seeds."""
    agg: dict[str, dict[str, float]] = {}
    for key in METRICS_KEYS:
        values = np.array([item[key] for item in overall_data], dtype=float)
        agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return agg


def _aggregate_per_snr(
    per_snr_data: list[dict[float, dict[str, float]]]
) -> list[dict[str, float]]:
    """Compute per-SNR mean/std for all required metrics."""
    snr_values = sorted(per_snr_data[0].keys())
    for data in per_snr_data[1:]:
        if set(data.keys()) != set(snr_values):
            raise ValueError("Per-SNR CSV files have mismatched SNR sets.")

    rows: list[dict[str, float]] = []
    for snr in snr_values:
        row: dict[str, float] = {"snr_db": float(snr)}
        for key in METRICS_KEYS:
            values = np.array([seed_data[snr][key] for seed_data in per_snr_data], dtype=float)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        rows.append(row)
    return rows


def _write_per_snr_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write per-SNR aggregated rows to CSV."""
    fieldnames = [
        "snr_db",
        "ls_mse_mean",
        "ls_mse_std",
        "ls_nmse_mean",
        "ls_nmse_std",
        "mmse_mse_mean",
        "mmse_mse_std",
        "mmse_nmse_mean",
        "mmse_nmse_std",
        "ls_dnn_mse_mean",
        "ls_dnn_mse_std",
        "ls_dnn_nmse_mean",
        "ls_dnn_nmse_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
