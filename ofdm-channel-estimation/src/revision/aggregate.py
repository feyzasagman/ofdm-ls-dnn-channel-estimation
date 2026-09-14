"""Multi-seed aggregation for revision experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def aggregate_per_snr_metrics(
    seed_results: list[dict[str, Any]],
    value_keys: list[str],
) -> dict[str, Any]:
    """Aggregate per-SNR metrics across seeds (mean/std)."""
    if not seed_results:
        raise ValueError("seed_results must be non-empty.")

    snr_values = sorted({row["snr_db"] for r in seed_results for row in r["per_snr"]})
    aggregated: list[dict[str, Any]] = []
    for snr in snr_values:
        row: dict[str, Any] = {"snr_db": float(snr)}
        for key in value_keys:
            vals = np.asarray(
                [float(next(x for x in r["per_snr"] if x["snr_db"] == snr)[key]) for r in seed_results],
                dtype=np.float64,
            )
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals))
        aggregated.append(row)

    overall: dict[str, Any] = {}
    for key in value_keys:
        vals = np.asarray([float(r["overall"][key]) for r in seed_results], dtype=np.float64)
        overall[f"{key}_mean"] = float(np.mean(vals))
        overall[f"{key}_std"] = float(np.std(vals))

    return {"per_snr": aggregated, "overall": overall, "n_seeds": len(seed_results)}


def aggregate_ber_results(seed_ber_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate BER per SNR across seeds."""
    if not seed_ber_results:
        raise ValueError("seed_ber_results must be non-empty.")

    methods = seed_ber_results[0]["methods"]
    snr_keys = sorted(seed_ber_results[0]["per_snr"].keys(), key=float)
    per_snr: list[dict[str, Any]] = []
    for snr_key in snr_keys:
        row: dict[str, Any] = {"snr_db": float(snr_key)}
        for method in methods:
            bers = np.asarray(
                [float(r["per_snr"][snr_key][method]["ber"]) for r in seed_ber_results],
                dtype=np.float64,
            )
            row[f"{method}_ber_mean"] = float(np.mean(bers))
            row[f"{method}_ber_std"] = float(np.std(bers))
        per_snr.append(row)

    overall: dict[str, Any] = {}
    for method in methods:
        bers = np.asarray([float(r["overall"][method]["ber"]) for r in seed_ber_results], dtype=np.float64)
        overall[f"{method}_ber_mean"] = float(np.mean(bers))
        overall[f"{method}_ber_std"] = float(np.std(bers))

    return {"per_snr": per_snr, "overall": overall, "n_seeds": len(seed_ber_results)}


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
