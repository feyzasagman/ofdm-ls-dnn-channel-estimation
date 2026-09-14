"""High-SNR error-floor analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


HIGH_SNR_LEVELS: tuple[float, ...] = (20.0, 25.0, 30.0)

DIAGNOSTIC_HYPOTHESES: tuple[str, ...] = (
    "mixed-SNR training may leave residual bias at high SNR",
    "residual prediction bias when LS is already accurate",
    "limited training samples per SNR stratum",
    "float precision limits for very small NMSE differences",
    "model capacity may be insufficient for near-oracle refinement",
)


def analyze_high_snr_error_floor(
    per_snr_rows: list[dict[str, float]],
    channel_type: str,
    seed: int,
) -> dict[str, Any]:
    """Detect SNR points where LS+DNN is worse than LS."""
    table: list[dict[str, Any]] = []
    for row in per_snr_rows:
        snr = float(row["snr_db"])
        if snr not in HIGH_SNR_LEVELS:
            continue
        ls_mse = float(row["ls_mse"])
        dnn_mse = float(row["ls_dnn_mse"])
        dnn_worse = dnn_mse > ls_mse
        ratio = dnn_mse / ls_mse if ls_mse > 0 else float("inf")
        pct_diff = 100.0 * (dnn_mse - ls_mse) / ls_mse if ls_mse > 0 else float("inf")
        table.append(
            {
                "channel_type": channel_type,
                "seed": int(seed),
                "snr_db": snr,
                "ls_mse": ls_mse,
                "ls_dnn_mse": dnn_mse,
                "dnn_worse_than_ls": bool(dnn_worse),
                "ratio_dnn_over_ls": float(ratio),
                "percent_difference": float(pct_diff),
            }
        )

    worse_points = [r for r in table if r["dnn_worse_than_ls"]]
    return {
        "channel_type": channel_type,
        "seed": int(seed),
        "high_snr_levels_db": list(HIGH_SNR_LEVELS),
        "rows": table,
        "n_worse_than_ls": len(worse_points),
        "diagnostic_hypotheses": list(DIAGNOSTIC_HYPOTHESES),
        "note": (
            "Diagnostic hypotheses are not proven causes; they require targeted "
            "ablation experiments."
        ),
    }
