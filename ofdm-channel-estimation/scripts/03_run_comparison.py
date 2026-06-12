"""Run first LS vs MMSE vs LS+DNN comparison on a shared test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.metrics import mse, nmse
from src.dnn.evaluate import evaluate_dnn


def parse_args() -> argparse.Namespace:
    """Parse CLI args for estimator comparison."""
    parser = argparse.ArgumentParser(description="Compare LS, MMSE and LS+DNN.")
    parser.add_argument("--raw-path", type=str, required=True, help="Path to raw dataset .npz")
    parser.add_argument("--processed-path", type=str, required=True, help="Path to processed dataset .npz")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained DNN model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-prefix", type=str, default="first_comparison")
    parser.add_argument("--tag", type=str, default=None, help="Optional suffix for output filenames.")
    return parser.parse_args()


def main() -> None:
    """Evaluate DNN and compare against LS/MMSE baselines."""
    args = parse_args()
    raw_path = Path(args.raw_path)
    processed_path = Path(args.processed_path)
    model_path = Path(args.model_path)

    eval_out = evaluate_dnn(
        processed_dataset_path=processed_path,
        model_path=model_path,
        residual_mode=None,
    )

    eval_npz = np.load(Path(eval_out["results_path"]), allow_pickle=False)
    dnn_pred = np.asarray(eval_npz["pred_channel"], dtype=np.complex128)
    dnn_true = np.asarray(eval_npz["true_channel"], dtype=np.complex128)
    snr_test = np.asarray(eval_npz["snr_test"], dtype=np.float64)

    raw_npz = np.load(raw_path, allow_pickle=False)
    ls_all = np.asarray(raw_npz["ls_estimate"], dtype=np.complex128)
    mmse_all = np.asarray(raw_npz["mmse_estimate"], dtype=np.complex128)
    true_all = np.asarray(raw_npz["true_channel"], dtype=np.complex128)
    snr_all = np.asarray(raw_npz["snr_db"], dtype=np.float64)

    split = _split_indices(
        n_samples=ls_all.shape[0],
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    test_idx = split["test"]
    ls_test = ls_all[test_idx]
    mmse_test = mmse_all[test_idx]
    true_test = true_all[test_idx]
    snr_test_from_raw = snr_all[test_idx]

    if ls_test.shape != dnn_pred.shape:
        raise ValueError(
            "Raw test split and DNN test predictions have mismatched shapes. "
            "Use the same seed/test_ratio/val_ratio used in preprocessing."
        )
    if snr_test_from_raw.shape != snr_test.shape:
        raise ValueError("snr_db test split shape mismatch between raw and processed data.")

    ls_mse = mse(true_test, ls_test)
    ls_nmse = nmse(true_test, ls_test)
    mmse_mse = mse(true_test, mmse_test)
    mmse_nmse = nmse(true_test, mmse_test)
    dnn_mse = mse(dnn_true, dnn_pred)
    dnn_nmse = nmse(dnn_true, dnn_pred)

    per_snr_rows = _compute_per_snr_rows(
        true_test=true_test,
        ls_test=ls_test,
        mmse_test=mmse_test,
        dnn_pred=dnn_pred,
        snr_test=snr_test_from_raw,
    )
    per_snr = {str(row["snr_db"]): row for row in per_snr_rows}

    metrics_dir = PROJECT_ROOT / "data" / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag = _normalize_tag(args.tag)
    npz_out = metrics_dir / _with_optional_tag(f"{args.output_prefix}.npz", tag)
    json_out = metrics_dir / _with_optional_tag(f"{args.output_prefix}.json", tag)
    overall_json_out = metrics_dir / _with_optional_tag("comparison_overall.json", tag)
    per_snr_json_out = metrics_dir / _with_optional_tag("comparison_per_snr.json", tag)
    per_snr_csv_out = metrics_dir / _with_optional_tag("comparison_per_snr.csv", tag)

    np.savez_compressed(
        npz_out,
        snr_test=snr_test,
        ls_test=ls_test,
        mmse_test=mmse_test,
        dnn_pred=dnn_pred,
        true_test=true_test,
        ls_mse=ls_mse,
        ls_nmse=ls_nmse,
        mmse_mse=mmse_mse,
        mmse_nmse=mmse_nmse,
        dnn_mse=dnn_mse,
        dnn_nmse=dnn_nmse,
    )

    payload = {
        "overall": {
            "ls_mse": ls_mse,
            "ls_nmse": ls_nmse,
            "mmse_mse": mmse_mse,
            "mmse_nmse": mmse_nmse,
            "dnn_mse": dnn_mse,
            "dnn_nmse": dnn_nmse,
        },
        "per_snr": per_snr,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "model_path": str(model_path),
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    overall_payload = {
        "ls_mse": ls_mse,
        "ls_nmse": ls_nmse,
        "mmse_mse": mmse_mse,
        "mmse_nmse": mmse_nmse,
        "ls_dnn_mse": dnn_mse,
        "ls_dnn_nmse": dnn_nmse,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "model_path": str(model_path),
    }
    overall_json_out.write_text(json.dumps(overall_payload, indent=2), encoding="utf-8")
    per_snr_json_out.write_text(json.dumps(per_snr_rows, indent=2), encoding="utf-8")
    _write_per_snr_csv(per_snr_csv_out, per_snr_rows)

    print("Comparison completed.")
    print(f"Output tag          : {tag or '(none)'}")
    print(f"Overall LS     MSE/NMSE: {ls_mse:.6f} / {ls_nmse:.6f}")
    print(f"Overall MMSE   MSE/NMSE: {mmse_mse:.6f} / {mmse_nmse:.6f}")
    print(f"Overall LS+DNN MSE/NMSE: {dnn_mse:.6f} / {dnn_nmse:.6f}")
    print("Per-SNR detail:")
    for row in per_snr_rows:
        print(
            f"SNR={int(row['snr_db']):>3} dB | "
            f"LS MSE={row['ls_mse']:.6f} | "
            f"MMSE MSE={row['mmse_mse']:.6f} | "
            f"LS+DNN MSE={row['ls_dnn_mse']:.6f}"
        )
    print(f"Saved metrics NPZ : {npz_out}")
    print(f"Saved metrics JSON: {json_out}")
    print(f"Saved overall JSON: {overall_json_out}")
    print(f"Saved per-SNR JSON: {per_snr_json_out}")
    print(f"Saved per-SNR CSV : {per_snr_csv_out}")


def _split_indices(
    n_samples: int,
    test_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce preprocessing split to get the same test subset."""
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_samples)

    n_test = max(1, int(round(n_samples * test_ratio)))
    remaining = n_samples - n_test
    n_val = max(1, int(round(remaining * val_ratio))) if val_ratio > 0 else 0

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _metric_per_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    metric_fn,
) -> dict[str, float]:
    """Compute metric separately for each SNR."""
    result: dict[str, float] = {}
    for snr_val in np.unique(snr):
        mask = snr == snr_val
        result[str(float(snr_val))] = float(metric_fn(y_true[mask], y_pred[mask]))
    return result


def _compute_per_snr_rows(
    true_test: np.ndarray,
    ls_test: np.ndarray,
    mmse_test: np.ndarray,
    dnn_pred: np.ndarray,
    snr_test: np.ndarray,
) -> list[dict[str, float]]:
    """Compute LS/MMSE/LS+DNN MSE and NMSE for each SNR group."""
    rows: list[dict[str, float]] = []
    for snr_val in np.sort(np.unique(snr_test)):
        mask = snr_test == snr_val
        row = {
            "snr_db": float(snr_val),
            "ls_mse": float(mse(true_test[mask], ls_test[mask])),
            "ls_nmse": float(nmse(true_test[mask], ls_test[mask])),
            "mmse_mse": float(mse(true_test[mask], mmse_test[mask])),
            "mmse_nmse": float(nmse(true_test[mask], mmse_test[mask])),
            "ls_dnn_mse": float(mse(true_test[mask], dnn_pred[mask])),
            "ls_dnn_nmse": float(nmse(true_test[mask], dnn_pred[mask])),
        }
        rows.append(row)
    return rows


def _write_per_snr_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    """Write per-SNR comparison rows to CSV."""
    fieldnames = [
        "snr_db",
        "ls_mse",
        "ls_nmse",
        "mmse_mse",
        "mmse_nmse",
        "ls_dnn_mse",
        "ls_dnn_nmse",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
