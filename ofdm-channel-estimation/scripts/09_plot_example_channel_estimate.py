"""Plot one example subcarrier-wise channel magnitude comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"

TRUE_KEYS = ("true_channel", "true_test", "true")
LS_KEYS = ("ls_estimate", "ls_test", "h_ls", "ls")
MMSE_KEYS = ("mmse_estimate", "mmse_test", "h_mmse", "mmse")
DNN_KEYS = ("pred_channel", "dnn_pred", "corrected_y", "ls_dnn", "h_dnn", "dnn")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for example channel estimate plot."""
    parser = argparse.ArgumentParser(description="Plot true/LS/MMSE/LS+DNN for one sample.")
    parser.add_argument("--eval-path", type=str, required=True, help="Path to eval or comparison .npz")
    parser.add_argument("--tag", type=str, required=True, help="Suffix for output filename.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to visualize.")
    return parser.parse_args()


def main() -> None:
    """Load arrays and save magnitude comparison figure for one sample."""
    args = parse_args()
    eval_path = Path(args.eval_path)
    tag = args.tag.strip()
    if not tag:
        raise ValueError("--tag must be non-empty.")

    data = np.load(eval_path, allow_pickle=False)
    true_ch = _pick_array(data, TRUE_KEYS, "true channel")
    dnn_ch = _pick_array(data, DNN_KEYS, "LS+DNN prediction")

    try:
        ls_ch = _pick_array(data, LS_KEYS, "LS estimate")
        mmse_ch = _pick_array(data, MMSE_KEYS, "MMSE estimate")
        used_fields = _fields_used(data, true_ch, ls_ch, mmse_ch, dnn_ch)
    except KeyError:
        fallback = METRICS_DIR / f"first_comparison_{tag}.npz"
        if not fallback.exists():
            raise FileNotFoundError(
                "LS/MMSE not found in eval file and fallback missing: "
                f"{fallback}"
            )
        cmp_data = np.load(fallback, allow_pickle=False)
        ls_ch = _pick_array(cmp_data, LS_KEYS, "LS estimate")
        mmse_ch = _pick_array(cmp_data, MMSE_KEYS, "MMSE estimate")
        used_fields = {
            "source_eval": str(eval_path),
            "source_comparison": str(fallback),
            "true": _matched_key(data, TRUE_KEYS) or _matched_key(cmp_data, TRUE_KEYS),
            "ls": _matched_key(cmp_data, LS_KEYS),
            "mmse": _matched_key(cmp_data, MMSE_KEYS),
            "dnn": _matched_key(data, DNN_KEYS),
        }

    idx = args.sample_index
    _validate_index(idx, true_ch.shape[0])

    subcarriers = np.arange(true_ch.shape[1])
    true_mag = np.abs(true_ch[idx])
    ls_mag = np.abs(ls_ch[idx])
    mmse_mag = np.abs(mmse_ch[idx])
    dnn_mag = np.abs(dnn_ch[idx])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"example_channel_estimate_{tag}.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(subcarriers, true_mag, linewidth=1.8, label="True Channel")
    plt.plot(subcarriers, ls_mag, linewidth=1.5, linestyle="--", label="LS")
    plt.plot(subcarriers, mmse_mag, linewidth=1.5, linestyle="-.", label="MMSE")
    plt.plot(subcarriers, dnn_mag, linewidth=1.5, linestyle=":", label="LS+DNN")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.title("Example Channel Estimate Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Example channel estimate plot saved.")
    print(f"Eval path     : {eval_path}")
    print(f"Sample index  : {idx}")
    print(f"Fields used   : {used_fields}")
    print(f"Saved figure  : {out_path}")


def _pick_array(data: np.lib.npyio.NpzFile, candidates: tuple[str, ...], label: str) -> np.ndarray:
    """Return complex channel array from NPZ (direct complex or Re/Im forms)."""
    key = _matched_key(data, candidates)
    if key is not None:
        return _to_complex(np.asarray(data[key]))

    for base in candidates:
        real_key, imag_key = f"{base}_real", f"{base}_imag"
        if real_key in data.files and imag_key in data.files:
            real = np.asarray(data[real_key], dtype=np.float64)
            imag = np.asarray(data[imag_key], dtype=np.float64)
            return real + 1j * imag

    raise KeyError(label)


def _to_complex(arr: np.ndarray) -> np.ndarray:
    """Convert complex array or [Re, Im] feature matrix to complex channel matrix."""
    if np.iscomplexobj(arr):
        out = np.asarray(arr, dtype=np.complex128)
        if out.ndim != 2:
            raise ValueError("Complex channel array must be 2D.")
        return out

    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Real channel array must be 2D.")
    if x.shape[1] % 2 != 0:
        raise ValueError("Real feature array must have even width [Re, Im].")
    half = x.shape[1] // 2
    return x[:, :half] + 1j * x[:, half:]


def _matched_key(data: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str | None:
    """Find first existing key among candidates."""
    for key in candidates:
        if key in data.files:
            return key
    return None


def _fields_used(
    data: np.lib.npyio.NpzFile,
    true_ch: np.ndarray,
    ls_ch: np.ndarray,
    mmse_ch: np.ndarray,
    dnn_ch: np.ndarray,
) -> dict[str, str]:
    """Report which NPZ keys were selected."""
    del true_ch, ls_ch, mmse_ch, dnn_ch
    return {
        "true": _matched_key(data, TRUE_KEYS) or "",
        "ls": _matched_key(data, LS_KEYS) or "",
        "mmse": _matched_key(data, MMSE_KEYS) or "",
        "dnn": _matched_key(data, DNN_KEYS) or "",
    }


def _validate_index(idx: int, n_samples: int) -> None:
    """Validate sample index bounds."""
    if idx < 0 or idx >= n_samples:
        raise IndexError(f"sample-index must be in [0, {n_samples - 1}], got {idx}.")


if __name__ == "__main__":
    main()
