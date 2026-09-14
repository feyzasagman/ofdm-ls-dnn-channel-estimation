"""Archive pre-revision outputs and build final manuscript figures/tables."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ber_evaluation import ber_for_log_plot
from src.revision.ber_figure import plot_ber_curves_on_axes, style_ber_axes
from src.core.ofdm_params import OFDMParams
from src.core.pilots import generate_pilot_indices
from src.core.split import DEFAULT_SEEDS
from src.revision.comparison import predict_dnn_test_channels

REVISION_ROOT = PROJECT_ROOT / "data" / "results" / "revision"
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
ARCHIVE_ROOT = RESULTS_ROOT / "archive_pre_revision"
FINAL_FIGURES = REVISION_ROOT / "final_figures"
FINAL_TABLES = REVISION_ROOT / "final_tables"

CHANNELS = ("awgn", "rayleigh", "rician")
SEEDS = list(DEFAULT_SEEDS)
REPRESENTATIVE_SEED = 42

METHOD_STYLE = {
    "ls": {"label": "LS", "color": "#1f77b4", "marker": "o"},
    "simplified_mmse": {"label": "Simplified-MMSE", "color": "#ff7f0e", "marker": "s"},
    "lmmse_flat": {"label": "LMMSE-flat", "color": "#2ca02c", "marker": "^"},
    "ls_dnn": {"label": "LS+DNN", "color": "#d62728", "marker": "D"},
}

MSE_METRIC_MAP = {
    "ls": ("ls_mse_mean", "ls_mse_std"),
    "simplified_mmse": ("simplified_mmse_mse_mean", "simplified_mmse_mse_std"),
    "lmmse_flat": ("lmmse_flat_mse_mean", "lmmse_flat_mse_std"),
    "ls_dnn": ("ls_dnn_mse_mean", "ls_dnn_mse_std"),
}
NMSE_METRIC_MAP = {
    "ls": ("ls_nmse_mean", "ls_nmse_std"),
    "simplified_mmse": ("simplified_mmse_nmse_mean", "simplified_mmse_nmse_std"),
    "lmmse_flat": ("lmmse_flat_nmse_mean", "lmmse_flat_nmse_std"),
    "ls_dnn": ("ls_dnn_nmse_mean", "ls_dnn_nmse_std"),
}
BER_METRIC_MAP = {
    "ls": ("ls_ber_mean", "ls_ber_std"),
    "simplified_mmse": ("simplified_mmse_ber_mean", "simplified_mmse_ber_std"),
    "lmmse_flat": ("lmmse_flat_ber_mean", "lmmse_flat_ber_std"),
    "ls_dnn": ("ls_dnn_ber_mean", "ls_dnn_ber_std"),
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
    }
)


def archive_pre_revision_outputs() -> list[str]:
    """Move legacy result folders into archive_pre_revision."""
    moved: list[str] = []
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in ("figures", "metrics", "logs"):
        src = RESULTS_ROOT / folder
        if not src.exists():
            continue
        dst = ARCHIVE_ROOT / folder
        if dst.exists():
            for item in src.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(target))
                    moved.append(str(item.relative_to(RESULTS_ROOT)))
            for item in sorted(src.rglob("*"), reverse=True):
                if item.is_dir() and not any(item.iterdir()):
                    item.rmdir()
        else:
            shutil.move(str(src), str(dst))
            for item in dst.rglob("*"):
                if item.is_file():
                    moved.append(str(item.relative_to(ARCHIVE_ROOT)))
    return sorted(moved)


def write_archive_log(moved_files: list[str]) -> None:
    lines = [
        "# Archive Log — Pre-Revision Results",
        "",
        f"Archived at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Pre-revision outputs were moved from `data/results/{figures,metrics,logs}/` "
        "to `data/results/archive_pre_revision/` preserving folder structure.",
        "",
        "`data/results/revision/` was **not** modified.",
        "",
        f"Total files moved: {len(moved_files)}",
        "",
        "## Moved files",
        "",
    ]
    for path in moved_files:
        lines.append(f"- `{path}`")
    (ARCHIVE_ROOT / "ARCHIVE_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf"):
        out = FINAL_FIGURES / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        paths[ext] = str(out.relative_to(PROJECT_ROOT))
    plt.close(fig)
    return paths


def plot_metric_vs_snr(
    channel: str,
    metric: str,
    metric_map: dict[str, tuple[str, str]],
    stem: str,
) -> dict[str, str]:
    data = _load_json(REVISION_ROOT / "mse_nmse" / f"aggregate_{channel}.json")
    if metric == "ber":
        data = _load_json(REVISION_ROOT / "ber" / f"aggregate_{channel}.json")

    rows = data["per_snr"]
    snr = np.asarray([r["snr_db"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 5.8 if metric == "ber" else 5.5))
    if metric == "ber":
        style_map = {k: {**METHOD_STYLE[k], "label": METHOD_STYLE[k]["label"]} for k in metric_map}
        plot_ber_curves_on_axes(ax, snr, rows, tuple(metric_map.keys()), style_map)
        style_ber_axes(ax)
        ax.yaxis.set_major_locator(LogLocator(base=10))
    else:
        for key, (mean_k, std_k) in metric_map.items():
            style = METHOD_STYLE[key]
            y = np.asarray([r[mean_k] for r in rows], dtype=np.float64)
            y_std = np.asarray([r.get(std_k, 0.0) for r in rows], dtype=np.float64)
            y_plot = np.maximum(y, 1e-12)
            y_low = np.maximum(y - y_std, 1e-12)
            y_high = y + y_std
            ax.plot(snr, y_plot, marker=style["marker"], color=style["color"], label=style["label"], linewidth=2)
            ax.fill_between(snr, y_low, y_high, color=style["color"], alpha=0.18)
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    ylab = {"mse": "MSE", "nmse": "NMSE", "ber": "BER"}[metric]
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab} vs SNR — {channel.upper()} (mean ± std, seeds {SEEDS})")
    if metric != "ber":
        ax.legend(loc="best", frameon=True)
    else:
        ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    return _save_figure(fig, stem)


def plot_high_snr_error_floor(channel: str, stem: str) -> dict[str, str]:
    agg = _load_json(REVISION_ROOT / "mse_nmse" / f"aggregate_{channel}.json")
    rows = [r for r in agg["per_snr"] if r["snr_db"] >= 15.0]
    snr = np.asarray([r["snr_db"] for r in rows], dtype=np.float64)
    ls = np.maximum(np.asarray([r["ls_mse_mean"] for r in rows]), 1e-12)
    dnn = np.maximum(np.asarray([r["ls_dnn_mse_mean"] for r in rows]), 1e-12)
    ls_std = np.asarray([r["ls_mse_std"] for r in rows])
    dnn_std = np.asarray([r["ls_dnn_mse_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(snr, ls, "o-", color=METHOD_STYLE["ls"]["color"], label="LS", linewidth=2)
    ax.fill_between(snr, np.maximum(ls - ls_std, 1e-12), ls + ls_std, color=METHOD_STYLE["ls"]["color"], alpha=0.2)
    ax.plot(snr, dnn, "D-", color=METHOD_STYLE["ls_dnn"]["color"], label="LS+DNN", linewidth=2)
    ax.fill_between(snr, np.maximum(dnn - dnn_std, 1e-12), dnn + dnn_std, color=METHOD_STYLE["ls_dnn"]["color"], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_title(f"High-SNR error floor — {channel.upper()} (15–30 dB, mean ± std)")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, stem)


def plot_training_curve(channel: str, stem: str) -> dict[str, str]:
    history = _load_json(REVISION_ROOT / "training" / f"history_{channel}_seed{REPRESENTATIVE_SEED}.json")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="Training MSE loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation MSE loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"DNN training — {channel.upper()} (seed={REPRESENTATIVE_SEED})")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    meta = {
        "channel": channel,
        "seed": REPRESENTATIVE_SEED,
        "epochs_run": history.get("epochs_run"),
        "early_stopping_epoch": history.get("early_stopping_epoch"),
        "note": "Representative single-seed curve (not averaged across seeds).",
    }
    (FINAL_FIGURES / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _save_figure(fig, stem)


def _predict_dnn_for_sample(channel: str, seed: int, sample_index: int) -> np.ndarray:
    raw_path = REVISION_ROOT / "raw" / f"{channel}_ns100_seed{seed}.npz"
    processed_path = REVISION_ROOT / "processed" / f"{channel}_ns100_seed{seed}_residual_processed.npz"
    model_path = REVISION_ROOT / "models" / f"{channel}_ns100_seed{seed}_residual_processed_dnn.keras"

    raw = np.load(raw_path, allow_pickle=False)
    processed = dict(np.load(processed_path, allow_pickle=False))
    ls = np.asarray(raw["ls_estimate"][sample_index], dtype=np.complex128)

    feat = np.concatenate([np.real(ls), np.imag(ls)]).astype(np.float64)
    x_mean = np.asarray(processed["x_mean"], dtype=np.float64)
    x_std = np.asarray(processed["x_std"], dtype=np.float64)
    y_mean = np.asarray(processed["y_mean"], dtype=np.float64)
    y_std = np.asarray(processed["y_std"], dtype=np.float64)

    x_norm = ((feat - x_mean) / x_std).astype(np.float32)[None, :]
    model = keras.models.load_model(model_path)
    y_pred = np.asarray(model.predict(x_norm, verbose=0), dtype=np.float64)[0]
    y_den = y_pred * y_std + y_mean
    half = feat.size // 2
    residual = y_den[:half] + 1j * y_den[half:]
    return ls + residual


def plot_sample_channel_figure(channel: str, stem: str) -> dict[str, str]:
    meta_path = REVISION_ROOT / "metadata" / f"sample_channel_{channel}_seed{REPRESENTATIVE_SEED}.json"
    meta = _load_json(meta_path)
    sample_index = int(meta["sample_index"])
    snr_db = float(meta["snr_db"])

    raw_path = Path(meta["raw_path"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    raw = np.load(raw_path, allow_pickle=False)

    true_h = np.asarray(raw["true_channel"][sample_index], dtype=np.complex128)
    ls_h = np.asarray(raw["ls_estimate"][sample_index], dtype=np.complex128)
    simp_h = np.asarray(raw["simplified_mmse_estimate"][sample_index], dtype=np.complex128)
    lmmse_h = np.asarray(raw["lmmse_flat_estimate"][sample_index], dtype=np.complex128)
    dnn_h = _predict_dnn_for_sample(channel, REPRESENTATIVE_SEED, sample_index)

    k = np.arange(true_h.size)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, np.abs(true_h), "-", linewidth=2.2, label="True Channel", color="black")
    ax.plot(k, np.abs(ls_h), "--", linewidth=1.6, label="LS", color=METHOD_STYLE["ls"]["color"])
    ax.plot(k, np.abs(simp_h), "--", linewidth=1.6, label="Simplified-MMSE", color=METHOD_STYLE["simplified_mmse"]["color"])
    ax.plot(k, np.abs(lmmse_h), "--", linewidth=1.6, label="LMMSE-flat", color=METHOD_STYLE["lmmse_flat"]["color"])
    ax.plot(k, np.abs(dnn_h), "--", linewidth=1.6, label="LS+DNN", color=METHOD_STYLE["ls_dnn"]["color"])
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("|H|")
    ax.set_title(
        f"Sample channel estimate — {channel.upper()} "
        f"(SNR={snr_db:.0f} dB, seed={REPRESENTATIVE_SEED}, flat fading)"
    )
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    caption_meta = {
        **meta,
        "sample_index": sample_index,
        "flat_fading_note": "Single-tap flat fading: |H| is constant across subcarriers.",
        "estimators_shown": ["True Channel", "LS", "Simplified-MMSE", "LMMSE-flat", "LS+DNN"],
    }
    (FINAL_FIGURES / f"{stem}_metadata.json").write_text(json.dumps(caption_meta, indent=2), encoding="utf-8")
    return _save_figure(fig, stem)


def plot_inference_time(stem: str) -> dict[str, str]:
    estimators = ["ls", "simplified_mmse", "lmmse_flat", "ls_dnn"]
    labels = [METHOD_STYLE[e]["label"] for e in estimators]
    means, stds = [], []
    for est in estimators:
        vals = []
        for channel in CHANNELS:
            for seed in SEEDS:
                path = REVISION_ROOT / "complexity" / f"complexity_{channel}_seed{seed}.json"
                data = _load_json(path)
                vals.append(float(data["estimators"][est]["mean_seconds"]))
        arr = np.asarray(vals, dtype=np.float64)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[METHOD_STYLE[e]["color"] for e in estimators], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean inference time (s)")
    ax.set_title(f"Inference time comparison (mean over {len(CHANNELS)*len(SEEDS)} runs)")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_table_1() -> list[dict[str, str]]:
    params = OFDMParams()
    pilot_indices = generate_pilot_indices(params.n_subcarriers, params.pilot_spacing)
    split = _load_json(REVISION_ROOT / "metadata" / "split_awgn_seed42.json")
    snr_levels = sorted(float(k) for k in split["samples_per_snr"])
    rows = [
        {"Parameter": "Subcarriers (N)", "Value": str(params.n_subcarriers)},
        {"Parameter": "Cyclic prefix length", "Value": str(params.cp_len)},
        {"Parameter": "Modulation", "Value": "QPSK"},
        {"Parameter": "Pilot spacing", "Value": str(params.pilot_spacing)},
        {"Parameter": "Pilot indices", "Value": str(pilot_indices.tolist())},
        {"Parameter": "Number of pilots", "Value": str(pilot_indices.size)},
        {"Parameter": "Data subcarriers", "Value": str(params.n_subcarriers - pilot_indices.size)},
        {"Parameter": "Pilot symbol", "Value": "1+0j"},
        {"Parameter": "SNR range (dB)", "Value": str(snr_levels)},
        {"Parameter": "Samples per SNR", "Value": "100"},
        {"Parameter": "Train / Val / Test per SNR", "Value": "72 / 8 / 20"},
        {"Parameter": "Random seeds", "Value": str(SEEDS)},
        {"Parameter": "Rician K-factor", "Value": "6 dB"},
        {"Parameter": "Channel model", "Value": "Single-tap flat fading (AWGN / Rayleigh / Rician)"},
        {"Parameter": "Split strategy", "Value": split["split_strategy"]},
    ]
    return rows


def build_table_2() -> list[dict[str, str]]:
    complexity = _load_json(REVISION_ROOT / "complexity" / "complexity_awgn_seed42.json")
    history = _load_json(REVISION_ROOT / "training" / "history_awgn_seed42.json")
    params = complexity.get("ls_dnn_model", {})
    return [
        {"Parameter": "Input dimension", "Value": "128 (64 Re + 64 Im LS features)"},
        {"Parameter": "Hidden layer 1", "Value": "Dense(256, ReLU)"},
        {"Parameter": "Hidden layer 2", "Value": "Dense(256, ReLU)"},
        {"Parameter": "Output dimension", "Value": "128 (residual Re/Im)"},
        {"Parameter": "Learning mode", "Value": "Residual (predict true - LS)"},
        {"Parameter": "Optimizer", "Value": "Adam"},
        {"Parameter": "Learning rate", "Value": "1e-3"},
        {"Parameter": "Batch size", "Value": "64"},
        {"Parameter": "Maximum epochs", "Value": str(history.get("epochs_requested", 60))},
        {"Parameter": "Early stopping patience", "Value": "8 (monitor val_loss)"},
        {"Parameter": "Trainable parameters", "Value": str(params.get("trainable_params", "N/A"))},
        {"Parameter": "Total parameters", "Value": str(params.get("total_params", "N/A"))},
    ]


def build_mse_nmse_table(channel: str, table_num: int) -> tuple[list[dict], list[dict]]:
    data = _load_json(REVISION_ROOT / "mse_nmse" / f"aggregate_{channel}.json")
    rows: list[dict[str, Any]] = []
    for r in data["per_snr"]:
        row = {"SNR (dB)": r["snr_db"]}
        for prefix, label in [
            ("ls", "LS"),
            ("simplified_mmse", "Simplified-MMSE"),
            ("lmmse_flat", "LMMSE-flat"),
            ("ls_dnn", "LS+DNN"),
        ]:
            row[f"{label} MSE"] = f"{r[f'{prefix}_mse_mean']:.6g} ± {r[f'{prefix}_mse_std']:.2g}"
            row[f"{label} NMSE"] = f"{r[f'{prefix}_nmse_mean']:.6g} ± {r[f'{prefix}_nmse_std']:.2g}"
        rows.append(row)
    return rows, rows


def build_ber_table(channel: str) -> list[dict[str, Any]]:
    agg = _load_json(REVISION_ROOT / "ber" / f"aggregate_{channel}.json")
    seed0 = _load_json(REVISION_ROOT / "ber" / f"ber_{channel}_seed{SEEDS[0]}.json")
    rows: list[dict[str, Any]] = []
    for r in agg["per_snr"]:
        snr_key = str(float(r["snr_db"]))
        total_bits = int(seed0["per_snr"][snr_key]["ls"]["total_bits"])
        row: dict[str, Any] = {"SNR (dB)": r["snr_db"], "Total transmitted bits (per seed test set)": total_bits}
        for prefix, label in [
            ("ls", "LS"),
            ("simplified_mmse", "Simplified-MMSE"),
            ("lmmse_flat", "LMMSE-flat"),
            ("ls_dnn", "LS+DNN"),
        ]:
            row[f"{label} BER mean"] = f"{r[f'{prefix}_ber_mean']:.6g}"
            row[f"{label} BER std"] = f"{r[f'{prefix}_ber_std']:.6g}"
        rows.append(row)
    return rows


def build_table_9() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        data = _load_json(REVISION_ROOT / "snr_gain" / f"snr_gain_{channel}.json")
        for key, entry in data["snr_gain"].items():
            rows.append(
                {
                    "Channel": channel,
                    "Target BER": entry["target_ber"],
                    "Status": entry["status"],
                    "SNR_LS (dB)": entry["snr_ls_db"],
                    "SNR_LS+DNN (dB)": entry["snr_dnn_db"],
                    "SNR gain = SNR_LS - SNR_DNN (dB)": entry["snr_gain_db"],
                }
            )
    return rows


def build_table_10() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in ("rayleigh", "rician"):
        ls_vals: dict[float, list[float]] = {20.0: [], 25.0: [], 30.0: []}
        dnn_vals: dict[float, list[float]] = {20.0: [], 25.0: [], 30.0: []}
        for seed in SEEDS:
            comp = _load_json(REVISION_ROOT / "mse_nmse" / f"comparison_{channel}_seed{seed}.json")
            for row in comp["per_snr"]:
                snr = float(row["snr_db"])
                if snr in ls_vals:
                    ls_vals[snr].append(float(row["ls_mse"]))
                    dnn_vals[snr].append(float(row["ls_dnn_mse"]))
        for snr in (20.0, 25.0, 30.0):
            ls_mean = float(np.mean(ls_vals[snr]))
            dnn_mean = float(np.mean(dnn_vals[snr]))
            ratio = dnn_mean / ls_mean if ls_mean > 0 else float("nan")
            rows.append(
                {
                    "Channel": channel,
                    "SNR (dB)": snr,
                    "LS MSE (mean over 3 seeds)": f"{ls_mean:.6g}",
                    "LS+DNN MSE (mean over 3 seeds)": f"{dnn_mean:.6g}",
                    "Ratio (DNN/LS)": f"{ratio:.4f}",
                    "LS+DNN worse than LS?": "yes" if dnn_mean > ls_mean else "no",
                }
            )
    return rows


def build_table_11() -> list[dict[str, Any]]:
    estimators = ["ls", "simplified_mmse", "lmmse_flat", "ls_dnn"]
    env = _load_json(REVISION_ROOT / "complexity" / "complexity_awgn_seed42.json")["environment"]
    rows: list[dict[str, Any]] = []
    for est in estimators:
        times = []
        for channel in CHANNELS:
            for seed in SEEDS:
                data = _load_json(REVISION_ROOT / "complexity" / f"complexity_{channel}_seed{seed}.json")
                times.append(float(data["estimators"][est]["mean_seconds"]))
        stds = []
        for channel in CHANNELS:
            for seed in SEEDS:
                data = _load_json(REVISION_ROOT / "complexity" / f"complexity_{channel}_seed{seed}.json")
                stds.append(float(data["estimators"][est]["std_seconds"]))
        row = {
            "Estimator": METHOD_STYLE[est]["label"],
            "Mean inference time (s)": f"{np.mean(times):.6g}",
            "Std inference time (s)": f"{np.mean(stds):.6g}",
            "Trainable parameters": "—",
            "Measurement environment": f"{env['platform']}, Python {env['python_version']}, {env['processor']}",
        }
        if est == "ls_dnn":
            params = _load_json(REVISION_ROOT / "complexity" / "complexity_awgn_seed42.json")["ls_dnn_model"]
            row["Trainable parameters"] = str(params["trainable_params"])
        rows.append(row)
    return rows


def build_all_tables() -> dict[str, Any]:
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Any] = {}

    t1 = build_table_1()
    _write_csv(FINAL_TABLES / "table01_system_parameters.csv", t1)
    _write_md_table(FINAL_TABLES / "table01_system_parameters.md", "Table 1 — System and Simulation Parameters", t1)
    generated["table01"] = t1

    t2 = build_table_2()
    _write_csv(FINAL_TABLES / "table02_dnn_configuration.csv", t2)
    _write_md_table(FINAL_TABLES / "table02_dnn_configuration.md", "Table 2 — DNN Configuration", t2)
    generated["table02"] = t2

    for i, channel in enumerate(CHANNELS, start=3):
        rows, _ = build_mse_nmse_table(channel, i)
        stem = f"table{i:02d}_mse_nmse_{channel}"
        _write_csv(FINAL_TABLES / f"{stem}.csv", rows)
        _write_md_table(FINAL_TABLES / f"{stem}.md", f"Table {i} — MSE/NMSE by SNR ({channel.upper()})", rows)
        generated[stem] = rows

    for i, channel in enumerate(CHANNELS, start=6):
        rows = build_ber_table(channel)
        stem = f"table{i:02d}_ber_{channel}"
        _write_csv(FINAL_TABLES / f"{stem}.csv", rows)
        _write_md_table(FINAL_TABLES / f"{stem}.md", f"Table {i} — BER by SNR ({channel.upper()})", rows)
        generated[stem] = rows

    t9 = build_table_9()
    _write_csv(FINAL_TABLES / "table09_target_ber_snr_gains.csv", t9)
    _write_md_table(FINAL_TABLES / "table09_target_ber_snr_gains.md", "Table 9 — Target BER SNR Gains", t9)
    generated["table09"] = t9

    t10 = build_table_10()
    _write_csv(FINAL_TABLES / "table10_high_snr_behavior.csv", t10)
    _write_md_table(FINAL_TABLES / "table10_high_snr_behavior.md", "Table 10 — High-SNR Behavior", t10)
    generated["table10"] = t10

    t11 = build_table_11()
    _write_csv(FINAL_TABLES / "table11_computational_complexity.csv", t11)
    _write_md_table(FINAL_TABLES / "table11_computational_complexity.md", "Table 11 — Computational Complexity", t11)
    generated["table11"] = t11

    return generated


def build_all_figures() -> dict[str, Any]:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Any] = {}

    for channel in CHANNELS:
        figs[f"fig_mse_{channel}"] = plot_metric_vs_snr(channel, "mse", MSE_METRIC_MAP, f"fig_mse_{channel}")
        figs[f"fig_nmse_{channel}"] = plot_metric_vs_snr(channel, "nmse", NMSE_METRIC_MAP, f"fig_nmse_{channel}")
        figs[f"fig_ber_{channel}"] = plot_metric_vs_snr(channel, "ber", BER_METRIC_MAP, f"fig_ber_{channel}")
        figs[f"fig_high_snr_{channel}"] = plot_high_snr_error_floor(channel, f"fig_high_snr_{channel}")
        figs[f"fig_training_{channel}"] = plot_training_curve(channel, f"fig_training_{channel}")
        figs[f"fig_sample_{channel}"] = plot_sample_channel_figure(channel, f"fig_sample_{channel}")

    figs["fig_inference_time"] = plot_inference_time("fig_inference_time")
    return figs


def write_final_report(moved_files: list[str], figures: dict[str, Any], tables: dict[str, Any]) -> None:
    report = REVISION_ROOT / "FINAL_FIGURE_TABLE_REPORT.md"
    lines = [
        "# Final Figure and Table Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Archived old figures/tables",
        "",
        f"- **Location:** `data/results/archive_pre_revision/`",
        f"- **Files moved:** {len(moved_files)} (see `archive_pre_revision/ARCHIVE_LOG.md`)",
        "- Includes legacy MSE/NMSE/BER plots, improvement-percentage figures, seed bar charts,",
        "  old comparison CSV/JSON/NPZ, pre-revision training logs/models, and methodology diagrams.",
        "",
        "## New figures generated",
        "",
        "| Figure | PNG/PDF | Source data |",
        "| --- | --- | --- |",
    ]
    fig_sources = {
        "fig_mse_*": "revision/mse_nmse/aggregate_{channel}.json",
        "fig_nmse_*": "revision/mse_nmse/aggregate_{channel}.json",
        "fig_ber_*": "revision/ber/aggregate_{channel}.json",
        "fig_high_snr_*": "revision/mse_nmse/aggregate_{channel}.json (SNR ≥ 15 dB)",
        "fig_training_*": "revision/training/history_{channel}_seed42.json",
        "fig_sample_*": "revision/raw/*.npz + revision/models/*.keras + metadata/sample_channel_*.json",
        "fig_inference_time": "revision/complexity/complexity_*_seed*.json",
    }
    for name, paths in figures.items():
        src = fig_sources.get(name.split("_")[0] + "_*", "see above")
        if "mse" in name:
            src = f"revision/mse_nmse/aggregate_{name.split('_')[-1]}.json"
        elif "nmse" in name:
            src = f"revision/mse_nmse/aggregate_{name.split('_')[-1]}.json"
        elif "ber" in name and "high" not in name:
            src = f"revision/ber/aggregate_{name.split('_')[-1]}.json"
        elif "high_snr" in name:
            src = f"revision/mse_nmse/aggregate_{name.split('_')[-1]}.json"
        elif "training" in name:
            src = f"revision/training/history_{name.split('_')[-1]}_seed42.json"
        elif "sample" in name:
            src = f"revision/raw/{name.split('_')[-1]}_ns100_seed42.npz + model + metadata"
        elif "inference" in name:
            src = "revision/complexity/complexity_*_seed*.json"
        lines.append(f"| `{name}` | `{paths.get('png', '')}` | `{src}` |")

    lines.extend(
        [
            "",
            "## New tables generated",
            "",
            "| Table | Files | Source |",
            "| --- | --- | --- |",
            "| Table 1 — System parameters | `final_tables/table01_*` | Code (`OFDMParams`, pilots) + `metadata/split_awgn_seed42.json` |",
            "| Table 2 — DNN config | `final_tables/table02_*` | `model.py`, `training/history_awgn_seed42.json`, `complexity/complexity_awgn_seed42.json` |",
            "| Tables 3–5 — MSE/NMSE | `final_tables/table03–05_*` | `mse_nmse/aggregate_{awgn,rayleigh,rician}.json` |",
            "| Tables 6–8 — BER | `final_tables/table06–08_*` | `ber/aggregate_*.json` + `ber/ber_*_seed42.json` (total bits) |",
            "| Table 9 — SNR gains | `final_tables/table09_*` | `snr_gain/snr_gain_*.json` |",
            "| Table 10 — High-SNR | `final_tables/table10_*` | `mse_nmse/comparison_*_seed{42,123,999}.json` |",
            "| Table 11 — Complexity | `final_tables/table11_*` | `complexity/complexity_*_seed*.json` |",
            "",
            "## Zero-BER plotting treatment",
            "",
            "- Raw BER values in tables/JSON are unchanged (including exact zeros).",
            "- Log-scale BER figures apply `ber_for_log_plot()` with floor ε=1e-6 **for visualization only**.",
            "",
            "## Missing data",
            "",
            "- Target BER = 1e-3 SNR gain: **not estimable within simulated SNR range** (all channels).",
            "- Training curves: representative **seed=42** only (not seed-averaged).",
            "",
            "## Recommended manuscript figure order",
            "",
            "1. System diagram (from archive: `archive_pre_revision/figures/ofdm_system_diagram.png`)",
            "2. DNN architecture (from archive: `archive_pre_revision/figures/dnn_diagram.png`)",
            "3. `fig_sample_awgn`, `fig_sample_rayleigh`, `fig_sample_rician`",
            "4. `fig_mse_{awgn,rayleigh,rician}`",
            "5. `fig_nmse_{awgn,rayleigh,rician}`",
            "6. `fig_ber_{awgn,rayleigh,rician}`",
            "7. `fig_high_snr_rayleigh`, `fig_high_snr_rician`",
            "8. `fig_training_{awgn,rayleigh,rician}`",
            "9. `fig_inference_time` (optional; Table 11 is primary)",
            "",
            "## Replacement mapping (old → new)",
            "",
            "| Old (archived) | New (final) |",
            "| --- | --- |",
            "| `figures/snr_vs_mse*.png` (linear/old) | `final_figures/fig_mse_*.png` |",
            "| `figures/snr_vs_nmse*.png` | `final_figures/fig_nmse_*.png` |",
            "| `figures/snr_vs_ber*.png` | `final_figures/fig_ber_*.png` |",
            "| `figures/improvement_percentage_*.png` | **Removed** — not included in final set |",
            "| `figures/awgn_seed_mean_std_bar.png` | **Removed** — replaced by multi-seed curves |",
            "| `figures/example_channel_estimate_*.png` | `final_figures/fig_sample_*.png` |",
            "| `figures/training_vs_validation_loss_*.png` | `final_figures/fig_training_*.png` |",
            "| `metrics/comparison_per_snr*.csv/json` | `final_tables/table03–08_*.csv` |",
            "| `metrics/improvement_percentage_*.csv` | **Removed** |",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Archiving pre-revision outputs...")
    moved = archive_pre_revision_outputs()
    write_archive_log(moved)
    print(f"Archived {len(moved)} files.")

    print("Building final figures...")
    figures = build_all_figures()
    print(f"Generated {len(figures)} figure sets.")

    print("Building final tables...")
    tables = build_all_tables()
    print(f"Generated {len(tables)} tables.")

    write_final_report(moved, figures, tables)
    print(f"Done. Final outputs: {FINAL_FIGURES} and {FINAL_TABLES}")


if __name__ == "__main__":
    main()
