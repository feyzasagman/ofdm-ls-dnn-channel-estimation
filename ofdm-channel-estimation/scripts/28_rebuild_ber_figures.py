"""Rebuild BER figures (4 estimators incl. LMMSE-flat) from revision aggregates — no retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.split import DEFAULT_SEEDS
from src.revision.ber_figure import ber_figure_note, plot_ber_curves_on_axes, style_ber_axes

REVISION_ROOT = PROJECT_ROOT / "data" / "results" / "revision"
FINAL_FIGURES = REVISION_ROOT / "final_figures"
WORD_READY_FIGURES = FINAL_FIGURES / "word_ready"
FINAL_TABLES = REVISION_ROOT / "final_tables"

SEEDS = list(DEFAULT_SEEDS)

METHOD_ORDER = ("ls", "simplified_mmse", "lmmse_flat", "ls_dnn")
METHOD_STYLE = {
    "ls": {"label": "LS", "color": "#1f77b4", "marker": "o"},
    "simplified_mmse": {"label": "Simplified-MMSE", "color": "#ff7f0e", "marker": "s"},
    "lmmse_flat": {"label": "LMMSE-flat", "color": "#2ca02c", "marker": "^"},
    "ls_dnn": {"label": "LS+DNN", "color": "#d62728", "marker": "D"},
}

CHANNEL_FIGURES = [
    ("awgn", "fig_ber_awgn", "figure_3_11_ber_awgn", "Figure 3.11"),
    ("rayleigh", "fig_ber_rayleigh", "figure_3_12_ber_rayleigh", "Figure 3.12"),
    ("rician", "fig_ber_rician", "figure_3_13_ber_rician", "Figure 3.13"),
]

MANUSCRIPT_CAPTIONS: dict[str, str] = {
    "figure_3_11_ber_awgn": (
        "Figure 3.11. BER versus SNR under the AWGN scenario (mean ± standard deviation "
        "over three independent repetitions). Zero BER values, corresponding to no observed "
        "bit errors in the finite test set, are omitted from the logarithmic plot."
    ),
    "figure_3_12_ber_rayleigh": (
        "Figure 3.12. BER versus SNR under the single-tap Rayleigh flat-fading scenario "
        "(mean ± standard deviation over three independent repetitions). Zero BER values, "
        "corresponding to no observed bit errors in the finite test set, are omitted from "
        "the logarithmic plot."
    ),
    "figure_3_13_ber_rician": (
        "Figure 3.13. BER versus SNR under the single-tap Rician flat-fading scenario "
        "(mean ± standard deviation over three independent repetitions). Zero BER values, "
        "corresponding to no observed bit errors in the finite test set, are omitted from "
        "the logarithmic plot."
    ),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_ber_sources() -> None:
    """Ensure per-seed BER JSON includes lmmse_flat for all channel/seed combos."""
    missing: list[str] = []
    for channel in ("awgn", "rayleigh", "rician"):
        for seed in SEEDS:
            path = REVISION_ROOT / "ber" / f"ber_{channel}_seed{seed}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            data = _load_json(path)
            methods = data.get("methods", [])
            if "lmmse_flat" not in methods:
                missing.append(f"{path} (no lmmse_flat in methods)")
            per = data["per_snr"]
            sample_key = next(iter(per))
            if "lmmse_flat" not in per[sample_key]:
                missing.append(f"{path} (no lmmse_flat per_snr)")
    if missing:
        raise RuntimeError("BER source verification failed:\n" + "\n".join(missing))
    print("BER source verification: OK (9 per-seed files, lmmse_flat present)")


def plot_ber_figure(channel: str) -> plt.Figure:
    import numpy as np

    agg = _load_json(REVISION_ROOT / "ber" / f"aggregate_{channel}.json")
    rows = agg["per_snr"]
    snr = np.asarray([r["snr_db"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    plot_ber_curves_on_axes(ax, snr, rows, METHOD_ORDER, METHOD_STYLE)
    style_ber_axes(ax)
    ax.set_title(f"BER vs SNR — {channel.upper()} (mean ± std, seeds {list(SEEDS)})")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def save_png_word_safe(fig: plt.Figure, path: Path) -> None:
    """Write a standard RGB PNG suitable for Word and other strict viewers."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    with Image.open(path) as im:
        im.load()
        im.convert("RGB").save(path, format="PNG", optimize=False)


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        out = FINAL_FIGURES / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            save_png_word_safe(fig, out)
        else:
            fig.savefig(out, **kwargs)
        paths[ext] = str(out)
    return paths


def save_manuscript_word_ready(fig: plt.Figure, manuscript_stem: str) -> dict[str, str]:
    """Write manuscript PNG (300 dpi, RGB) and PDF under word_ready/."""
    WORD_READY_FIGURES.mkdir(parents=True, exist_ok=True)
    png_out = WORD_READY_FIGURES / f"{manuscript_stem}.png"
    pdf_out = WORD_READY_FIGURES / f"{manuscript_stem}.pdf"
    save_png_word_safe(fig, png_out)
    fig.savefig(
        pdf_out,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    return {"png": str(png_out.resolve()), "pdf": str(pdf_out.resolve())}


def build_ber_summary_table(channel: str) -> list[dict[str, str]]:
    agg = _load_json(REVISION_ROOT / "ber" / f"aggregate_{channel}.json")
    table: list[dict[str, str]] = []
    for row in agg["per_snr"]:
        entry = {"SNR (dB)": str(row["snr_db"])}
        for key in METHOD_ORDER:
            entry[METHOD_STYLE[key]["label"] + " BER"] = f"{row[f'{key}_ber_mean']:.6g}"
        table.append(entry)
    return table


def write_markdown_table(channel: str, rows: list[dict[str, str]]) -> Path:
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    path = FINAL_TABLES / f"ber_summary_{channel}_four_estimators.md"
    headers = list(rows[0].keys())
    lines = [f"# BER summary ({channel.upper()}) — four estimators", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    verify_ber_sources()

    outputs: dict[str, object] = {"figures": {}, "tables": {}}

    note = ber_figure_note()

    for channel, legacy_stem, manuscript_stem, fig_id in CHANNEL_FIGURES:
        fig = plot_ber_figure(channel)
        for stem in (legacy_stem, manuscript_stem):
            paths = save_figure(fig, stem)
            outputs["figures"][stem] = paths
        word_paths = save_manuscript_word_ready(fig, manuscript_stem)
        outputs["figures"][f"{manuscript_stem}_word_ready"] = word_paths
        plt.close(fig)

        rows = build_ber_summary_table(channel)
        table_path = write_markdown_table(channel, rows)
        outputs["tables"][channel] = str(table_path)

        caption = MANUSCRIPT_CAPTIONS[manuscript_stem]
        (FINAL_FIGURES / f"{manuscript_stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")
        (WORD_READY_FIGURES / f"{manuscript_stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    meta_path = FINAL_FIGURES / "ber_figures_rebuild_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "note": "Rebuilt from existing revision/ber/aggregate_*.json; no retraining.",
                "seeds": SEEDS,
                "methods": list(METHOD_ORDER),
                "zero_ber_plotting": "omit zero mean BER from log-scale curves",
                "caption_note": note,
                "snr_gain_unchanged": "snr_gain/*.json not modified",
                "stored_ber_unchanged": "aggregate and per-seed JSON retain exact zeros",
                **outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Rebuilt BER figures (positive BER only on log scale):")
    for channel, legacy_stem, manuscript_stem, _ in CHANNEL_FIGURES:
        print(f"  {channel}: {legacy_stem}.*, {manuscript_stem}.*")


if __name__ == "__main__":
    main()
