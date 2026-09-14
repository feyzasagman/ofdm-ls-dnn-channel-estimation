"""Export manuscript BER figures (PNG + PDF) under final_figures/word_ready/."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "rebuild_ber",
    PROJECT_ROOT / "scripts" / "28_rebuild_ber_figures.py",
)
_rebuild = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rebuild)

WORD_READY = _rebuild.WORD_READY_FIGURES

EXPORTS = [
    ("awgn", "figure_3_11_ber_awgn"),
    ("rayleigh", "figure_3_12_ber_rayleigh"),
    ("rician", "figure_3_13_ber_rician"),
]


def channel_fingerprint(channel: str) -> str:
    agg = json.loads(
        (_rebuild.REVISION_ROOT / "ber" / f"aggregate_{channel}.json").read_text(encoding="utf-8")
    )
    parts = []
    for row in agg["per_snr"]:
        parts.append(
            f"{row['snr_db']}:"
            f"{row['ls_ber_mean']:.8g},"
            f"{row['ls_dnn_ber_mean']:.8g}"
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def verify_png(path: Path) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.load()
        if im.format != "PNG":
            raise ValueError(f"{path}: expected PNG, got {im.format}")
        return "OK", im.size


def assert_legend_and_axes(fig: plt.Figure, stem: str) -> None:
    ax = fig.axes[0]
    legend = ax.get_legend()
    if legend is None:
        raise RuntimeError(f"{stem}: missing legend")
    labels = [t.get_text() for t in legend.get_texts()]
    expected = [_rebuild.METHOD_STYLE[k]["label"] for k in _rebuild.METHOD_ORDER]
    if labels != expected:
        raise RuntimeError(f"{stem}: legend {labels} != {expected}")
    from src.revision.ber_figure import BER_SNR_XMARGIN, BER_SNR_XMAX, BER_SNR_XMIN, BER_SNR_XTICKS

    expected_xlim = (BER_SNR_XMIN - BER_SNR_XMARGIN, BER_SNR_XMAX + BER_SNR_XMARGIN)
    if ax.get_xlim() != expected_xlim:
        raise RuntimeError(f"{stem}: xlim {ax.get_xlim()} != {expected_xlim}")
    if list(ax.get_xticks()) != list(BER_SNR_XTICKS):
        raise RuntimeError(f"{stem}: xticks {list(ax.get_xticks())} != {list(BER_SNR_XTICKS)}")


def main() -> None:
    _rebuild.verify_ber_sources()
    WORD_READY.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    fingerprints: dict[str, str] = {}

    for channel, stem in EXPORTS:
        fig = _rebuild.plot_ber_figure(channel)
        assert_legend_and_axes(fig, stem)
        paths = _rebuild.save_manuscript_word_ready(fig, stem)
        plt.close(fig)

        png_path = Path(paths["png"])
        fingerprints[channel] = channel_fingerprint(channel)
        verify_result, dimensions = verify_png(png_path)
        size_bytes = png_path.stat().st_size

        caption = _rebuild.MANUSCRIPT_CAPTIONS[stem]
        (WORD_READY / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

        reports.append(
            {
                "path_png": str(png_path),
                "path_pdf": paths["pdf"],
                "channel": channel,
                "size_kb": round(size_bytes / 1024, 2),
                "width": dimensions[0],
                "height": dimensions[1],
                "verify": verify_result,
                "data_fingerprint": fingerprints[channel],
                "md5_png": hashlib.md5(png_path.read_bytes()).hexdigest(),
            }
        )

    if len({r["md5_png"] for r in reports}) != 3:
        raise RuntimeError("PNG files are duplicates (identical MD5).")
    if len(set(fingerprints.values())) != 3:
        raise RuntimeError("Channel aggregate fingerprints collide.")

    (WORD_READY / "export_verification.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    for r in reports:
        print(f"PNG: {r['path_png']}")
        print(f"PDF: {r['path_pdf']}")
        print(f"  channel: {r['channel']}")
        print(f"  size_kb: {r['size_kb']}")
        print(f"  dimensions: {r['width']} x {r['height']}")
        print(f"  verify: {r['verify']}")
        print(f"  md5_png: {r['md5_png']}")
        print()


if __name__ == "__main__":
    main()
