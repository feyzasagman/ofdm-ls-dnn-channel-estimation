"""Regenerate only Figure 3.12 Rayleigh BER PNG/PDF (no data changes)."""

from __future__ import annotations

import hashlib
import importlib.util
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

STEM = "figure_3_12_ber_rayleigh"
CHANNEL = "rayleigh"


def main() -> None:
    _rebuild.verify_ber_sources()
    fig = _rebuild.plot_ber_figure(CHANNEL)
    paths = {}
    paths["final"] = _rebuild.save_figure(fig, STEM)["png"]
    paths["word_ready"] = _rebuild.save_manuscript_word_ready(fig, STEM)["png"]
    _rebuild.save_figure(fig, "fig_ber_rayleigh")
    plt.close(fig)

    for name, p in paths.items():
        path = Path(p)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            print(f"{name}: {path.resolve()}")
            print(f"  size_kb: {path.stat().st_size / 1024:.2f}")
            print(f"  dimensions: {im.size[0]} x {im.size[1]}")
            print(f"  verify: OK")
            print(f"  md5: {hashlib.md5(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
