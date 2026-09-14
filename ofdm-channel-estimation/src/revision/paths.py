"""Revision experiment output paths."""

from __future__ import annotations

from pathlib import Path


def revision_root(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "data" / "results" / "revision"


def ensure_revision_dirs(project_root: Path | None = None) -> dict[str, Path]:
    base = revision_root(project_root)
    dirs = {
        "root": base,
        "mse_nmse": base / "mse_nmse",
        "ber": base / "ber",
        "training": base / "training",
        "complexity": base / "complexity",
        "figures": base / "figures",
        "metadata": base / "metadata",
        "high_snr": base / "high_snr",
        "snr_gain": base / "snr_gain",
        "raw": base / "raw",
        "processed": base / "processed",
        "models": base / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
