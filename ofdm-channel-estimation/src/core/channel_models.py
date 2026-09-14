"""Canonical channel model names and metadata."""

from __future__ import annotations

CHANNEL_MODEL_NAMES: dict[str, str] = {
    "awgn": "AWGN (deterministic h=1, no fading)",
    "rayleigh": "single-tap Rayleigh flat fading",
    "rician": "single-tap Rician flat fading",
}


def channel_metadata(
    channel_type: str,
    *,
    k_factor_db: float = 6.0,
    flat_fading: bool = True,
) -> dict[str, object]:
    """Build channel metadata for experiment outputs."""
    key = channel_type.strip().lower()
    if key not in CHANNEL_MODEL_NAMES:
        raise ValueError(f"Unknown channel_type: {channel_type}")
    meta: dict[str, object] = {
        "channel_type": key,
        "channel_model_name": CHANNEL_MODEL_NAMES[key],
        "flat_fading": flat_fading,
        "frequency_selective": False,
    }
    if key == "rician":
        meta["k_factor_db"] = float(k_factor_db)
        meta["rician_note"] = (
            "h = h_los + h_scatter with K = P_los/P_scatter; "
            "LoS amplitude sqrt(K/(K+1)), scatter std sqrt(1/(2(K+1)))."
        )
    return meta
