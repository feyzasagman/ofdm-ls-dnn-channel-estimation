"""Factory utilities for selecting channel models by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .awgn import apply_awgn
from .rayleigh import apply_rayleigh_flat_fading
from .rician import apply_rician_flat_fading


ChannelCallable = Callable[..., Any]


def get_channel(channel_type: str) -> ChannelCallable:
    """Return channel function based on channel type string.

    Supported values:
        - "awgn"
        - "rayleigh"
        - "rician"
    """
    normalized = channel_type.strip().lower()
    mapping: dict[str, ChannelCallable] = {
        "awgn": apply_awgn,
        "rayleigh": apply_rayleigh_flat_fading,
        "rician": apply_rician_flat_fading,
    }

    if normalized not in mapping:
        supported = ", ".join(mapping.keys())
        raise ValueError(f"Unsupported channel type '{channel_type}'. Supported: {supported}.")

    return mapping[normalized]
