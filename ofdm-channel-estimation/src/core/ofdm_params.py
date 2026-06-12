"""Core OFDM parameter definitions."""

from dataclasses import dataclass


@dataclass
class OFDMParams:
    """Container for basic OFDM system parameters.

    Attributes:
        n_subcarriers: Number of OFDM subcarriers.
        cp_len: Cyclic prefix length in samples.
        pilot_spacing: Spacing between pilot subcarriers.
        modulation_order: Constellation order. For now, only QPSK (4).
        pilot_value: Complex value used for pilots.
        random_seed: Seed used for reproducible random generation.
    """

    n_subcarriers: int = 64
    cp_len: int = 16
    pilot_spacing: int = 4
    modulation_order: int = 4
    pilot_value: complex = 1.0 + 0.0j
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate parameter ranges for safe defaults and usage."""
        if self.n_subcarriers <= 0:
            raise ValueError("n_subcarriers must be positive.")
        if self.cp_len < 0:
            raise ValueError("cp_len must be non-negative.")
        if self.pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive.")
        if self.modulation_order != 4:
            raise ValueError("Only QPSK is supported for now (modulation_order=4).")
