"""Generate raw and processed datasets for the first experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ofdm_params import OFDMParams
from src.dataset.generate_dataset import generate_raw_dataset
from src.dataset.preprocess import preprocess_for_dnn


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data generation."""
    parser = argparse.ArgumentParser(description="Generate raw+processed OFDM datasets.")
    parser.add_argument("--channel-type", type=str, default="rayleigh", choices=["awgn", "rayleigh", "rician"])
    parser.add_argument("--samples-per-snr", type=int, default=100)
    parser.add_argument("--residual-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--k-factor-db", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    """Run small-scale raw generation then preprocessing."""
    args = parse_args()
    snr_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    params = OFDMParams()

    raw_path = generate_raw_dataset(
        channel_type=args.channel_type,
        snr_db_list=snr_list,
        n_samples_per_snr=args.samples_per_snr,
        ofdm_params=params,
        random_seed=args.seed,
        k_factor_db=args.k_factor_db,
    )
    processed_path = preprocess_for_dnn(
        raw_dataset_path=raw_path,
        residual_mode=args.residual_mode,
        normalize=args.normalize,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )

    print("Dataset generation completed.")
    print(f"Raw dataset path      : {raw_path}")
    print(f"Processed dataset path: {processed_path}")
    print(f"Channel type          : {args.channel_type}")
    print(f"SNR list              : {snr_list}")
    print(f"Samples per SNR       : {args.samples_per_snr}")
    print(f"Residual mode         : {args.residual_mode}")


if __name__ == "__main__":
    main()
