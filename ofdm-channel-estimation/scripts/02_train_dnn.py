"""Train DNN regressor from a processed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn.train import train_dnn


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DNN training."""
    parser = argparse.ArgumentParser(description="Train LS-DNN model.")
    parser.add_argument("--processed-path", type=str, required=True, help="Path to processed .npz dataset.")
    parser.add_argument("--hidden-dims", type=str, default="256,256", help="Comma-separated hidden layer sizes.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def _parse_hidden_dims(hidden_dims_str: str) -> tuple[int, ...]:
    """Convert comma-separated hidden dims string to tuple."""
    dims = tuple(int(item.strip()) for item in hidden_dims_str.split(",") if item.strip())
    if len(dims) < 2:
        raise ValueError("Please provide at least two hidden layer sizes, e.g. 256,256")
    return dims


def main() -> None:
    """Train model and print concise summary."""
    args = parse_args()
    processed_path = Path(args.processed_path)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    outputs = train_dnn(
        processed_dataset_path=processed_path,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    history_data = json.loads(outputs["history_path"].read_text(encoding="utf-8"))
    val_losses = history_data.get("val_loss", [])
    final_val_loss = float(val_losses[-1]) if val_losses else float("nan")

    processed_npz = np.load(processed_path, allow_pickle=False)
    residual_mode = bool(np.asarray(processed_npz["residual_mode"]).item())

    print("Training completed.")
    print(f"Model path   : {outputs['model_path']}")
    print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Residual mode: {residual_mode}")


if __name__ == "__main__":
    main()
