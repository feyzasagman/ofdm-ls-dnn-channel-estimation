"""Training utilities for the LS-based DNN regressor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.dnn.model import build_dnn_regressor


from src.core.seed import set_global_seed


def train_dnn(
    processed_dataset_path: str | Path,
    hidden_dims: tuple[int, ...] = (256, 256),
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    random_seed: int = 42,
    model_path: str | Path | None = None,
    history_path: str | Path | None = None,
) -> dict[str, Path]:
    """Train DNN with x_train/y_train and x_val/y_val from processed NPZ."""
    set_global_seed(random_seed)
    data = np.load(Path(processed_dataset_path), allow_pickle=False)
    x_train = np.asarray(data["x_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32)

    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("x_train and y_train must be 2D arrays.")
    if x_val.ndim != 2 or y_val.ndim != 2:
        raise ValueError("x_val and y_val must be 2D arrays.")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("x_train and x_val feature dimensions must match.")
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError("y_train and y_val feature dimensions must match.")

    model = build_dnn_regressor(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=hidden_dims,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=2,
    )

    resolved_model_path = _resolve_model_path(processed_dataset_path, model_path)
    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(resolved_model_path)

    resolved_history_path = _resolve_history_path(processed_dataset_path, history_path)
    resolved_history_path.parent.mkdir(parents=True, exist_ok=True)
    n_epochs_run = len(history.history.get("loss", []))
    history_payload = {
        "random_seed": int(random_seed),
        "epochs_requested": int(epochs),
        "epochs_run": int(n_epochs_run),
        "early_stopping_epoch": int(n_epochs_run) if n_epochs_run < epochs else None,
        "loss": [float(v) for v in history.history.get("loss", [])],
        "mae": [float(v) for v in history.history.get("mae", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_mae": [float(v) for v in history.history.get("val_mae", [])],
    }
    resolved_history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    return {
        "model_path": resolved_model_path,
        "history_path": resolved_history_path,
    }


def _resolve_model_path(processed_dataset_path: str | Path, model_path: str | Path | None) -> Path:
    """Resolve output model path."""
    if model_path is not None:
        return Path(model_path)
    project_root = Path(__file__).resolve().parents[2]
    stem = Path(processed_dataset_path).stem
    return project_root / "data" / "results" / "logs" / f"{stem}_dnn.keras"


def _resolve_history_path(
    processed_dataset_path: str | Path, history_path: str | Path | None
) -> Path:
    """Resolve output history path."""
    if history_path is not None:
        return Path(history_path)
    project_root = Path(__file__).resolve().parents[2]
    stem = Path(processed_dataset_path).stem
    return project_root / "data" / "results" / "logs" / f"{stem}_history.json"
