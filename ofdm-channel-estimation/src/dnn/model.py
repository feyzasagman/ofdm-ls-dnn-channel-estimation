"""DNN model definition for channel-regression tasks."""

from __future__ import annotations

from typing import Sequence

from tensorflow import keras


def build_dnn_regressor(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256),
) -> keras.Model:
    """Build a small fully-connected regressor (Dense-ReLU-Dense-ReLU-Dense)."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")
    if len(hidden_dims) < 2:
        raise ValueError("hidden_dims must contain at least two layers.")
    _validate_hidden_dims(hidden_dims)

    model = keras.Sequential(name="ls_dnn_regressor")
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(hidden_dims[0], activation="relu"))
    model.add(keras.layers.Dense(hidden_dims[1], activation="relu"))
    for units in hidden_dims[2:]:
        model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dense(output_dim, activation=None))
    return model


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> None:
    """Validate hidden layer size configuration."""
    for idx, units in enumerate(hidden_dims):
        if units <= 0:
            raise ValueError(f"hidden_dims[{idx}] must be positive.")
