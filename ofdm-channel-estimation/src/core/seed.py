"""Deterministic seed utilities for reproducible experiments."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Best-effort deterministic ops where supported.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass
