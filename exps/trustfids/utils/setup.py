"""Module for ML settings and helpers.

This module contains functions for setting up the ML environment, such as
setting the random seed and configuring the GPU memory growth.
Other modules of Trust-FIDS shoud import this module instead of importing
TensorFlow directly.
"""
import numpy as np
import tensorflow as tf
from trustfids.utils.log import logger


def set_seed(seed: int) -> None:
    """Set random seed for numpy and tensorflow."""

    tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()
