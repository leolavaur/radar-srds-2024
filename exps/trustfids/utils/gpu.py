"""Module for GPU settings and helpers."""

import tensorflow as tf


def tf_gpu_count() -> int:
    """Count the number of GPUs available."""
    return len(tf.config.list_physical_devices("GPU"))


def tf_gpu_setup() -> None:
    """Setup TensorFlow GPU memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # automatically place the operations to run on CPU if
                # a GPU is not available
                tf.config.set_soft_device_placement(True)

                # "Logical GPUs")
            logical_gpus = tf.config.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
