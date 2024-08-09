"""Module for experiment-related tooling."""

import os  # isort:skip

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import wraps
from typing import Callable, List

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from trustfids.utils.log import logger
from trustfids.utils.setup import tf_gpu_setup


def tf_setup(f: Callable) -> Callable:
    """Decorator for setting up TensorFlow GPU memory growth."""

    def wrapper(*args, **kwargs) -> None:
        logger.info("Initializing GPUs for TensorFlow")
        # Setup TensorFlow configuration
        tf_gpu_setup()

        # Run actual function
        return f(*args, **kwargs)

    return wrapper


def save_progress(f: Callable) -> Callable:
    """Decorator for saving progress of experiment.

    This decorator can currently only be used on the following class methods:
        * `flwr.client.NumPyClient.fit`

    The decorator also requires that the following attributes are present in the
    config dictionary (defined in the server's strategy):
        * `round`

    Ie.:
    ```python
        strategy=fl.server.strategy.FedAvg(
            on_fit_config_fn=lambda rnd: { "round": rnd }
        ),
    ```

    Saved models use the following naming convention:
    ```
        <round>_<client_id>_<n_samples>.npz
    ```
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        assert "round" in args[2], "Missing round in config dictionary"

        cid = args[0].cid
        round = args[2]["round"]

        # Run actual function
        response = f(*args, **kwargs)

        np_model, n_samples, m_dict = response
        threshold = m_dict["threshold"]

        # `np.save` cannot handle lists of heterogeneously dimensioned numpy arrays.
        # `np.savez` can, but it requires values to be passed individualy. Therefore,
        # we need to unpack the list of numpy arrays using the `*` operator.

        np.savez(
            f"{str(round).zfill(3)}_{cid}_{n_samples}_{threshold}.npz",
            *np_model,
        )

        return response

    return wrapper


def save_progress_with_conf(cfg: DictConfig) -> Callable:
    """Decorator for saving progress of experiment.

    This decorator can currently only be used on the following class methods:
        * `flwr.client.NumPyClient.fit`

    The decorator also requires that the following attributes are present in the
    config dictionary (defined in the server's strategy):
        * `round`

    Ie.:
    ```python
        strategy=fl.server.strategy.FedAvg(
            on_fit_config_fn=lambda rnd: { "round": rnd }
        ),
    ```

    Saved models use the following naming convention:
    ```
        <round>_<client_id>_<n_samples>.npz
    ```
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            assert "round" in args[2], "Missing round in config dictionary"

            cid = args[0].cid
            round = args[2]["round"]

            # Run actual function
            response = f(*args, **kwargs)

            np_model, n_samples, m_dict = response
            threshold = m_dict["threshold"]

            # `np.save` cannot handle lists of heterogeneously dimensioned numpy arrays.
            # `np.savez` can, but it requires values to be passed individualy. Therefore,
            # we need to unpack the list of numpy arrays using the `*` operator.

            np.savez(
                f"{str(round).zfill(3)}_{cid}_{n_samples}_{threshold}.npz",
                *np_model,
            )

            return response

        return wrapper

    return decorator
