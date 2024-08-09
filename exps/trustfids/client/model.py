"""ML Models for NF-V2.

This module contains functions to create ML models for the NF-V2 dataset.

Part of the code in this module is based on the code from Bertoli et al. (2022),
who tested Federated Learning on the NF-V2 dataset.

The code is available at:
    https://github.com/c2dc/fl-unsup-nids

References:
    * Bertoli, G., Junior, L., Santos, A., & Saotome, O., Generalizing intrusion
      detection for heterogeneous networks: A stacked-unsupervised federated
      learning approach. arXiv preprint arxiv:2209.00721 (2022).
      https://arxiv.org/abs/2209.00721
"""
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
from flwr.common import Scalar
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

keras = tf.keras
from keras.layers import Dense, Input
from keras.losses import Loss
from keras.metrics import (
    AUC,
    FalseNegatives,
    FalsePositives,
    Precision,
    Recall,
    TrueNegatives,
    TruePositives,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam, Optimizer


def create_model(
    num_features: int,
    loss_fn: Loss | Callable[[ArrayLike, ArrayLike], float],
    optimizer: str | Optimizer | None = None,
) -> Model:
    """Create a Keras model for the NF-V2 dataset.

    Args:
        num_features (int): Number of features in the dataset.

    Returns:
        Model: Keras model.
    """
    model = Sequential(
        [
            Input(shape=(num_features,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(8, activation="relu"),
            Dense(16, activation="relu"),
            Dense(32, activation="relu"),
            Dense(num_features, activation="sigmoid"),
        ]
    )

    if optimizer is None:
        optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        # metrics=[
        #     Precision(name="precision"),
        #     Recall(name="recall"),
        #     AUC(name="auc"),
        #     TruePositives(name="tp"),
        #     FalsePositives(name="fp"),
        #     TrueNegatives(name="tn"),
        #     FalseNegatives(name="fn"),
        # ],
    )

    return model
