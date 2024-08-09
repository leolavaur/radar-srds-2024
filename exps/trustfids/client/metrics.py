from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Add metrics from Keras to scope to allow for easier imports
import tensorflow as tf
from flwr.common import Scalar
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

keras = tf.keras
from keras.metrics import mean_absolute_error, mean_squared_error


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Scalar]:
    """Evaluate the predictions of a model.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary with the evaluation metrics (accuracy, precision,
            recall, f1, mcc, auc, missrate, fallout)
    """
    conf = confusion_matrix(y_true, y_pred)
    try:
        tn = conf[0][0]
        fp = conf[0][1]
        fn = conf[1][0]
        tp = conf[1][1]
    except IndexError:
        if all(y_pred) and all(y_true):
            tn = 0
            fp = 0
            fn = 0
            tp = len(y_true)
        elif not any(y_pred) and not any(y_true):
            tn = len(y_true)
            fp = 0
            fn = 0
            tp = 0
        else:
            raise ValueError("Invalid confusion matrix.")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        # removed; see https://stats.stackexchange.com/a/372977
        # "auc": roc_auc_score(y_true, y_pred),
        "missrate": fn / (fn + tp) if (fn + tp) != 0 else 0,
        "fallout": fp / (fp + tn) if (fp + tn) != 0 else 0,
    }  # type: ignore


def root_mean_squared_error(y_true, y_pred):
    """Calculate the root mean squared error.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Root mean squared error.
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))
