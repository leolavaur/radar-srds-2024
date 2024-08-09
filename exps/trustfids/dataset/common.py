"""Dataset management utilities.

This submodule is """

# Required for type annotations to work with Python 3.7+, and especially forward
# references:
# class C:
#     def foo(self) -> "C":
#         return self
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from efc import EnergyBasedFlowClassifier
from flwr.common import Scalar
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle as sk_shuffle
from trustfids.utils.log import logger

keras = tf.keras
from keras.utils import Sequence


class BatchLoader(Sequence):
    """Generator of batches for training."""

    X: pd.DataFrame
    target: pd.DataFrame | pd.Series | None

    batch_size: int

    def __init__(
        self,
        batch_size: int,
        X: pd.DataFrame,
        target: pd.DataFrame | pd.Series | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        """Initialise the BatchLoader."""
        self.batch_size = batch_size

        self.X = X
        self.target = target if target is not None else X.copy()

        if shuffle:
            indices = np.arange(len(X))
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)

            self.X = X.iloc[indices]
            self.target = self.target.iloc[indices]

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch_size.
        high = min(low + self.batch_size, len(self.X))

        batch_x = self.X[low:high]
        if self.target is None:
            return batch_x

        batch_target = self.target[low:high]

        # A Sequence should apparently return a tuple of NumPy arrays, as DataFrames
        # cause errors in the fit() method.
        return batch_x.to_numpy(), batch_target.to_numpy()


@dataclass
class Dataset:
    """Dataset class."""

    X: pd.DataFrame
    y: pd.Series  # | pd.DataFrame
    m: pd.DataFrame

    def __getitem__(self, key):
        """Overload the [] operator to access the dataset attributes.

        This allows to consider a Dataset object as a tuple of the form (X, y, m),
        enabling easier access to the attributes:
            * X = dataset[0]
            * X, y, m = dataset
            * some_function(*dataset)
            * ...
        """
        if key == 0:
            return self.X
        elif key == 1:
            return self.y
        elif key == 2:
            return self.m
        else:
            raise IndexError()

    def __len__(self):
        return len(self.X)

    def __eq__(self, other):
        return (
            self.X.equals(other.X) and self.y.equals(other.y) and self.m.equals(other.m)
        )

    def to_sequence(
        self,
        batch_size: int,
        target: int | None = None,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> BatchLoader:
        """Convert the dataset to a BatchLoader object.

        Args:
            batch_size (int): Size of the batches.
            target (int): Target to use for the batches, defaults to None. 0 for X, 1
                for y, 2 for m.

        Returns:
            BatchLoader: BatchLoader object.
        """
        if target is None:
            return BatchLoader(batch_size, self.X, seed=seed, shuffle=shuffle)
        if 0 <= target <= 2:
            return BatchLoader(
                batch_size, self.X, self[target], seed=seed, shuffle=shuffle
            )
        raise IndexError("If not None, parameter `target` must be in [0, 2]")

    def split(
        self,
        at: float,
        seed: int | None = None,
        stratify: Optional[pd.Series] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Split the dataset into a training and a test set.

        Parameters:
        -----------
        at : float
            Ratio where to split the dataset. Must be in [0, 1]. The first dataset will
            contain `at`% of the samples, the second one will contain the remaining.
        seed : int, optional
            Seed for the random number generator, by default None.
        strat_series : pd.Series, optional
            Series to use for stratification, by default None.

        Returns:
        --------
        Tuple[Dataset, Dataset]
            Tuple of the training and test sets.
        """

        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            *self,
            train_size=at,
            random_state=seed,
            stratify=np.array(stratify) if stratify is not None else None,
        )

        return (
            Dataset(X_train, y_train, m_train),
            Dataset(X_test, y_test, m_test),
        )

    def copy(self):
        """Return a copy of the dataset."""
        return Dataset(self.X.copy(), self.y.copy(), self.m.copy())

    def get(self, indices: List[int]) -> Dataset:
        """Return a subset of the dataset."""
        return Dataset(self.X.loc[indices], self.y.loc[indices], self.m.loc[indices])

    def drop_random_class(
        self,
        class_to_avoid: List[str] = ["Benign"],
        class_to_drop: Optional[str] = None,
        nb_class_to_drop: int = 1,
        client_id: Optional[str] = None,
    ) -> List[str]:
        """Drop a random class from a Dataset.

        Args:
            class_to_avoid (List[str], optional): Class that shouldn't be dropped. Defaults to ["Benign"].
            class_to_drop (List[str], optional): Force the specified class to be droped instead of a random class. Specified class must not be in class to avoid. Defaults to None.
            nb_class_to_drop (int, optional): Number of dropped classes. Default to 1.
            client_id (Optional[str], optional): Id of the client that will use the dataset, used for logging purposes Defaults to None.
        Return :
            List[str]: Dropped class(es)
        """
        class_column: str = "Attack"
        dropped_classes: List[str] = []

        for _ in range(nb_class_to_drop):
            available_classes = self.m[class_column].unique()

            if len(set(available_classes) - set(class_to_avoid)) < nb_class_to_drop:
                logger.warn(
                    "Not enough class to drop",
                    f"Potential class to drop {available_classes}",
                    f"Classes to avoid : {class_to_avoid}",
                )
                raise ValueError(
                    f"There are not enough class to drop when the class to avoid have been excluded. Nb class to drop {nb_class_to_drop} Classes to avoid : {class_to_avoid}"
                )
            dropped_class: str = class_to_drop if class_to_drop else "Benign"
            while any(dropped_class == c for c in class_to_avoid):
                dropped_class = np.random.choice(available_classes, replace=True)

            logger.info(
                f"Dropping class {dropped_class}" + f" on client {client_id}"
                if client_id
                else ""
            )
            mask = self.m[class_column].isin([dropped_class])
            self.m = self.m.drop(mask[mask].index)
            self.X = self.X.drop(mask[mask].index)
            self.y = self.y.drop(mask[mask].index)
            dropped_classes.append(dropped_class)
        return dropped_classes

    def drop_all_but_random_class(
        self,
        classes_to_avoid: List[str] = [],
        classes_to_keep: List[str] = ["Benign"],
        classes_to_keep_only: bool = False,
        client_id: Optional[str] = None,
    ) -> str:
        """Drop all attacker classes except a random one from a Dataset.

        Args:
            classes_to_avoid (List[str], optional): Class that shouldn't be kept as the only class.
            classes_to_keep (List[str], optional): Classes that should be kept in addition to the random selected attacker class.
            classes_to_keep_only (bool, optional): Only keep classes_to_keep on the client, drop all other classes (no supplementary random attacker class is kept). Default to False.
            client_id (Optional[str], optional): Id of the client that will use the dataset, used for logging purposes Defaults to None.
        Return :
            str: Kept class
        """
        class_column: str = "Attack"
        available_classes: List[str] = list(self.m[class_column].unique())
        kept_class: str = "Benign"
        classes_to_avoid.append(kept_class)

        if not classes_to_keep_only:
            while any(kept_class == c for c in classes_to_avoid):
                kept_class = np.random.choice(available_classes)
            classes_to_keep.append(kept_class)

        dropped_classes: List[str] = list(set(available_classes) - set(classes_to_keep))

        logger.info(
            f"Keeping class {classes_to_keep}" + f" on client {client_id}"
            if client_id
            else ""
        )

        mask = self.m[class_column].isin(dropped_classes)
        self.m = self.m.drop(mask[mask].index)
        self.X = self.X.drop(mask[mask].index)
        self.y = self.y.drop(mask[mask].index)

        return kept_class

def mean_absolute_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean absolute error.

    Args:
        x_orig (pd.DataFrame): True labels.
        x_pred (pd.DataFrame): Predicted labels.

    Returns:
        ndarray[float]: Mean absolute error.
    """
    return np.mean(np.abs(x_orig - x_pred), axis=1)


def mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean squared error.

    Args:
        x_orig (pd.DataFrame): True labels.
        x_pred (pd.DataFrame): Predicted labels.

    Returns:
        ndarray[float]: Mean squared error.
    """
    return np.mean((x_orig - x_pred) ** 2, axis=1)


def root_mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Root mean squared error.

    Args:
        x_orig (pd.DataFrame): True labels.
        x_pred (pd.DataFrame): Predicted labels.

    Returns:
        ndarray[float]: Root mean squared error.
    """
    return np.sqrt(np.mean((x_orig - x_pred) ** 2, axis=1))
