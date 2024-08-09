"""NF-V2 Dataset utilities.

This module contains functions to load and prepare the NF-V2 dataset for
Deep Learning applications. The NF-V2 dataset is a collection of 4 datasets
with a standardised set of features. The datasets are:
    * CSE-CIC-IDS-2018
    * UNSW-NB15
    * ToN-IoT
    * Bot-IoT

The NF-V2 dataset is available at:
    https://staff.itee.uq.edu.au/marius/NIDS_datasets/

Part of the code in this module is based on the code from Bertoli et al. (2022),
who tested Federated Learning on the NF-V2 dataset.

The code is available at:
    https://github.com/c2dc/fl-unsup-nids

References:
    * Sarhan, M., Layeghy, S. & Portmann, M., Towards a Standard Feature Set for
      Network Intrusion Detection System Datasets. Mobile Netw Appl (2021).
      https://doi.org/10.1007/s11036-021-01843-0 
    * Bertoli, G., Junior, L., Santos, A., & Saotome, O., Generalizing intrusion
      detection for heterogeneous networks: A stacked-unsupervised federated
      learning approach. arXiv preprint arxiv:2209.00721 (2022).
      https://arxiv.org/abs/2209.00721
"""
import math
import operator
from pathlib import Path
from tempfile import gettempdir
from typing import Callable, List, Optional, Tuple, overload

import numpy as np
import pandas as pd
from efc import EnergyBasedFlowClassifier
from omegaconf import ListConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..dataset.poisoning import PoisonOp
from ..utils.log import logged, logger
from .common import Dataset

# Default path to the datasets.
# -----------------------------
# The dataset is downloaded to this path if it is not found.
# On Linux, the default path is `/tmp/trustfids-data/nfv2`.
# In the directory, datasets are organised as follows:
#   /tmp/trustfids-data/nfv2
#   ├── origin
#   │   ├── NF-BoT-IoT-v2.csv.gz
#   │   ├── NF-CSE-CIC-IDS2018-v2.csv.gz
#   │   ├── NF-ToN-IoT-v2.csv.gz
#   │   └── NF-UNSW-NB15-v2.csv.gz
#   ├── reduced
#   │   ├── botiot_reduced.csv.gz
#   │   ├── cicids_reduced.csv.gz
#   │   ├── toniot_reduced.csv.gz
#   │   └── nb15_reduced.csv.gz
#   └── sampled
#       └── ...
DEFAULT_BASE_PATH = Path(gettempdir()) / f"{__name__.split('.')[0]}-data" / "nfv2"

# Shortcuts names for dataset paths.
# ----------------------------------
DATASET_KEYS = {
    "origin/botiot": "origin/NF-BoT-IoT-v2.csv.gz",
    "origin/cicids": "origin/NF-CSE-CIC-IDS2018-v2.csv.gz",
    "origin/toniot": "origin/NF-ToN-IoT-v2.csv.gz",
    "origin/nb15": "origin/NF-UNSW-NB15-v2.csv.gz",
    "reduced/botiot": "reduced/botiot_reduced.csv.gz",
    "reduced/cicids": "reduced/cicids_reduced.csv.gz",
    "reduced/toniot": "reduced/toniot_reduced.csv.gz",
    "reduced/nb15": "reduced/nb15_reduced.csv.gz",
    "sampled/botiot": "sampled/botiot_sampled.csv.gz",
    "sampled/cicids": "sampled/cicids_sampled.csv.gz",
    "sampled/toniot": "sampled/toniot_sampled.csv.gz",
    "sampled/nb15": "sampled/nb15_sampled.csv.gz",
}

# Columns to drop from the dataset.
# ---------------------------------
# The sampled and reduced datasets contain an additional column called `Dataset` which
# must be dropped as well.
RM_COLS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    "Attack",
]


@overload
def load_data(
    name: str,
    test_ratio: None = None,
    n_partitions: None = None,
    common_test: bool = False,
    base_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Dataset:
    """Load the NF-V2 dataset.

    If no path is given, the dataset is loaded from the default path.
    If the dataset is not found at the given path, it is downloaded using the `download`
    function.
    If the download function is not implemented, a `NotImplementedError` is raised.
    """
    ...


@overload
def load_data(
    name: str,
    test_ratio: float,
    n_partitions: None = None,
    common_test: bool = False,
    base_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load the NF-V2 dataset.

    If `test_ratio` is given, the dataset is split into a training and a testing set.
    """
    ...


@overload
def load_data(
    name: str,
    test_ratio: None = None,
    n_partitions: int = 0,
    common_test: bool = False,
    base_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> List[Dataset]:
    """Load the NF-V2 dataset.

    If `n_partition` is given, the dataset is split into `n_partition` partitions.
    """
    ...


@overload
def load_data(
    name: str,
    test_ratio: float,
    n_partitions: int,
    common_test: bool = False,
    base_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[Dataset, Dataset]]:
    """Load the NF-V2 dataset.

    If `test_ratio` and `n_partition` are given, the dataset is split into
    `n_partition` partitions, each of which is split into a training and a testing set.
    """
    ...


@logged
def load_data(
    name: str,
    test_ratio: Optional[float] = None,
    n_partitions: Optional[int] = None,
    common_test: bool = False,
    base_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Dataset | Tuple[Dataset, Dataset] | List[Dataset] | List[Tuple[Dataset, Dataset]]:
    """Load a NF-V2 dataset.

    This function is overloaded to allow different output types depending on the given
    parameters. The following output types are possible:

    - `Dataset`: If no split is performed.
    - `Tuple[Dataset, Dataset]`: If `test_ratio` is given. The first element is the
        training set, the second element is the testing set.
    - `List[Dataset]`: If `n_partition` is given. The dataset is split into
      `n_partition`.
    - `List[Tuple[Dataset, Dataset]]`: If `test_ratio` and `n_partition` are given. The
        dataset is split into training and testing sets, which are then split into
        `n_partition` depending on the `common_test` parameter.

    Parameters:
    -----------
    name : str
        Name of the dataset to load. Can be a shortcut name or a path to a CSV file.
    test_ratio : float, optional
        Ratio of the testing set. If given, the dataset is split into a training and a
        testing set.
    n_partitions : int, optional
        Number of partitions to split the dataset into. If given, the dataset is split
        into `n_partition` partitions.
    common_test : bool, optional
        If `True`, `test_ratio` is given and `n_partition` is greater than 1, the
        testing set is the same for all partitions.
    base_path : str or Path, optional
        Path to the directory containing the dataset. If not given, the dataset is
        loaded from the default path.
    seed : int, optional
        Seed for shuffling the dataset.

    shuffle : bool, optional
        If `True`, the dataset is shuffled before being split.

    Returns:
    --------
    Union
        Depending on the parameters, the function returns a single dataset, a tuple of
        two datasets, a list of datasets or a list of tuples of two datasets.

    Raises:
    -------
    FileNotFoundError
        If the dataset is not found at the given path.
    """

    # PATH MANAGEMENT
    # ---------------

    if name in DATASET_KEYS:
        # Assume name is a key to find the dataset in base_path

        if base_path is None:
            base_path = DEFAULT_BASE_PATH

        base = Path(base_path)

        path = base / Path(DATASET_KEYS[name])

    else:
        # Assume name is a path to a CSV file

        if base_path is None:
            if Path(name).exists():
                # It the path is reachable, use it

                if Path(name).is_absolute():
                    base = Path("/")
                else:
                    base = Path(".")

                path = base / Path(name)

            else:
                # Else, assume the path is relative to the default base path
                base = Path(DEFAULT_BASE_PATH)
                path = base / Path(name)

        else:
            # Assume the path is relative to the given base path
            base = Path(base_path)
            path = base / Path(name)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Either check your inputs, or download the dataset first.",
            {
                "name": name,
                "base": base,
                "current": Path(".").absolute(),
                "path": path.resolve().absolute(),
            },
        )

    logger.info(f"Loading dataset '{name}'.")

    df = pd.read_csv(path, low_memory=True)

    # INPUT VALIDATION
    # ----------------

    if test_ratio is not None and not 0 < test_ratio < 1:
        raise ValueError(
            f"Invalid value for `test_ratio`: {test_ratio}. Must be between 0 and 1."
        )

    if n_partitions is not None and not (1 <= n_partitions <= len(df)):
        raise ValueError(
            f"Invalid value for `n_partitions`: {n_partitions}. Must be between 1 and length of the dataset."
        )

    # DATA PREPROCESSING
    # ------------------

    # shuffle the dataset
    if shuffle:
        df = df.sample(frac=1, random_state=seed)

    # Remove classes that have too few occurences
    dropped_classes = ["injection", "mitm", "ransomware", "worms"]
    mask = df["Attack"].isin(dropped_classes)
    df = df.drop(mask[mask].index)

    # drop the "Dataset" column if it exists
    if "Dataset" in df.columns:
        df = df.drop(columns=["Dataset"])

    # select the columns to compose the Dataset object
    X = df.drop(columns=RM_COLS)
    y = df["Label"]
    m = df[RM_COLS]

    # convert classes to numerical values
    X = pd.get_dummies(X)

    # normalize the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X[X.columns] = scaler.transform(X)

    # DATA PARTITIONING
    # -----------------

    if test_ratio is None:
        if n_partitions is None:
            return Dataset(X, y, m)

        else:
            return _partition(Dataset(X, y, m), n_partitions)

    else:
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            X, y, m, test_size=test_ratio, random_state=seed, stratify=m["Attack"]
        )

        train = Dataset(X_train, y_train, m_train)
        test = Dataset(X_test, y_test, m_test)

        if n_partitions is None:
            return train, test

        else:
            if common_test:
                return list(
                    zip(
                        _partition(train, n_partitions),
                        [test] * n_partitions,
                    )
                )
            else:
                return list(
                    zip(
                        _partition(train, n_partitions),
                        _partition(test, n_partitions),
                    )
                )

    raise ValueError("Invalid combination of arguments.")


def download(path: Optional[str] = None) -> None:
    """Download the NF-V2 dataset to a given path.

    If no path is given, the dataset is downloaded to the default path.
    If the dataset is already present at the given path, no action is taken."""
    raise NotImplementedError("Download function not implemented.")


def poison(
    dataset: Dataset,
    ratio: float,
    op: PoisonOp,
    target_classes: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dataset, int]:
    """Poison a dataset by apply a function to a given number of samples.

    Parameters:
    -----------
    dataset: Dataset
        Dataset to poison.
    n: int
        Number of samples to poison in the target. If `target` is None, the whole
        dataset is poisoned.
    op: PoisonOp
        Poisoning operation to apply. Either PoisonOp.INC or PoisonOp.DEC.
    target_classes: Optional[List[str]]
        List of classes to poison. If None, all classes are poisoned, including benign samples.
    seed: Optional[int]
        Seed for reproducibility.

    Returns:
    --------
    Dataset
        The poisoned dataset.
    int
        The number of samples that have been modified.
    """
    d = dataset.copy()

    assert target_classes is None or (
        isinstance(target_classes, List | ListConfig)
        and all(isinstance(c, str) for c in target_classes)
    ), "Invalid value for `target_classes`. Must be a list of strings or None."

    if target_classes is None:
        # If targeted means all dataset (including benign samples)
        # target is a boolean Series of the same length as the dataset
        target = pd.Series(True, index=d.y.index)

    elif target_classes == ["*"]:
        # If targeted means all attacks (excluding benign samples)
        target = d.m["Attack"] != "Benign"

    else:
        target = d.m["Attack"].isin(target_classes)

    # reindex target using the dataset index

    n = np.ceil(sum(target) * ratio).astype(int)
    if n > sum(target):
        raise ValueError(
            f"Invalid value for `ratio`: ratio * len(target) = {n}. Must be less or equal to len(target)."
        )

    if len(target) != len(dataset):
        raise ValueError(
            f"Invalid value for `target`. Must be of the same length as the dataset."
        )

    if target.dtype != bool:
        raise ValueError(f"Invalid value for `target`. Must be a boolean Series.")

    # get poisoning metadata
    if "Poisoned" not in d.m.columns:
        d.m["Poisoned"] = False

    if op == PoisonOp.DEC:
        target = target & d.m["Poisoned"]
    else:
        target = target & ~d.m["Poisoned"]

    # get the indices of the samples to poison (cap n at the number of available samples)
    n = min(n, sum(target))
    idx = d.y[target].sample(n=n, random_state=seed).index.to_list()

    # apply the poisoning operation
    d.y.loc[idx] = d.y[idx].apply(operator.not_)
    d.y = d.y.astype(int)
    if op == PoisonOp.DEC:
        d.m.loc[idx, "Poisoned"] = False
    else:
        d.m.loc[idx, "Poisoned"] = True

    # clean up
    del target
    del dataset

    return d, len(idx)


def _partition(d: Dataset, n_partition: int) -> List[Dataset]:
    """Partition the dataset into n partitions.

    Args:
        d (Dataset): Dataset to partition.
        n_partition (int): Number of partitions.
    Returns:
        List of Datasets.
    """

    partition_size = math.floor(len(d.X) / n_partition)
    partitions = []
    for i in range(n_partition):
        idx_from, idx_to = i * partition_size, (i + 1) * partition_size
        X_part = d.X[idx_from:idx_to]
        y_part = d.y[idx_from:idx_to]
        m_part = d.m[idx_from:idx_to]

        partitions.append(Dataset(X_part, y_part, m_part))

    # TODO make it optionnal (get an hydra conf attribute here)
    # get available classes.
    # /!\ Prevent the drop of the benign class
    return partitions


########################################################################################
# Old functions.
########################################################################################


def add_efc(
    datasets: Tuple[Dataset, Dataset],
    benign_efc: bool,
    norm_efc: bool,
) -> Tuple[Dataset, Dataset]:
    """Add EFC to the dataset.

    Args:
        datasets (Tuple[Dataset, Dataset]): Tuple of train and test datasets.
        benign_efc (bool): Whether to train EFC on benign samples only.
        norm_efc (bool): Whether to normalize the energies.

    Returns:
        Tuple[Dataset, Dataset]: Tuple of train and test datasets with EFC.
    """
    train, test = datasets
    if not any(train.y == 1):
        logger.warn(
            "No attack samples in training set, training EFC only on benign samples."
        )

    X_efc = train.X.copy()
    y_efc = train.y.copy()
    if benign_efc:
        # EFC is typically trained using labels. However, in this case we aim at
        # unsupervised learning only. This is to test the effect of using only
        # benign samples for training EFC.
        filt = pd.Series(y_efc == 0)
        X_efc = X_efc[filt]
        y_efc = y_efc[filt]

    ebfc = EnergyBasedFlowClassifier(cutoff_quantile=0.95)
    ebfc.fit(X_efc, y_efc)

    _, y_train_energies = ebfc.predict(train.X, return_energies=True)
    _, y_test_energies = ebfc.predict(test.X, return_energies=True)
    y_train_energies = y_train_energies.reshape(-1, 1)
    y_test_energies = y_test_energies.reshape(-1, 1)

    if norm_efc:
        # Normalize energies to [0, 1] using min-max scaling
        scaler = MinMaxScaler()
        scaler.fit(y_train_energies)
        y_train_energies = scaler.transform(y_train_energies)
        y_test_energies = scaler.transform(y_test_energies)

    train.X["ENERGY"] = y_train_energies
    test.X["ENERGY"] = y_test_energies

    filt = pd.Series(train.y == 0)
    train.X = train.X[filt]
    train.y = train.y[filt]
    train.m = train.m[filt]

    return train, test


def load_siloed_data(
    D: List[str],
    clients: List[int],
    seed: int,
    only_benign: Optional[bool] = None,  # Deprecated
    with_efc: bool = False,
    norm_efc: bool = False,
    benign_efc: bool = False,
) -> List[Tuple[Dataset, Dataset]]:
    """Create datasets for N clients.

    Args:
        D (List[str]): List of dataset paths.
        clients (List[Scalar]): List of number of clients per dataset,
            with len(D) == len(clients).
        seed (int): Seed for reproducibility.

    Returns:
        List[Tuple[Dataset,Dataset]]: List of datasets for each client.
    """
    if only_benign is not None:
        logger.warn("only_benign is deprecated and will be removed in the future.")

    assert len(D) == len(
        clients
    ), "Number of datasets and number of clients must be equal."

    data = []
    for d, n in zip(D, clients):
        l = load_data(
            d,
            test_ratio=0.2,
            n_partitions=n,
            seed=seed,
        )
        data.extend(l if n > 1 else [l])

    if with_efc:
        ret_data = []
        logger.info("Experiment run with EFC, computing energies...")
        for d in data:
            ret_data.append(add_efc(d, benign_efc, norm_efc))
        data = ret_data

    return data
