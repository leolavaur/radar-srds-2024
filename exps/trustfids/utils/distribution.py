from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from trustfids.utils.log import logger


def make_original_distribution(
    distribution: List[int], cids: List[str], silos: List[str]
) -> OrderedDict[str, List[str]]:
    """Represent the distribution of clients in a dictionary.

    Arguments:
        distribution: The number of clients per silo.
        cids: The list of client IDs.
        silos: The list of silo names.

    Returns:
        A dictionary mapping silo names to a list of client IDs.
    """
    clusters = OrderedDict()
    for silo, n_clients in zip(silos, distribution):
        name = Path(silo).stem.split(".")[0]
        clusters[name] = cids[:n_clients]
        cids = cids[n_clients:]
    return clusters


def build_distribution_baseline(distribution: List[int]) -> List[List[str]]:
    """return the expected cluster distribution from distribution scalars.
    Args:
        distribution: List of scalar reprensenting the distribution of clients.
    Return:
        List of List of client_id representing the expected cluster distribution.

    """
    client_id = 0
    result = []
    for silo_size in distribution:
        l = []
        for c in range(silo_size):
            l.append(f"client_{client_id}")
            client_id += 1
        result.append(l)
    return result


def build_merged_distribution_baseline(distribution: List[int]) -> List[List[str]]:
    """return the expected cluster distribution with IoT dataset merged from distribution scalars.
    Args:
        distribution: List of scalar reprensenting the distribution of clients.
    Return:
        List of List of client_id representing the expected cluster distribution with IoT datasets merged
    """
    client_id = 0
    result = []
    for silo_size in distribution:
        l = []
        for c in range(silo_size):
            l.append(f"client_{client_id}")
            client_id += 1
        result.append(l)
    # IoT datasets are currently in first and last place
    result[0] += result[-1]
    return result[:-1]


def clients_per_silo(N: int, D: List[str]) -> List[int]:
    """Create datasets for N clients.

    Args:
        N (int): Number of clients.
        D (List[str]): List of dataset paths.

    Returns:
        List[Scalar]: Distribution of clients per dataset.
    """
    # get dataset sizes
    sizes = []
    for d in D:  # TODO:  avoid loading the whole dataset another time
        df = pd.read_csv(d, low_memory=True)
        sizes.append(len(df[df["Label"] == 0]))
        logger.debug(f"Loaded dataset {d} with {len(df)} samples.")

    # get number of clients per dataset
    # n_clients = round_keep_sum(list(sizes * N / sum(sizes)))
    n_clients = distribute_clients(N, sizes, min=2)
    logger.debug(f"Number of clients per dataset: {n_clients}")

    # n_client will probably not be equal to N, but rather N - 1
    # if sum(n_clients) != N:
    #     # we need to chose which dataset to add the client to
    #     # we add it to the one which is the farthest from the ideal size
    #     sizes_prime = sum(sizes) * n_clients // N
    #     deltas = np.abs(sizes_prime - sizes)
    #     idx = np.argmax(deltas)
    #     n_clients[idx] += 1

    assert (
        sum(n_clients) == N
    ), f"Number of clients ({sum(n_clients)}) is not equal to N ({N})."

    return list(n_clients)


def distribute_clients(N: int, sizes: List[int], min: int = 1) -> List[int]:
    """Distribute N clients propertionally over the datasets.

    When min >= 1, the number of clients per dataset will be at least min, which means
    that other datasets will get less clients.

    Args:
        N (int): Number of clients.
        sizes (List[int]): List of dataset sizes.
        min (int, optional): Minimum number of clients per dataset. Defaults to 1.

    Returns:
        List[int]: List of number of clients per dataset.
    """
    if len(sizes) * min > N:
        raise ValueError(
            f"Number of clients ({N}) is too small for the minimum number of clients per datasets ({min})."
        )
    n_clients = np.asarray([min for _ in range(len(sizes))])

    # get the number of clients remaining after assigning min to each dataset
    weights = np.array(sizes) / sum(sizes)
    N_remain = N - sum(n_clients)

    # repartition of the remaining clients as floats
    n = weights * N_remain

    # round the floats to integers, while keeping their sum
    n_r = np.around(n).astype(int)
    while sum(n_r) != sum(n):
        inc = 1 if sum(n_r) < sum(n) else -1
        idx = np.argmax(np.abs(n_r - n))
        n_r[idx] += inc

    # should never trigger
    assert sum(n_r + n_clients) == N, "Sum of clients is not equal to N."

    return (n_clients + n_r).tolist()
