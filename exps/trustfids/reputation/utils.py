"""
A set of utils for the reputation module. 
"""
import math
from typing import Any, Dict, List

import numpy as np
from scipy.stats import norm


def discretize(number: float, class_nb: int) -> int:
    """Discretization of evaluations
    Chose the class closest to given number.
    Args :
        number: number to discretize 0<=number<=1
        granularity : number of class for the discretization.
    Return:
        class closest to the number.
    """
    seuil = 1.0 / class_nb
    if number < seuil:
        # Prevent value from being zero.
        return 1

    for i in range(class_nb):
        if i * seuil <= number < (i + 1) * seuil:
            return i

    if number == 1.0:
        return class_nb - 1


def discretize_matrice(
    ratings: Dict[str, Dict[str, float]], class_nb: int
) -> Dict[str, Dict[str, int]]:
    """Discretize matrice
    Args:
        ratings : inversed xevals.
        class_number : number of class for the discretization.
    Return:
        discretized ratings.
    """
    for participant in ratings.values():
        for evaluator in participant:
            participant[evaluator] = discretize(participant[evaluator], class_nb)
    return ratings


def xeval_max(xeval: Dict[str, Dict[str, float]]) -> float:
    """Discretize matrice
    Args:
        xeval : xevals.
    Return:
        Maximum value from xeval.
    """
    return max([max(evals.values()) for evals in xeval.values()])


def normalize_loss_matrice(
    xeval: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Normalize matrice
    Args:
        xeval : xevals.
    Return:
        normalized xevals.
    """
    for participant in xeval.values():
        for evaluator in participant:
            participant[evaluator] = 1 - (np.divide(2, np.pi)) * np.arctan(
                participant[evaluator]
            )
    return xeval


def explode_reput(weights: Dict[str, float], std: float = 0) -> Dict[str, float]:
    """split the reputation score over a wider range

    Args:
        weights (Dict[str,float]): weights taken out of the reputation system.
        std (float): if specified this value is used as the standard deviation of the normal distribution. Other standard deviation of weights is used as a parameter.
    Returns:
        Dict[str,float]: splitted weight, sum is equal to 1
    """

    std = std if std else np.std(list(weights.values()))
    avg = np.average(list(weights.values()))

    avg_gap: Dict[str, Any] = {k: x - avg for k, x in weights.items()}
    cdf: Dict[str, float] = {k: float(norm.cdf(x, 0, std)) for k, x in avg_gap.items()}
    total_w: float = math.fsum(cdf.values())
    return {k: np.divide(x, total_w) for k, x in cdf.items()}


def normalize_matrice(
    xeval: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Normalize matrice
    Args:
        xeval : xevals.
    Return:
        normalized xevals.
    """
    max = xeval_max(xeval)
    assert max > 0
    # Find maximum from xeval
    for participant in xeval.values():
        for evaluator in participant:
            participant[evaluator] = np.divide(participant[evaluator], max)
    return xeval


def reverse_matrice(xeval: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Substract the current matrice from a matrice filled with 1's
    Args:
        xeval : xevals.
    Return:
        reversed xevals.
    """
    for participant in xeval.values():
        for evaluator in participant:
            participant[evaluator] = 1 - participant[evaluator]
    return xeval


def get_sub_xevals(
    cluster: List[str], xevals: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Extract a sub portion of xevals containing only cluster participants in primary key"""
    sub_xevals = {}
    for p_key in cluster:
        sub_xevals[p_key] = xevals[p_key]
    return sub_xevals


def centroid_from_cluster(
    cluster: List[str], xevals: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute centroid from a specific cluster
    Args:
        xevals : cross evaluation results for a single round
        cluster : list of participants, all participants form should be in xevals.
    Return:
        Mean xeval from cluster
    """
    size = len(cluster)
    centroid = {}
    for rated_participant in xevals:
        centroid[rated_participant] = 0
        for cluster_participant in cluster:
            centroid[rated_participant] += xevals[cluster_participant][
                rated_participant
            ]
        centroid[rated_participant] = centroid[rated_participant] / size
    return centroid


class InaproriateXevals(Exception):
    "Raised when discretize xevals are all equal to zero inside a cluster"
    pass
