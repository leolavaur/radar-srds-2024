""" similarity functions
"""
from typing import Dict, List

import numpy as np

from .utils import *


def similarity_to_centroid(
    p_evals: Dict[str, float], centroid: Dict[str, float]
) -> float:
    """Similarity of a participant to it' centroid.

    Args:
        p_evals: Dict[str, float] Dict of evaluations emitted by a specific participant.
        centroid: Dict[str, float] Dict of mean evaluations from the cluster on every participants.

    Return:
        float : standard deviation used as similarity for the two evaluations vector.
    """
    diff = 0
    for p_id in p_evals.keys():
        diff += (abs(p_evals[p_id] - centroid[p_id])) ** 2
    return 1 - np.sqrt((diff / len(p_evals)))


def similarity_participants(
    p1: str, p2: str, xevals: Dict[str, Dict[str, float]]
) -> float:
    """Similarity score of two participants.
    Similarity is based on the standard deviation of their evaluations
    for every participants (not only clusters member).

    Args:
        p1:
            string : the id of the first participant ex : "cid1"
        p2:
            string : the id of the second participant ex : "cid2"
        xevals:
            Dict of dict containing the results of the cross evaluation.

    Return:
        float : similarity of the two participants, the higher the more similar.

    """
    diff = 0
    for p in xevals[p1]:
        diff += (abs(xevals[p1][p] - xevals[p1][p])) ** 2
    return 1 - np.sqrt((diff / len(xevals)))


#  possible d'inclure une fonction dans les params pour faire varier le param de similarité ?
def similarity_for_cluster_members(
    cluster: List[str], xevals: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute participants similarity
    Args :

    Return :
        Similarity of every other participant to their centroid.
    """
    sim = {}
    centroid = centroid_from_cluster(cluster, xevals)
    # sub_xevals = get_sub_xevals(cluster,xevals)
    for p_key in cluster:
        sim[p_key] = similarity_to_centroid(xevals[p_key], centroid)
    total = sum(sim.values())

    for p_key in sim:
        sim[p_key] = sim[p_key] / total
    return sim
