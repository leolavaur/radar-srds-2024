# Prototypes for Trust-FIDS
# Author: L. Lavaur & PM. Lechevalier
#
# This file should not be run directly, as it only contains prototypes for
# standardization purposes.
#
# Authors shall update the prototypes when they want
# to change the API of their functions, and assign the others in a Gitlab issue
# for review.

from typing import Dict, List, Tuple


# Module: clustering
def build_clusters(xevals: Dict[str, Dict[str, float]]) -> List[List[str]]:
    """Build clusters from a dictionary of evaluations.

    Cross-evaluations are stored in a dictionary of dictionaries, where the keys
    represent client IDs (cid), and the values are dictionaries of evaluations.
    This structure is comparable to a square matrix, where the rows and columns
    are the clients, and the values are the evaluations.

    Args:
        xevals: A dictionary of evaluations, with the following structure:
            {
                "cid1": {
                    "cid1": 0.7,
                    "cid2": 0.6,
                    ...
                },
                "cid2": {
                    "cid1": 0.8,
                    "cid2": 0.5,
                    ...
                },
                ...
            }

    Returns:
        A list of clusters, with the following structure:
            [
                ["cid1", "cid2", ...],
                ["cid3", "cid4", ...],
                ...
            ]
    """


# Signature for reputation.
class ReputationEnvironment:
    """
    Create the reputation environment and handle it's interactions
    Workflow :
        -> add_round
        -> weight_votes
    """

    def __init__(self, class_nb: int = 10) -> None:
        """
        Args:
            class_nb : number of class for discretization.
        """

    def new_round(
        self, clusters: List[List[str]], xevals: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Add a new round to the reputation environment.

        Args:
            cluster:
                List[string] : every participants id from cluster members:
                    [
                        "cid1","cid3","cid7"
                    ]
            xevals:
                cross evaluation results for the current round
        """

    def compute_cluster_weights(
        self, cluster: List[str], round_nb: int = None
    ) -> Dict[str, float]:
        """Compute weights for a cluster at a specific round
        Args:
            cluster: cluster to compute the weights on.
            round_nb: weights for this round are computed, default to current round.
        Return:
            Dict[str, float] : aggregtion weights for cluster members:
            {
                "cid1" : 0.2,
                "cid3" : 0.5,
                "cid7" : 0.3,
            }
        """
