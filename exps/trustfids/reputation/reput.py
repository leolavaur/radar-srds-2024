"""Reputation related functions
"""
import math
from copy import deepcopy
from typing import Dict

from trustfids.reputation.reput_logger import log_weights
from trustfids.utils.log import logger

from ..clustering.utils import *

from .abstract import ReputationEnvironment
from .history import *
from .similarity import *
from .utils import *


############
# Réputation
############
class DirichletReputationEnvironment(ReputationEnvironment):
    """
    Create the reputation environment and handle it's interactions
    Workflow :
        -> add_round
        -> weight_votes
    """

    def __init__(
        self,
        class_nb: int = 100,
        lmbd: float = 0.2,
        evaluation_metric: str = "accuracy",
        log=True,
    ):
        """
        Args:
            class_nb : number of class for discretization.
            lmbd : lambda parameter comprised between 0 and 1 that arbitrate the
            prevalence of recent and older evaluation. 1 mean all evaluation have
            equal weight, 0 only take into account the latest evaluation.
            evaluation_metric: metric used for reputation. Must take one of the following value :
                "accuracy": Evaluations returns accuracy
                "loss": Evaluations returns the loss
            log (bool): weights of the clients are printed to a file name "client_weights.json" when true.
        """
        self.class_nb = class_nb
        if lmbd > 1.0 or lmbd < 0.0:
            raise ValueError("Lambda must be in range O.0-1.0")

        if evaluation_metric not in ["accuracy", "loss"]:
            raise ValueError(
                "evaluation_metric must be one of the following:", "accuracy", "loss"
            )
        self.evaluation_metric = evaluation_metric
        self.lmbd = lmbd
        self.hist = Hist()
        self.log = log

    def exponential_decay(self, val: float, elapsed_time: int) -> float:
        """reputation decay
        Args :
            val : value to decay
            elapsed_time : time elapsed
        Return :
            [O-1]: evaluation ponderated with elapsed time,
        """
        return val * (self.lmbd**elapsed_time)

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
        mat = deepcopy(xevals)
        if self.evaluation_metric == "loss":
            # R+ -> [0,1] reversed
            mat = normalize_loss_matrice(mat)

        d_xevals = discretize_matrice(mat, self.class_nb)
        self.hist.add_round(xevals, d_xevals, clusters)

    def compute_cluster_weights(
        self, cluster: List[str], round_nb: int = 0
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

        if round_nb:
            if round_nb >= self.hist.round_counter:
                raise IndexError("Submited round doesn't exist yet")
            round: Round = self.hist.rounds[f"r{round_nb}"]
        else:
            round: Round = self.hist.get_last_round()
        current_round_counter: int = (
            round_nb if round_nb else self.hist.round_counter - 1
        )

        # cluster parameter edge cases
        if len(cluster) == 1:
            client_weight = {}
            client_weight[cluster[0]] = 1.0
            if self.log:
                log_weights(client_weight, self.hist, exploded=True)
                log_weights(client_weight, self.hist)
            return client_weight
        elif len(cluster) == 0:
            raise ValueError("Cluster is empty")

        # for each discretized outcome of each participant
        # d_summed_xevals store the number of evaluations
        # ponderated with history and similarity.
        d_summed_xevals: Dict[str, Dict[int, float]] = {
            p: {i: 0 for i in range(self.class_nb)} for p in cluster
        }

        # populating d_summed_xevals with every round evals
        while round:
            d_xevals = round.normed_xevals
            # similarity is round specific
            sim = similarity_for_cluster_members(cluster, round.xevals)

            # for each round compute ponderated xevals for cluster members
            for rated_p in cluster:
                for rater_p in cluster:
                    score = d_xevals[rater_p][rated_p]

                    # time and similarity ponderation for the evaluation
                    elapsed_time = current_round_counter - round.round_number
                    hist_ponderated = self.exponential_decay(1, elapsed_time)
                    sim_ponderated = hist_ponderated * sim[rater_p]
                    d_summed_xevals[rated_p][score] += sim_ponderated
            round = self.hist.get_previous_round(round.id)

        received_votes = 0
        total_votes = 0
        for client in d_summed_xevals.values():
            received_votes = sum(client.values())
            total_votes += received_votes

        client_reput = {}
        for client in d_summed_xevals:
            client_reput[client] = 0
            for xi, yi in d_summed_xevals[client].items():
                # client_reput[client] += xi*(yi/received_votes)
                client_reput[client] = np.add(
                    client_reput[client],
                    np.multiply(xi, (np.divide(yi, received_votes))),
                )

        total_reput: float = math.fsum(list(client_reput.values()))
        client_weight: Dict[str, float] = {}
        for client in client_reput:
            if total_reput:
                # client_weight[client] = client_reput[client]/ total_reput
                client_weight[client] = np.divide(client_reput[client], total_reput)
            else:
                # case where no cluster member received a vote that pass discretization threshold
                logger.warn(
                    f"All evaluations from cluster participant {cluster} were equal 0"
                )
                client_weight[client] = np.divide(1, len(list(client_reput.keys())))

        if self.log:
            log_weights(client_weight, self.hist, exploded=False)

        client_weight = explode_reput(client_weight, 0.0005)
        if not math.isclose(math.fsum(list(client_weight.values())), 1.0):
            logger.warn(
                f"Compute cluster weight sum differs from 1.0 by more than 1e-09 : {math.fsum(list(client_weight.values()))} "
            )
        if self.log:
            log_weights(client_weight, self.hist, exploded=True)

        return client_weight
