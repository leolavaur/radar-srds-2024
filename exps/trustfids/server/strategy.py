"""Server module."""

import json
import math
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
import scipy as sp
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import Metrics as MetricDict
from flwr.common.typing import FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from numpy.typing import ArrayLike, NDArray
from omegaconf import DictConfig

from trustfids.clustering.cluster import build_clusters
from trustfids.reputation.reput import DirichletReputationEnvironment
from trustfids.utils.log import logger

from .utils import (
    cosine_matrix,
    euclidean_matrix,
    flatten_model,
    foolsgold,
    load_json_metrics,
    save_xeval,
    weighted_average,
    zipd,
)

# Typings
# -------

# fmt: off
Metric = str                        # Metric name, eg. "loss"
JSONMetrics = str                   # JSON-serialized metrics dictionary
ClientID = str                      # Client ID, eg. "client_1"
ClusterID = str                     # Cluster ID, eg. "cluster_1"
TaskID = ClientID | ClusterID       # Task ID, eg. "client_1" or "cluster_1"
# fmt: on

# --------------------------------------------------------------------------------------
#   Federated Learning simulation strategy
# --------------------------------------------------------------------------------------


class PersistantFedAvg(FedAvg):
    """Simulation FedAvg strategy for Flower.

    This strategy allows to maintain persistent client state between rounds, by hooking
    all functions of the base Strategy class. This is useful for simulation purposes,
    where we want to maintain a client's state between rounds. When using the base
    Strategy class, clients are destroyed after each round and a new client is created
    for the next round. This makes it impossible to maintain persistent client state.

    This class is of no use for real-deployment settings, where clients are not
    destroyed between rounds. However, it is still possible to use this class in such
    settings, as it does not change the behavior of the base FedAvg class, except for a
    small computational overhead (unmeasured as of writing).
    """

    # Client state
    # ------------
    # This is a dictionary that contains the client's state. State is a JSON-serialized
    # dictionary that contains informations that the client needs to maintain between
    # rounds. For example, a threshold value that is generated during training and used
    # during evaluation.
    _states: Dict[str, Scalar]

    def __init__(
        self,
        *args,
        num_epochs: int,
        batch_size: int,
        initial_parameters: Parameters,
        num_rounds: int,
        **kwargs,
    ) -> None:
        """Initialize simulation strategy."""
        super().__init__(
            *args,
            on_fit_config_fn=self.on_fit_config_fn,
            on_evaluate_config_fn=self.on_evaluate_config_fn,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
            **kwargs,
        )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.thresholds = {}
        self._states: Dict[str, Scalar] = {}
        self.global_model = parameters_to_ndarrays(initial_parameters)
        self.model_length = len(self.global_model)

        self.num_rounds = num_rounds

    def on_fit_config_fn(self, rnd: int) -> dict:
        """Return a dictionary with the configuration for the current training round."""
        return {
            "round": rnd,
            "epochs": self.num_epochs,
            "batch_size": self.batch_size,
        }

    def on_evaluate_config_fn(self, rnd: int) -> dict:
        """Return a dictionary with the configuration for the current evaluation round."""
        return {
            "round": rnd,
            "num_rounds": self.num_rounds,
            "batch_size": self.batch_size,
        }

    def fit_metrics_aggregation_fn(
        self, metrics: List[Tuple[int, MetricDict]]
    ) -> MetricDict:
        """Aggregate metrics from all clients."""
        # Way this function is called:
        # ----------------------------
        # metrics_aggregated = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        return {str(m["_cid"]): m["loss"] for _, m in metrics}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the results of the current training round; persist client state."""

        # loop over all clients and save their state in the `states` dictionary
        # for client, result in results:
        #     self._states[client.cid] = result.metrics["_state"]

        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the evaluation of the current round; persist client state."""

        ins = super().configure_evaluate(server_round, parameters, client_manager)

        # Make deep copies of the evaluation instructions, so that they can be modified
        # afterwards. By default, Flower reuses the same evaluation instructions for
        # all clients, and the same object is passed.
        # TODO: open an issue or a PR to Flower to make EvaluateIns endepent objects.
        return [(client, deepcopy(evaluate_ins)) for client, evaluate_ins in ins]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        client_loss = []
        metrics_aggregated: Dict[str, Scalar] = {}
        for client, evaluate_res in results:
            c_metrics = evaluate_res.metrics
            if "attack_stats" in c_metrics:
                c_metrics["attack_stats"] = json.loads(str(c_metrics["attack_stats"]))
            metrics_aggregated[client.cid] = json.dumps(c_metrics)
            client_loss.append((evaluate_res.num_examples, c_metrics["loss"]))

        return weighted_loss_avg(client_loss), metrics_aggregated


# --------------------------------------------------------------------------------------
#   Federated cross-evaluation strategy
# --------------------------------------------------------------------------------------


class FedXeval(PersistantFedAvg):
    """Federated cross-evaluation strategy for Flower.

    This strategy allows to perform cross-evaluation between clients. It is based on
    the FedAvg strategy, but overrides the `aggregate_fit` and `configure_evaluate`
    functions. `aggregate_fit` now returns all client models, concatenated into a
    single list of `ndarrays`. `configure_evaluate` passes this list of models to each
    client, so they can evaluate each other's models.

    This strategy also adds the `aggregate_fit_evaluate` function, which is called
    after the evaluation step. This function aggregates the model parameters depending
    on cross-evaluation results, using a reputation system.
    """

    # CID list of clients in the current round
    client_order: List[str]

    # cluster for model distribution and aggregation
    clusters: OrderedDict[str, List[str]]

    # round tracking for stateful `configure_evaluate` method
    evalround: int

    # model length of the initial parameters (in ndarrays)
    model_length: int

    def __init__(
        self,
        *,  # enforce keyword arguments, even if there are no default values
        initial_parameters: Parameters,
        reputation_system: DirichletReputationEnvironment | None = None,
        evaluation_metric: str = "accuracy",
        clustering_metric: Optional[str] = None,
        reputation_metric: Optional[str] = None,
        clustering_kwargs: Optional[Dict[str, Any]] = None,
        distribution: Optional[OrderedDict[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        """Initialize simulation strategy.

        Arguments:
            initial_parameters: Initial model parameters, mandatory.
            reputation: Reputation of each client.
            evaluation_metric: Evaluation metric to use for cross-evaluation. One of
                "loss", "accuracy", "precision", "recall", "f1", "mcc", "auc",
                "missrate", "fallout".
            **kwargs: arguments passed to `FedAvg`.
        """
        assert initial_parameters is not None

        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.perfect_distribution = None
        if clustering_kwargs and "perfect" in clustering_kwargs:
            if clustering_kwargs["perfect"]:
                self.perfect_distribution = distribution
            del clustering_kwargs["perfect"]

        self.RS = reputation_system
        self.clustering_metric = clustering_metric or evaluation_metric
        self.reputation_metric = reputation_metric or evaluation_metric
        self.clustering_kwargs = clustering_kwargs
        self.model_length = len(parameters_to_ndarrays(initial_parameters))
        self.clusters = OrderedDict()
        self.client_order = []
        self.xevalstep = True  # flag to indicate if we are in the cross-evaluation step

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        This function is called by the server before the training round starts. It
        generates fit instructions for each client. In the case of Trust-FIDS, because
        we use a clustering strategy for aggregation, this function goes over the
        `clusters` OrderedDict to determine to which client to send which model,
        depending on the current round cluster assignments.

        Arguments:
            server_round: Current round number.
            parameters: Current model parameters.
            client_manager: Client manager.

        Returns:
            List of (client, fit_ins) tuples.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # If there are no clusters yet, create a single cluster with all clients
        if len(self.clusters.keys()) == 0:
            self.clusters["cluster_0"] = [c.cid for c in clients]

        client_fitins = []

        for i, kv in enumerate(self.clusters.items()):
            kid, k_clients = kv
            offset = i * self.model_length  # 0 if 1 cluster; clients get the same model

            # extract model parameters for the cluster
            np_parameters = parameters_to_ndarrays(parameters)
            cluster_np_parameters = np_parameters[offset : offset + self.model_length]
            cluster_parameters = ndarrays_to_parameters(cluster_np_parameters)

            # reference issue from flower (cf. super().configure_evaluate)
            fit_ins = deepcopy(FitIns(cluster_parameters, config))

            for cid in k_clients:
                # Find the ClientProxy with the matching `cid`
                proxy = next(filter(lambda c: c.cid == cid, clients))
                # Add the client and their `fit_ins` to `client_fitins`
                client_fitins.append((proxy, fit_ins))

        # Return client/config pairs
        return client_fitins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using model chaining.

        Typical aggregation functions will aggregate the model parameters of each client
        and return the aggregated parameters. However, in the case of cross-evaluation,
        aggregation should only be performed at the end of the round, after federated
        evaluation.

        This function instead returns all client models, concatenated into a single
        list of `ndarrays`. The aggregated parameters will be distributed as-is to all
        clients.

        Arguments:
            server_round: The current round of the server.
            results: A list of tuples containing the client and the result of
                the fit operation.
            failures: A list of tuples containing the client and the result of
                the fit operation or the exception raised during the fit
                operation.

        Returns:
            A tuple containing the concatenated parameters and the metrics.
        """
        # Call super method to save client states
        _, _ = super().aggregate_fit(server_round, results, failures)

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Save client order in which the results were received.
        # It will be used by `configure_evaluate` to determine the order in which
        # clients should review the models.
        self.client_order = []
        client_models = []
        for client, res in results:
            self.client_order.append(client.cid)
            client_models.append(parameters_to_ndarrays(res.parameters))

        # Aggregate parameters
        parameters_aggregated: Parameters = ndarrays_to_parameters(
            list(chain(*client_models))
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Generate evalation instructions for the current round.

        Unlike the base Strategy class, this function considers the global model
        parameter as a concatenated list of models, either one per client or one per
        cluster, depending on the current step.

        Arguments:
            server_round: The current round of the server.
            parameters: The current global model parameters.
            client_manager: The client manager.

        Returns:
            A list of tuples containing a client and the evaluation instructions.
        """
        ins = super().configure_evaluate(server_round, parameters, client_manager)

        # Get cluster models
        np_parameters = parameters_to_ndarrays(parameters)
        cluster_models: Dict[str, NDArrays] = {}
        for i, kv in enumerate(self.clusters.items()):
            kid, _ = kv
            offset = i * self.model_length
            cluster_models[kid] = np_parameters[offset : offset + self.model_length]

        # loop over all clients and add their evaluation tasks to the `EvaluateIns` object
        for prxy, evaluate_ins in ins:
            evaluate_ins.config.update(
                {
                    "model_length": self.model_length,
                    # "tasks": json.dumps(list(self.clusters.keys())),
                }
            )

            # find the cluster to which the client currently belongs
            kid = ""
            for k, cluster in self.clusters.items():
                if prxy.cid in cluster:
                    kid = k
            if not kid:
                raise ValueError("Client not found in any cluster")

            evaluate_ins.parameters = ndarrays_to_parameters(cluster_models[kid])

        return ins

    def configure_crossevaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Generate cross-evalation instructions for the current round.

        Unlike the base Strategy class, this function considers the global model
        parameter as a concatenated list of models: one per client.

        Parameters:
        -----------
        server_rount : int
            The current round of the server.
        parameters : Parameters
            The current global model parameters. A concatenated list of models, one per
            client.
        client_manager : ClientManager
            The client manager.

        Returns:
        --------
        List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples containing a client and the evaluation instructions for
            each client.
        """
        ins = super().configure_evaluate(server_round, parameters, client_manager)

        for _, evaluate_ins in ins:
            evaluate_ins.config.update(
                {
                    "model_length": self.model_length,
                    "tasks": json.dumps(self.client_order),
                }
            )

        return ins

    def aggregate_fit_evaluate(
        self,
        server_round: int,
        fit_results: List[Tuple[ClientProxy, FitRes]],
        fit_failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        evaluate_results: List[Tuple[ClientProxy, EvaluateRes]],
        evaluate_failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using cross-evaluation results.

        This function does not exist in the base Strategy class. It is used to aggregate
        the training (fit) results using the cross-evaluation. To be used, this function
        requires a specific instance of `flwr.server.Server`.

        See `trustfids.server.XevalServer` for more details.

        Arguments:
            server_round: The current round of the server.
            fit_results: results of the fit function.
            fit_failures: failures of the fit function.
            evaluate_results: results of the evaluate function.
            evaluate_failures: failures of the evaluate function.

        Returns:
            A tuple containing the aggregated parameters and the aggregated metrics.
        """

        model_updates = {
            proxy.cid: parameters_to_ndarrays(fit_res.parameters)
            for proxy, fit_res in fit_results
        }

        # Aggregated metrics are of form `{client_id: {task_id: {metric_name: value}}}`,
        # in which `task_id` is another client's cid.
        extracted_metrics: Dict[ClientID, Dict[TaskID, Dict[Metric, float]]] = {
            proxy.cid: load_json_metrics(eval_res.metrics)
            for proxy, eval_res in evaluate_results
        }

        # CLUSTERING
        M_clust: Dict[ClientID, NDArray] | Dict[ClientID, Dict[ClientID, float]] = {}

        if (
            self.clustering_kwargs
            and self.clustering_kwargs.get("input_type") == "models"
        ):
            M_clust = {
                cid: np.concatenate([layer.ravel() for layer in model])
                for cid, model in model_updates.items()
            }

        else:
            # Build cross-evaluation matrix, filtering metrics
            M_clust = {
                cid_i: {
                    cid_j: eval_for_j[self.clustering_metric]
                    for cid_j, eval_for_j in evals_from_i.items()
                }
                for cid_i, evals_from_i in extracted_metrics.items()
            }

            # Here we can modify M_clust depending on the attacker behavior.
            save_xeval(M_clust, server_round, self.clustering_metric)

        if self.perfect_distribution:
            self.clusters = self.perfect_distribution
        elif self.clustering_kwargs:
            self.clusters = OrderedDict(
                {
                    f"cluster_{i}": c
                    for i, c in enumerate(
                        build_clusters(deepcopy(M_clust), **self.clustering_kwargs)  # type: ignore
                    )
                }
            )
        else:
            self.clusters = OrderedDict(
                {
                    f"cluster_{i}": c
                    for i, c in enumerate(default_cluster_fn(deepcopy(M_clust)))  # type: ignore
                }
            )

        # REPUTATION SYSTEM

        # Build cross-evaluation matrix, filtering metrics
        M_rep: Dict[ClientID, Dict[ClientID, float]] = {
            cid_i: {
                cid_j: eval_for_j[self.reputation_metric]
                for cid_j, eval_for_j in evals_from_i.items()
            }
            for cid_i, evals_from_i in extracted_metrics.items()
        }
        save_xeval(M_rep, server_round, self.reputation_metric)

        # default weights
        sum_examples = sum(n.num_examples for _, n in fit_results)

        default_weights: Dict[str, float] = {
            #    proxy.cid: n.num_examples / sum_examples for proxy, n in fit_results
            proxy.cid: n.num_examples
            for proxy, n in fit_results
        }

        if self.RS is not None:
            self.RS.new_round(list(self.clusters.values()), deepcopy(M_rep))

        cluster_models: List[NDArrays] = []

        for kid, cluster in self.clusters.items():
            if self.RS is not None:
                new_weights: dict[str, float] = self.RS.compute_cluster_weights(cluster)
            else:
                new_weights = {cid: default_weights[cid] for cid in cluster}
                # project weights to that they sum to 1
                new_weights = {
                    cid: weight / sum(new_weights.values())
                    for cid, weight in new_weights.items()
                }

            # Map model updates to their new weights (cid, model, weight)
            cmw: List[Tuple[str, NDArrays, float]] = list(
                zipd(model_updates, new_weights)
            )
            weights_results: List[Tuple[NDArrays, float]] = [
                (model, weight) for cid, model, weight in cmw if cid in cluster
            ]

            cluster_models.append(weighted_average(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in fit_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        parameters_aggregated = ndarrays_to_parameters(list(chain(*cluster_models)))

        return parameters_aggregated, metrics_aggregated

    # def _get_cluster(self, cid: str) -> Tuple[str, List[str]]:
    #     """Get the cluster of a client.

    #     Arguments:
    #         cid (str): Client id.

    #     Returns:
    #         kid (str): Cluster ID of the form "cluster_{i}".
    #         cluster (List[str]): List of client ids in the cluster.
    #     """
    #     for kid, cluster in self.clusters.items():
    #         if cid in cluster:
    #             return kid, cluster
    #     raise ValueError(f"Client {cid} not found in any cluster.")


# --------------------------------------------------------------------------------------
#   FoolsGold strategy for baseline comparison
# --------------------------------------------------------------------------------------


class FoolsGold(PersistantFedAvg):
    def __init__(self, initial_parameters: Parameters, *args, **kwargs) -> None:
        """Initialize simulation strategy."""

        # Each client has a history of its flattened gradients at each round
        self.history: Dict[str, NDArray] = {}

        super().__init__(initial_parameters=initial_parameters, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit using FoolsGold algorithm.

        FoolsGold

        Arguments:
            server_round: The current round of the server.
            results: A list of tuples containing the client and the result of
                the fit operation.
            failures: A list of tuples containing the client and the result of
                the fit operation or the exception raised during the fit
                operation.

        Returns:
            A tuple containing the aggregated parameters and the metrics.
        """

        # Call super method to save client states
        _, _ = super().aggregate_fit(server_round, results, failures)

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Get results and sort them by client ids
        client_results = [
            (
                proxy.cid,
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            )
            for proxy, fit_res in results
        ]
        client_results.sort(key=lambda x: x[0])

        # Update history
        for cid, m, _ in client_results:
            grads = flatten_model(m) - flatten_model(self.global_model)
            if cid not in self.history:
                self.history[cid] = np.zeros_like(grads)
            self.history[cid] += grads

        # Get a NDArray of shape (num_clients, num_parameters) with flattened models
        model_updates = np.array(
            [g for _, g in sorted(self.history.items(), key=lambda x: x[0])]
        )

        weights = foolsgold(model_updates)

        weights_results = [(p, w) for (_, p, _), w in zip(client_results, weights)]

        agg = aggregate(weights_results)
        self.global_model = agg

        parameters_aggregated = ndarrays_to_parameters(agg)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def _store_gradients(self, updates: Dict[str, NDArrays]) -> None:
        """Compute the gradients of an update.

        Arguments:
            updates: A dictionary containing the client id and the model
                update.

        Returns:
            A dictionary containing the client id and the gradients.
        """
        for cid, m in updates.items():
            grads = flatten_model(m) - self.global_model
            if cid not in self.history:
                self.history[cid] = np.zeros_like(grads)
            self.history[cid] += grads


# --------------------------------------------------------------------------------------
#   Distance-based strategy for baseline comparison
# --------------------------------------------------------------------------------------


class FedDistance(PersistantFedAvg):
    def __init__(self, **kwargs) -> None:
        """Initialize simulation strategy."""

        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average.

        Federated Averaging (FedAvg) is a the original federated learning
        aggregation strategy. This adaptation of FedAvg uses the eigenvalues of
        the covariance matrix of the clients' local models to weight the
        aggregation.

        Arguments:
            server_round: The current round of the server.
            results: A list of tuples containing the client and the result of
                the fit operation.
            failures: A list of tuples containing the client and the result of
                the fit operation or the exception raised during the fit
                operation.

        Returns:
            A tuple containing the aggregated parameters and the metrics.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        models = [flatten_model(m) for m, _ in weights_results]

        # Compute distance matrix
        M: np.ndarray = cosine_matrix(models)  # CHANGE HERE FOR ANOTHER DISTANCE

        # REPUTATION SYSTEM

        # SET WEIGHTS HERE
        weights: List[float] = [1 / len(results) for _ in range(len(results))]

        # COMPUTE NEW WEIGHTS
        weights_results = [
            (p, int(n * w)) for (p, n), w in zip(weights_results, weights)
        ]  # maping to int is a hack to use the aggregate function directly

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# --------------------------------------------------------------------------------------
#   Helper functions
# --------------------------------------------------------------------------------------


def default_cluster_fn(M: Dict[str, Dict[str, float]]) -> List[List[str]]:
    """Cluster clients based on cross-evaluation.

    Arguments:
        M: Cross-evaluation matrix.

    Returns:
        A list of clusters, each cluster being a list of client ids.
    """
    return [list(M.keys())]


# FLOWER DOCUMENTATION REMINDER
# -----------------------------
#
# Bellow are some definitions that are taken from the Flower documentation. This is
# useful to understand the code above, especially the types and classes defined in the
# framework's internals.
#
# * class EvaluateIns(parameters: Parameters, config: Dict[str, Union[bool, bytes,
#   float, int, str]])
# * class FitIns(parameters: Parameters, config: Dict[str, Union[bool, bytes, float,
#   int, str]])
# * class EvaluateRes(status: Status, loss: float, num_examples: int, metrics: Dict[str,
#   Union[bool, bytes, float, int, str]])
# * class FitRes(status: Status, parameters: Parameters, num_examples: int, metrics:
#   Dict[str, Union[bool, bytes, float, int, str]])
# * class Parameters(tensors: List[bytes], tensor_type: str)
#
#
# * fn aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
#    """Compute weighted average."""
#    # Calculate the total number of examples used during training
#    num_examples_total = sum([num_examples for _, num_examples in results])
#
#    # Create a list of weights, each multiplied by the related number of examples
#    weighted_weights = [
#        [layer * num_examples for layer in weights] for weights, num_examples in results
#    ]
#
#    # Compute average weights of each layer
#    weights_prime: NDArrays = [
#        reduce(np.add, layer_updates) / num_examples_total
#        for layer_updates in zip(*weighted_weights)
#    ]
#    return weights_prime


class FedNoAgg(PersistantFedAvg):
    """Configurable FedNoAgg strategy based on FedAvg's implementation."""

    client_models: dict[str, Parameters] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.initial_parameters is None:
            self.initial_parameters = parameters

        # Return client/config pairs
        return [
            (
                client,
                FitIns(
                    self.client_models.get(client.cid, self.initial_parameters), config
                ),
            )
            for client in clients
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [
            (
                client,
                EvaluateIns(
                    self.client_models.get(client.cid, self.initial_parameters), config
                ),
            )
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        for client, fit_res in results:
            self.client_models[client.cid] = fit_res.parameters

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.warn("No fit_metrics_aggregation_fn provided")

        return self.initial_parameters, metrics_aggregated
