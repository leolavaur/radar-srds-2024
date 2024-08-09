"""Utility functions for the server."""

import json
import math
from copy import deepcopy
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import scipy as sp
import sklearn.metrics.pairwise as smp
from flwr.common.typing import Metrics as MetricDict
from flwr.common.typing import NDArrays, Scalar
from numpy.typing import ArrayLike, NDArray


def flatten_model(model: List[NDArray]) -> NDArray:
    """Flatten the model into a 1D array.

    Arguments:
        model: A list of numpy arrays.

    Returns:
        The flattened model.
    """
    return np.concatenate([layer.ravel() for layer in model])


def euclidean_matrix(M: List[np.ndarray]) -> np.ndarray:
    """Compute the euclidian distance matrix between the clients' models.

    Euclidean distance is computed as in: https://stackoverflow.com/a/1401828

    Arguments:
        M: A list of clients' models.

    Returns:
        The euclidian distance matrix.
    """
    return np.array([[np.linalg.norm(x - y) for x in M] for y in M])


def cosine_matrix(M: List[np.ndarray]) -> np.ndarray:
    """Compute the cosine distance matrix between the clients' models.

    Arguments:
        M: A list of clients' models.

    Returns:
        The cosine distance matrix.
    """
    return np.array([[sp.spatial.distance.cosine(x, y) for x in M] for y in M])


def evalres_to_dict(evalres: Dict[str, Scalar]) -> Dict[str, Dict[str, float]]:
    """Convert global evaluation results to a per-client evaulation.

    In cross-evaluation, we exploit facilities in Flower to evaluate the clients' models
    on the others' data. However, the evaluation results are limited to a metric
    dictionary of the form {"<metric>": float}. Therefore, when clients evaluate their
    peers, they return a dictionary of the form {"<cid>:<metric>": float}. This function
    converts this dictionary to a dictionary of the form {"<cid>": {"<metric>": float}}.

    Arguments:
        evres: The evaluation results for all clients, of the form {"<cid>:<metric>": float}.

    Returns:
        A dictionary of the form {"<cid>": {"<metric>": float}}.
    """

    evalres_dict: Dict[str, Dict[str, float]] = {}
    for k, v in evalres.items():
        cid, metric = k.split(":")
        if cid not in evalres_dict:
            evalres_dict[cid] = {}
        evalres_dict[cid][metric] = float(v)
    return evalres_dict


def evalres_to_xeval(evalres: Dict[str, Scalar], metric: str) -> Dict[str, float]:
    """Convert global evaluation results to a per-client evaulation.

    In cross-evaluation, we exploit facilities in Flower to evaluate the clients' models
    on the others' data. However, the evaluation results are limited to a metric
    dictionary of the form {"<metric>": float}. Therefore, when clients evaluate their
    peers, they return a dictionary of the form {"<cid>:<metric>": float}. This function
    filters the results to a dictionary of the form {"<cid>": float}.

    Arguments:
        evres: The evaluation results for all clients, of the form {"<cid>:<metric>": float}.

    Returns:
        A dictionary of the form {"<cid>": {float}.
    """

    evalres_dict: Dict[str, float] = {}
    for k, v in evalres.items():
        cid, m = k.split(":")
        if m == metric:
            if metric == "loss":
                # If the metric is loss, we want to project it to [0, 1], where 1 is the
                # best value. This way, we can use the same aggregation function as for
                # accuracy or other similar metrics.
                # See justification in the paper or on #92.
                evalres_dict[cid] = 1 - (np.divide(2, np.pi)) * np.arctan(float(v))
            else:
                evalres_dict[cid] = float(v)

    return evalres_dict


def weighted_average(models_weights: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Compute the weighted average of the models.

    Arguments:
        models_weights: A list of tuples of the form (model, weight).

    Returns:
        The weighted average of the models, layer by layer.
    """
    models: List[NDArrays]
    weights: List[float]

    # unzip models and weights
    # e.g. [(model1, weight1), (model2, weight2)] -> ([model1, model2], [weight1, weight2])
    models, weights = map(list, zip(*models_weights))

    # Commented out because unnecessary with np.average
    # if not math.isclose(math.fsum(weights), 1.0):
    #     raise ValueError("Sum of weights must be close to 1")

    # group layers together (e.g. [[w1, b1], [w2, b2]] -> [[w1, w2], [b1, b2]])
    layers = list(zip(*models))

    # compute the weighted average of each layer
    layers_avg = [np.average(layer, axis=0, weights=weights) for layer in layers]

    return layers_avg


def zipd(*dcts: Dict[str, Any]) -> Generator[Tuple[Any, ...], Dict[str, Any], None]:
    """Zip dictionaries on their common keyes.

    Arguments:
        d: A list of dictionaries.

    Returns:
        A generator that yields tuples of the form (key, [value1, value2, ...]).
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def save_xeval(M: Dict[str, Dict[str, float]], round: int, metric) -> None:
    """Save the cross-evaluation results to a JSON file.

    Arguments:
        M: The cross-evaluation results.
        round: The round number.
    """
    with open(f"xeval_{metric}_{round}.json", "w") as f:
        m = deepcopy(M)
        # sort keys on every level
        for k, v in m.items():
            m[k] = dict(sorted(v.items()))
        m = dict(sorted(m.items()))
        json.dump(m, f, indent=4)


def load_json_metrics(input: MetricDict) -> Dict[str, Any]:
    """Load a MetricDict with JSON-serialized values recursively.

    Arguments:
        input: A MetricDict with JSON-serialized values.

    Returns:
        A Python dictionary with the deserialized values.
    """
    output: Dict[str, Any] = {}
    for k, v in input.items():
        if isinstance(v, dict):
            output[k] = load_json_metrics(v)
        else:
            try:
                output[k] = json.loads(str(v))
            except json.JSONDecodeError:
                output[k] = v

    return output


def foolsgold(grads: NDArray) -> NDArray:
    """FoolsGold algorithm.

    The content of this function is based on the original implementation of FoolsGold
    devilvered by the authors of the paper. The function is only slightly modified to
    provide explicit typing annotations.

    Link to FoolsGold's repository:
    https://github.com/DistributedML/FoolsGold/blob/master/deep-fg/fg/foolsgold.py

    Arguments:
        grads (NDArray): A list of historically aggregated gradients, each gradient
            being a list of layers as numpy arrays. Unlike in the original
            implementation, gradients here are the difference between w_i^r and
            w_0^{r-1}, not the gradients themselves.

    Returns:
        A list of weights, one for each client. The sum of the weights must be 1.
    """

    n_clients = grads.shape[0]
    cs: NDArray = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs: NDArray = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv: NDArray = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = 0.99

    # Logit function
    wv = np.log(wv / (1 - wv)) + 0.5
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv
