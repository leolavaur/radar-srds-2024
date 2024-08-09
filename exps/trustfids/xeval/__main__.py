"""Cross-evaluation experiment script.

This script is the entry point for everyrhing cross-evaluation related. However, it does
not aim to be used with Hydra, but rather as a standalone script. Its purposes are
testing, debugging, and temporary experiments.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import os  # isort:skip

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import multiprocessing
import pathlib
from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import Scalar
from trustfids.utils.client import FIDSClient
from trustfids.utils.data import Dataset, clients_per_silo, load_siloed_data
from trustfids.utils.decorators import tf_gpu_setup
from trustfids.utils.helpers import unflatten

# type definitions
Parameters = List[np.ndarray]
Evals = Dict[str, Scalar]

ModelMatrix = Dict[str, List[Parameters]]
ThresholdMatrix = Dict[str, List[Scalar]]
DataMatrix = Dict[str, Tuple[Dataset, Dataset]]
XEvalMatrix = Dict[str, Dict[str, Scalar]]


@tf_gpu_setup
def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-evaluation experiment script.")

    parser.add_argument(
        "-p", "--path", type=str, help="path to saved models", required=True
    )
    parser.add_argument(
        "-m",
        "--build-matrix",
        action="store_true",
        help="build cross-evaluation matrix",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="seed for randomisation",
        default=1138,
        action="store",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="output directory",
        action="store",
        default=".",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="path to datasets",
        nargs="+",
        action="store",
        default=[
            "data/reduced/botiot_reduced.csv.gz",
            "data/reduced/cicids_reduced.csv.gz",
            "data/reduced/nb15_reduced.csv.gz",
            "data/reduced/toniot_reduced.csv.gz",
        ],
    )

    args = parser.parse_args()

    if not pathlib.Path(args.outdir).exists():
        pathlib.Path(args.outdir).mkdir(parents=True)
    elif not pathlib.Path(args.outdir).is_dir():
        raise ValueError("Output directory is not a directory")

    if args.build_matrix:
        print("Building cross-evaluation matrix")
        model_matrix, thresholds = load_models(pathlib.Path(args.path), 10, 10)
        # datasets = load_data(pathlib.Path(args.path) / "data.npz")
        silos = args.datasets
        clients = create_clients(10, silos=silos, seed=args.seed)
        for rnd in range(1, 11):
            print("Round", rnd, "-- building matrix")
            rnd_thresholds = {cid: thresholds[cid][rnd - 1] for cid in clients.keys()}
            matrix = build_matrix(rnd, clients, model_matrix, rnd_thresholds)

            print("Saving matrix", rnd, "as", f"matrix_{rnd}.json")
            json.dump(
                matrix, open(pathlib.Path(args.outdir) / f"matrix_{rnd}.json", "w")
            )

    else:
        print("No action specified...")
        exit(1)


def build_matrix(
    rnd: int,
    clients: Dict[str, FIDSClient],
    model_matrix: ModelMatrix,
    thresholds: Dict[str, Scalar],
    metric: str = "accuracy",
) -> XEvalMatrix:
    """Build cross-evaluation matrix.

    This function starts as many cross-evaluation tasks as there are clients. Each task
    consists of evaluating the models of all other clients on the data of the client.
    Each client is started in his own process, so that the evaluation can be done in
    parallel using multiprocessing.

    Args:
        models (ModelMatrix): Dictionary of models for each round for each client.
        thresholds (ThresholdMatrix): Dictionary of thresholds for each round for each client.
        datasets (DataMatrix): Dictionary of datasets for each client.

    Returns:
        XEvalMatrix: Dictionary of dictionaries containing the cross-evaluation results.
    """
    matrix: XEvalMatrix = {}

    for cid, client in clients.items():

        print("Evaluating models with data from", cid)

        matrix[cid] = evaluate_client(
            cid, rnd, client, model_matrix, thresholds[cid], metric
        )

    return matrix


def evaluate_client(
    eval_cid: str,
    rnd: int,
    client: FIDSClient,
    model_matrix: ModelMatrix,
    threshold: float,
    metric: str,
) -> Dict[str, Scalar]:
    """Evaluate a client on others datasets.

    This function evaluates a client on the datasets of other clients. It is used to
    build the cross-evaluation matrix afterward. The function is meant to be run in a
    separate process.

    Args:
        cid (str): Client ID.
        rnd (int): Round.
        clients (Dict[str, FIDSClient]): Dictionary of clients.
        models (ModelMatrix): Dictionary of models for each round for each client.
        threshold (float): Threshold to use for evaluation.
        metric (str): Metric to use for evaluation.

    Returns:
        Dict[str, Scalar]: Dictionary of cross-evaluation results. The keys are the
            client IDs and the values are the evaluation results.
    """
    evals: Dict[str, Scalar] = {}

    for cid, models in model_matrix.items():

        if cid == eval_cid:
            evals[eval_cid] = {  # perfect results for self-evaluation
                "accuracy": 1,
                "precision": 1,
                "recall": 1,
                "f1": 1,
                "mcc": 1,
                "auc": 1,
                "missrate": 0,
                "fallout": 0,
            }[metric]
            continue

        model = models[rnd - 1]
        _, _, results = client.evaluate(model, {"threshold": threshold})

        evals[cid] = results[metric]

    return evals


def load_data(path: pathlib.Path) -> DataMatrix:
    """Load saved data.

    This function loads saved data from a given path. The path should point to a
    folder containing data exported by np.savez. For instance, `data.npz`.

    Args:
        path (str): Path to saved data.

    Returns:
        DataMatrix: Dictionary of datasets for each client.
    """
    assert path.exists(), f"Path {path} does not exist"

    data: DataMatrix = {}

    flat = np.load(path)
    # build back a list of tuples from saved data
    datasets = unflatten(flat, 2)

    for i, dataset in enumerate(datasets):
        data[f"client_{i+1}"] = dataset

    return data


def load_models(
    path: pathlib.Path, n_clients: int, n_rounds: int
) -> Tuple[ModelMatrix, ThresholdMatrix]:
    """Load saved models.

    This function loads saved models from a given path. The path should point to a
    folder containing models exported by the `save_progress` decorator. Files in this
    folder follow the following naming convention:
    ```
    <round>_<client_id>_<n_samples>_<threshold>.npz
    ```
    Example:
    ```
    001_client_3_55673.npz
    ```

    Args:
        path (str): Path to saved models.

    Returns:
        ModelMatrix: Dictionary of saved models.
    """
    assert path.exists(), f"Path {path} does not exist"
    assert (
        len(list(path.glob("*.npz"))) == n_clients * n_rounds
    ), f"Wrong number of files in {path}"

    model_matrix: ModelMatrix = {}
    thresholds: ThresholdMatrix = {}

    for file in path.glob("*.npz"):
        # Extract round and client ID from filename
        # Warning: the client ID might contain underscores
        rnd, *middle, _, threshold = file.stem.split("_")
        cid = "_".join(middle)

        # Load model
        model = list(np.load(file).values())

        # Add model to dictionary
        if cid not in model_matrix:
            model_matrix[cid] = []
            thresholds[cid] = []
        model_matrix[cid].append(model)
        thresholds[cid].append(float(threshold))

    return model_matrix, thresholds


def create_clients(
    n_clients: int, silos: List[str], seed: int
) -> Dict[str, FIDSClient]:

    distribution = clients_per_silo(n_clients, silos)

    datasets = load_siloed_data(silos, distribution, seed)

    cids = [f"client_{i}" for i in range(n_clients)]

    clients = {}
    for i, c in enumerate(cids):
        train, test = datasets[i]
        clients[c] = FIDSClient(c, train, test)

    return clients


if __name__ == "__main__":
    main()
