import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.common.typing import Config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from trustfids.client import (
    Dataset,
    FIDSClient,
    VerbLevel,
    create_model,
    load_data,
    load_siloed_data,
    metrics_from_preds,
    root_mean_squared_error,
)
from trustfids.server.strategy import MetricDict
from trustfids.utils.decorators import tf_setup

keras = tf.keras
from keras.losses import MeanSquaredError
from keras.models import Sequential

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


@tf_setup
def eval_attack_detection(cfg: DictConfig, silos: List[Path] | None = None) -> Dict:
    """Evaluate the attack detection capabilities of the model.

    Args:
        cfg (DictConfig): Hydra configuration

    Returns:
        Dict: Dictionary containing the metrics
    """
    # Load the datasets
    path_silos = list(
        silos
        or map(
            lambda s: Path(__file__).parent.parent / s,
            cfg.fl.silos,
        )
    )
    logger.info(f"Loading datasets:\n+ { (chr(10)+'+ ').join(map(str, path_silos)) }")  # type: ignore

    datasets = [load_data_for_eval(p, seed=cfg.xp.seed) for p in path_silos]
    logger.info(f"Loaded {len(datasets)} datasets.")
    ret_metrics = {}

    logger.info("Creating model.")
    model: Sequential = create_model(
        datasets[0][0][0].shape[1], loss_fn=MeanSquaredError()
    )
    weights = model.get_weights()

    # Train a model for each dataset
    for s, d in zip(path_silos, datasets):
        logger.info(f"Training model on {s.stem}...")
        train_set, test_set, stats = d
        ret_metrics[s.stem] = {
            "stats": stats,
            "eval": eval_on_dataset(
                weights, Dataset(*train_set), Dataset(*test_set), cfg, s.stem
            ),
        }
        logger.info(f"Done training model on {s.stem}.")

    if len(datasets) > 1:
        # contatenate all the datasets
        X_train: pd.DataFrame = datasets[0][0][0]
        y_train: pd.DataFrame = datasets[0][0][1]
        meta_train: pd.DataFrame = datasets[0][0][2]
        X_test: pd.DataFrame = datasets[0][1][0]
        y_test: pd.DataFrame = datasets[0][1][1]
        meta_test: pd.DataFrame = datasets[0][1][2]

        logger.info(f"Preparing merged dataset...")

        for d in datasets[1:]:
            X_train = pd.concat([X_train, d[0][0]])
            y_train = pd.concat([y_train, d[0][1]])
            meta_train = pd.concat([meta_train, d[0][2]])
            X_test = pd.concat([X_test, d[1][0]])
            y_test = pd.concat([y_test, d[1][1]])
            meta_test = pd.concat([meta_test, d[1][2]])

        X_train, y_train, meta_train = shuffle(X_train, y_train, meta_train, random_state=cfg.xp.seed)  # type: ignore
        X_test, y_test, meta_test = shuffle(X_test, y_test, meta_test, random_state=cfg.xp.seed)  # type: ignore

        logger.info("Training model on all datasets...")
        ret_metrics["all"] = {
            "stats": "---",
            "eval": eval_on_dataset(
                weights,
                Dataset(X_train, y_train, meta_train),
                Dataset(X_test, y_test, meta_test),
                cfg,
                "all",
            ),
        }

    return {
        "epochs": cfg.fl.num_epochs * cfg.fl.num_rounds,
        "batch_size": cfg.fl.batch_size,
        "runs": ret_metrics,
    }


def eval_on_dataset(
    init_weights: List[np.ndarray],
    train: Dataset,
    test: Dataset,
    cfg: DictConfig,
    dataset_name: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Evaluate a model on a single dataset"""

    c: FIDSClient = instantiate(
        cfg.client, f"client", train, test, verbosity=VerbLevel.INPLACE
    )

    metrics: List[Tuple[int, Dict[str, Any]]] = []

    for i in range(1, cfg.fl.num_rounds + 1):

        logger.info(f"Round {i}: training...")

        trained_model, train_samples, infos = c.fit(
            init_weights,  # Initial parameters
            {
                "round": i,
                # Each model is trained for n epochs * m rounds in FL
                "epochs": cfg.fl.num_epochs,
                "batch_size": cfg.fl.batch_size,
            },
        )

        logger.info(f"Round {i}: evaluating...")

        loss, test_samples, c_metrics = c.evaluate(
            trained_model,
            {
                "round": i,
                "num_rounds": cfg.fl.num_rounds,
                "batch_size": cfg.fl.batch_size,
            },
        )

        for met, val in c_metrics.items():
            if met == "attack_stats":
                c_metrics[met] = json.loads(str(val))

        init_weights = trained_model

        metrics.append((i, c_metrics))

    return metrics

    # _state = json.loads(str(infos["_state"]))
    # threshold = _state["threshold"]

    # model = create_model(train[0].shape[1], loss_fn=MeanSquaredError())
    # model.set_weights(trained_model)

    # # Evaluate the model on the test set
    # X_test, y_test, meta_test = test
    # inferences = model.predict(X_test, verbose=VerbLevel.INPLACE)
    # losses = mean_squared_error(pd.DataFrame(X_test), inferences)

    # y_eval = np.array(losses > threshold, dtype=int)

    # mean_loss = np.mean(losses)
    # metrics = metrics_from_preds(y_test, y_eval)  # type: ignore
    # metrics.update({"loss": float(mean_loss)})

    # attack_stats = []
    # for attack in (a for a in meta_test["Attack"].unique() if a != "Benign"):
    #     count = len(meta_test[meta_test["Attack"] == attack])
    #     correct = len(meta_test[(meta_test["Attack"] == attack) & (y_test == y_eval)])
    #     missed = len(meta_test[(meta_test["Attack"] == attack) & (y_test != y_eval)])

    #     attack_stats.append(
    #         {
    #             "attack": attack,
    #             "count": count,
    #             "correct": correct,
    #             "missed": missed,
    #             "accuracy": f"{correct / count:.2%}",
    #         }
    #     )

    # return {
    #     "dataset": dataset_name,
    #     "train_samples": train_samples,
    #     "test_samples": len(y_test),
    #     "metrics": metrics,
    #     "attack_stats": attack_stats,
    # }


def load_data_for_eval(
    path: Path, seed: int | None = None, test_size: float = 0.2
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Dict,
]:
    """Load the data for evaluation.

    Data is here processed differently than for the FL experiments. The goal is to be
    able to count how many samples of each attack class are correctly classified by the
    model.

    Args:
        path (Path): Path to the dataset
        seed (int | None): Seed for the random generator. Defaults to None.

    Returns:
        DataFrame: Training set
        DataFrame: Testing set
        Dict: Dictionary containing the number of samples for each class
    """

    df = pd.read_csv(path.as_posix(), low_memory=True)
    cols = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
        "Label",
        "Attack",
    ]
    meta_cols = ["Attack"]

    if "Dataset" in df.columns:
        cols.append("Dataset")
        meta_cols.append("Dataset")

    X = df.drop(columns=cols)
    y = df["Label"]

    meta = df[meta_cols]

    # convert classes to numerical values
    X = pd.get_dummies(X)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X[X.columns] = scaler.transform(X)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame

    # Using `stratify=meta["Attack"]` ensures that the same proportion of samples
    # from each class is present in the training and testing sets. This is important
    # for the evaluation of the attack detection capabilities of the model, as we
    # want to be able to count how many samples of each attack class are correctly
    # classified by the model.
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=seed, stratify=meta["Attack"]
    )  # type: ignore

    # Get the number of samples for each class
    attack_train_samples = meta_train["Attack"].value_counts().to_dict()
    attack_test_samples = meta_test["Attack"].value_counts().to_dict()

    # Filter out malicious samples from the training set
    benign = y_train == 0
    X_train = X_train[benign]
    y_train = y_train[benign]
    meta_train = meta_train[benign]

    return (
        (X_train, y_train, meta_train),
        (X_test, y_test, meta_test),
        {
            "attack_train_samples": attack_train_samples,
            "attack_test_samples": attack_test_samples,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
    )
