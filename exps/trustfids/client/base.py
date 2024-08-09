"""Module for base client class."""

import json
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from flwr.client import NumPyClient
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig
from trustfids.dataset.poisoning import PoisonIns, PoisonTask
from trustfids.utils.typing import ArrayLike, Config, Dataset, NDArrays, Scalar

keras = tf.keras
from keras.callbacks import History
from keras.layers import Dense, Input
from keras.losses import Loss
from keras.optimizers import Optimizer


class VerbLevel(str, Enum):
    """Verbosity level.

    This class defines the verbosity level for the client, which is then passed directly
    to Keras' model training API.

    From https://keras.io/api/models/model_training_apis/#fit-method:

        verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 =
        one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with
        ParameterServerStrategy. Note that the progress bar is not particularly useful
        when logged to a file, so verbose=2 is recommended when not running
        interactively (eg, in a production environment).
    """

    AUTO = "auto"
    SILENT = 0
    INPLACE = 1
    VERBOSE = 2

    def __str__(self):
        return str(self.value)


class LearnerFactory:
    """Factory for creating Learner objects."""

    def __init__(
        self,
        cls: DictConfig,
        train_sets: Dict[str, Dataset],
        test_set: Dataset,
    ) -> None:
        """Initialize the factory.

        Parameters:
        -----------
        train_sets : Dict[str, Dataset]
            Dictionary mapping client IDs to training datasets.
        test_set : Dataset
            Test dataset.
        """
        self.train_sets = train_sets
        self.test_set = test_set
        self.learners = {}
        self.cls = cls

    def create(self, cid: str, *args, **kwargs) -> "Learner":
        """Create a Learner object.

        Once a Learner object has been created for a client, it is cached and returned
        on subsequent calls.

        Parameters:
        -----------
        cid : str
            Client ID.
        args, kwargs
            Additional arguments passed to the Learner constructor.

        Returns:
        --------
        Learner
            Learner object for the client.
        """

        if cid not in self.learners:
            self.learners[cid] = instantiate(
                self.cls,
                *args,
                train_set=self.train_sets[cid],
                test_set=self.test_set,
                **kwargs,
            )
        return self.learners[cid]


class Learner(metaclass=ABCMeta):
    """Base class for Trust-FIDS' learners.

    A learner is a object that represents all ML logic for a client. It is responsible
    for creating, training and evaluating a model.

    The Learner class is an abstract class, and must be subclassed to be used. The
    subclass MUST implement the following methods:
      * `create_model`: Create the Keras model.
      * `fit`: Train the model.
      * `evaluate`: Evaluate the model.

    Additionally, the subclass CAN implement the following methods:
      * `poison`: Poison the training / evaluation data.
      * `cross_evaluate`: Evaluate the given models on the local evaluation set.
    """

    model: keras.Model

    train_set: Dataset
    eval_set: Dataset
    test_set: Dataset

    def get_train(self) -> Dataset:
        """Getter for the train set."""
        return self.train_set

    def get_eval(self) -> Dataset:
        """Getter for the eval set."""
        return self.eval_set

    def get_test(self) -> Dataset:
        """Getter for the test set."""
        return self.test_set

    def get_parameters(self) -> NDArrays:
        """Get the model parameters.

        Returns:
        --------
        NDArrays
            Model parameters.
        """
        return self.model.get_weights()

    @classmethod
    def __subclasshook__(cls, subclass):
        """Check if the subclass implements the required methods."""
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "evaluate")
            and callable(subclass.evaluate)
            and hasattr(subclass, "create_model")
            and callable(subclass.create_model)
            or NotImplemented
        )

    def __init__(
        self,
        cid: str,
        *args,
        verbosity: VerbLevel = VerbLevel.SILENT,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the Learner.

        A significant part of the initialization is the creation of the model, which is
        done by the abstract method `create_model`. All mandatory parameters of
        `create_model` must be passed to this method. For a description of
        `create_model`'s arguments, please refer to its documentation.

        Parameters:
        -----------
        verbosity : VerbLevel
            Verbosity level for the model training API.
        **kwargs : dict
            Keyword arguments to be passed to `create_model`.
        """
        self.cid = cid
        self.verbosity: int | str = (
            "auto" if verbosity == VerbLevel.AUTO else int(verbosity)
        )
        self.seed = seed

    @staticmethod
    @abstractmethod
    def create_model(
        num_features: int,
        loss_fn: Loss | Callable[[ArrayLike, ArrayLike], float] | None = None,
        optimizer: str | Optimizer | None = None,
        learning_rate: float | None = None,
    ) -> keras.Model:
        """Create the Keras model.

        This method MUST be implemented by the subclass, and MUST return a Keras model
        object. It is called by the Learner's constructor, and its arguments are passed
        directly to the constructor of the Keras model.

        Parameters:
        -----------
        num_features : int
            Number of features in the input data.
        loss_fn : Loss | Callable[[ArrayLike, ArrayLike], float] | None
            Loss function to use for training. Defaults to None.
        optimizer : str | Optimizer | None
            Optimizer to use for training. Defaults to None.
        learning_rate : float | None
            Learning rate for the optimizer. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, parameters: NDArrays, config: Config) -> History:
        """Fit the model to the local data set.

        This method MUST be implemented by the subclass. It provides the learning logic.

        Parameters:
        -----------
        model : keras.Model
            Model to train.
        config : Config
            Configuration dictionary.

        Returns:
        --------
        History
            Training history of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Evaluate the current parameters on the test set.

        This method MUST be implemented by the subclass. It provides the evaluation
        logic.

        Parameters:
        -----------
        model : keras.Model
            Model to train.
        config : Config
            Configuration dictionary.

        Returns:
        --------
        float
            Loss value.
        Dict[str, Scalar]
            Dictionary of metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def crossevaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Evaluate the received parameters on the local evaluation set.

        This method MUST be implemented by the subclass. It provides the evaluation
        logic.

        Parameters:
        -----------
        model : keras.Model
            Model to train.
        config : Config
            Configuration dictionary.

        Returns:
        --------
        float
            Loss value.
        Dict[str, Scalar]
            Dictionary of metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def poison(self, task: PoisonTask) -> None:
        """Poison the local data set.

        This method MUST be implemented by the subclass. It provides the poisoning
        logic. Other informations than `task` (such as the target) shall be accessed
        through the class.

        Implementations SHOULD check the value of `self.poisoning_ins.poison_eval` to
        determine whether to poison the evaluation set or not.

        Parameters:
        -----------
        task : PoisonTask
            Poisoning task, i.e. the manipulation to perform on the data set.
        """
        raise NotImplementedError


class IDSClient(NumPyClient):
    """Base client class for Trust-FIDS' clients.

    A client is a object that represents a single client in a federated learning,
    inheriting from flwr.client.NumPyClient. It MUST implement the following methods, as
    expected by the Flower framework:
      * get_parameters: return the current model parameters
      * fit: fit the model to the local data set, starting from the given parameters
      * evaluate: evaluate the given parameters on the local data set
    """

    cid: str
    learner: ray.ActorID

    def __init__(self, cid: str, learner: ray.ActorID) -> None:
        """Initialize the client.

        Parameters:
        -----------
        cid : str
            Client ID, used to identify the client in the server.
        learner : Learner
            Learner object, containing the ML logic and training data for the client.
        """

        # TODO: format logs homogeneously
        # self.logger = logging.getLogger("IDSClient")
        # self.logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(logging.StreamHandler())
        # self.logger.info(f"   {cid}")

        self.cid = cid
        self.learner = learner

    def get_parameters(self) -> NDArrays:
        """Return the current parameters.

        Returns:
        --------
        NDArrays
            Current model parameters.
        """
        return ray.remote(self.learner.get_parameters.remote())

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Fit the model to the local data set.

        This method is called by the server to train the model on the client's data set.
        Its implementation simply respects the Flower API, while the logic is delegated
        to the Learner object.

        Parameters:
        -----------
        parameters : NDArrays
            Model parameters to start from.
        config : Config
            Dictionary containing the configuration for the training.

        Returns:
        --------
        NDArrays
            Model parameters after training.
        int
            Number of samples used for training.
        Dict[str, Scalar]
            Dictionary containing the metrics collected during the training.
        """

        hist = cast(History, ray.get(self.learner.fit.remote(parameters, config)))

        return (
            ray.get(self.learner.get_parameters.remote()),
            len(ray.get(self.learner.get_train.remote())),
            {
                "_cid": self.cid,
                "loss": hist.history["loss"][-1],
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model parameters on the local dataset.

        This method is called by the server to evaluate the model on the client's data
        set. Its implementation simply respects the Flower API, while the logic is
        delegated to the Learner object.

        Parameters:
        -----------
        parameters : NDArrays
            Model parameters to evaluate.
        config : Config
            Dictionary containing the configuration for the evaluation.

        Returns:
        --------
        float
            Loss on the local data set.
        int
            Number of samples used for evaluation.
        Dict[str, Scalar]
            Dictionary containing the metrics collected during the evaluation.
        """
        loss, metrics = ray.get(self.learner.evaluate.remote(parameters, config))

        return loss, len(ray.get(self.learner.get_test.remote())), metrics


class XevalClient(IDSClient):
    """Client class for Trust-FIDS' clients that suport Cross-Evaluation.

    This class extends the IDSClient class, adding support for Cross-Evaluation. It
    overrides the evaluate method, so that it can evaluate a set of models on the local
    data set, instead of a single model. Otherwise, it behaves exactly like the IDSClient
    class.
    """

    def __init__(self, *args, self_evaluation: bool = False, **kwargs) -> None:
        """Initialize the client.

        Additional parameters:
        ----------------------
        self_evaluation : bool
            If True, the client will evaluate its own model. Otherwise, it will skip its
            model, and return perfect metric values for it.
        """
        self.self_evaluation = self_evaluation
        super().__init__(*args, **kwargs)

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model parameters on the client."""

        model_length: int = int(config["model_length"])
        models = [
            parameters[i : i + model_length]
            for i in range(0, len(parameters), model_length)
        ]

        if "tasks" in config:
            # Cross-evaluation
            task_ids: List[str] = json.loads(str(config["tasks"]))
            metrics = {}
            task_metrics = {}
            losses = []
            loss = 0.0

            for tid, weights in zip(task_ids, models):
                if (
                    tid == self.cid and not self.self_evaluation
                ):  # if self-evaluation is disabled, skip self's model and go to next client
                    # fmt: off
                    results: Tuple[float, int, Dict[str, Scalar]] = (
                        0,  # loss
                        -1, # number of samples (will be overwritten)
                        {   # perfect results for self-evaluation
                            "accuracy": 1,
                            "precision": 1,
                            "recall": 1,
                            "f1": 1,
                            "mcc": 1,
                            "missrate": 0,
                            "fallout": 0,
                        },
                    )
                    # fmt: on
                else:
                    loss, task_metrics = ray.get(
                        self.learner.crossevaluate.remote(weights, config)
                    )

                metrics[tid] = json.dumps(task_metrics)
                losses.append(loss)

            return (
                float(np.mean(losses)),
                len(ray.get(self.learner.get_eval.remote())),
                metrics,
            )

        else:
            # Normal evaluation
            return super().evaluate(models[0], config)


# --------------------------------------------------------------------------------------
# Debug IDSClient
# --------------------------------------------------------------------------------------

# if __name__ == "__main__":
#     import os

#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     from pathlib import Path

#     import hydra
#     import tensorflow as tf
#     from flwr.common import ndarrays_to_parameters
#     from hydra import compose, initialize
#     from trustfids.client.learners import MLPPopoola
#     from trustfids.client.model import create_model
#     from trustfids.dataset.nfv2 import load_siloed_data
#     from trustfids.utils.distribution import clients_per_silo
#     from trustfids.utils.parsing import ParsingError, parse_silo
#     from trustfids.utils.setup import set_seed

#     initialize(version_base=None, config_path="../conf", job_name="analysis_train")
#     cfg = compose(
#         config_name="config",
#         overrides=["archi=trustfids", "learner=mlp", "dataset=nfv2_sampled"],
#     )

#     set_seed(cfg.xp.seed)
#     # tf_gpu_setup()

#     # Silos and clients.
#     silos = []
#     for silo in cfg.fl.silos:
#         silos.append(parse_silo(silo))

#     num_benign = sum(s.benign for s in silos if s.dataset == "cicids.csv.gz")
#     num_malicious = sum(s.malicious for s in silos if s.dataset == "cicids.csv.gz")
#     num_clients = num_benign + num_malicious
#     # counters for constructing client ids
#     benign_counter = 0
#     malicious_counter = 0
#     clients = []
#     datasets = {}
#     for silo in silos:
#         clients += [
#             f"client_{i}" for i in range(benign_counter, benign_counter + silo.benign)
#         ]
#         clients += [
#             f"attacker_{i}"
#             for i in range(malicious_counter, malicious_counter + silo.malicious)
#         ]
#         benign_counter += silo.benign
#         malicious_counter += silo.malicious

#         # load the dataset
#         _old = os.getcwd()
#         os.chdir(Path(__file__).parent)
#         silo_sets: List[Tuple[Dataset, Dataset]] = hydra.utils.instantiate(
#             cfg.dataset.load_data,
#             silo.dataset + ".csv.gz",
#             n_partitions=len(clients),
#             base_path="../../data/sampled",
#         )
#         os.chdir(_old)

#         for client, data in zip(clients, silo_sets):
#             train, test = data
#             # store the references to datasets in the object store
#             datasets[client] = (train, test)

#     cids = list(datasets.keys())
#     client_configs = {}
#     for i, c in enumerate(cids):
#         train, test = datasets[c]
#         client_configs[c] = {
#             "train_set": train,
#             "test_set": test,
#         }

#     print(f"Made configuration for: {cids}")

#     lrnr = MLPPopoola(
#         client_configs[cids[0]]["train_set"],
#         client_configs[cids[0]]["test_set"],
#         "toto",
#         seed=cfg.xp.seed,
#     )

#     print(f"Created learner")

#     client = XevalClient(
#         cid=cids[0],
#         learner=lrnr,
#     )

#     client.model.summary()

#     print(f"Created client")
#     init_weights = client.get_parameters()

#     new_weights, _, _ = client.fit(
#         init_weights, {"epochs": 10, "batch_size": 128, "round": 1}
#     )
#     print(f"Trained model")

#     _, _, metrics = client.evaluate(
#         new_weights * 5,
#         {
#             "batch_size": 128,
#             "num_rounds": 1,
#             "round": 1,
#             "tasks": json.dumps(["T1", "T2", "T3", "T4", "T5"]),
#             "model_length": len(new_weights),
#         },
#     )

#     _, _, metrics = client.evaluate(
#         new_weights,
#         {
#             "batch_size": 128,
#             "num_rounds": 1,
#             "round": 1,
#             "model_length": len(new_weights),
#         },
#     )

#     print(f"Evaluated model")
#     print(metrics)
