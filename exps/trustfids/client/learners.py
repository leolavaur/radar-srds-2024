"""Module with Learner implementations."""

import json
import operator
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from sklearn.model_selection import train_test_split
from trustfids.client.base import Learner
from trustfids.client.metrics import (
    mean_absolute_error,
    mean_squared_error,
    metrics_from_preds,
    root_mean_squared_error,
)
from trustfids.dataset import Dataset
from trustfids.dataset.nfv2 import poison
from trustfids.dataset.poisoning import PoisonIns, PoisonOp, PoisonTask
from trustfids.utils.log import logger
from trustfids.utils.typing import Config, NDArray, NDArrays, Scalar

keras = tf.keras
from keras.callbacks import History
from keras.layers import Dense, Input
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam


class NFV2Learner(Learner):
    """Learner partial implementation for the NF-V2 dataset.

    This class implements the Learner API for the NF-V2 dataset. It overrides the
    `__init__` and `poison` methods to add support for poisoning attacks on NF-V2. It
    also overrides the `evaluate` and `crossevaluate` methods to use the NF-V2
    Subclasses will need to implement `create_model`, `fit`, and `evaluate`.


    Learner subclasses can be decorated with `ray.remote` to use it as an Actor in a
    Ray cluster. This way, the actor maintains its state between method calls.
    """

    poisoning_ins: Optional[PoisonIns] = None

    def __init__(
        self,
        train_set: Dataset,
        test_set: Dataset,
        *args,
        poisoning_ins: Optional[PoisonIns] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.test_set = test_set

        # train/eval set split
        self.train_set, self.eval_set = train_set.split(
            0.8,
            seed=self.seed,
            stratify=train_set.m["Attack"] if "Attack" in train_set.m else None,
        )

        if poisoning_ins is not None:
            # Data poisioning
            self.poisoning_ins = poisoning_ins
            self.poison(poisoning_ins.base)

        self.model = self.create_model(self.train_set.X.shape[1])

    def _poison_hook(self, round: int) -> None:
        if self.poisoning_ins is None or self.poisoning_ins.tasks is None:
            return

        task = self.poisoning_ins.tasks.get(round, None)
        if task is None:
            return

        self.poison(task)

    def poison(self, task: PoisonTask) -> None:
        if self.poisoning_ins is None:
            raise ValueError("Client not configured for poisoning")

        logger.debug(f"{self.cid}: poisoning train set with {task}")

        self.train_set, _ = poison(
            self.train_set,
            *task,
            target_classes=self.poisoning_ins.target,
            seed=self.seed,
        )

        if self.poisoning_ins.poison_eval:
            logger.debug(f"{self.cid}: poisoning eval set with {task}")
            self.eval_set, _ = poison(
                self.eval_set,
                *task,
                target_classes=self.poisoning_ins.target,
                seed=self.seed,
            )

    def evaluate(self, parameters: NDArrays, config: Config):
        loss, metrics, y_pred = self._eval(self.test_set, parameters, config)

        if config["num_rounds"] == config["round"]:
            # Compute attack-wise statistics. Only enabled for the last evaluation round.
            attack_stats = []
            class_df = self.test_set.m["Attack"]
            for attack in (a for a in class_df.unique() if a != "Benign"):
                attack_filter = class_df == attack

                count = len(self.test_set.m[attack_filter])
                correct = len(
                    self.test_set.m[(attack_filter) & (self.test_set.y == y_pred)]
                )
                missed = len(
                    self.test_set.m[(attack_filter) & (self.test_set.y != y_pred)]
                )

                attack_stats.append(
                    {
                        "attack": attack,
                        "count": count,
                        "correct": correct,
                        "missed": missed,
                        "percentage": f"{correct / count:.2%}",
                    }
                )

            metrics["attack_stats"] = json.dumps(attack_stats)

        return loss, metrics

    def crossevaluate(self, parameters: NDArrays, config: Config):
        loss, metrics, _ = self._eval(self.eval_set, parameters, config)

        return loss, metrics

    def _eval(
        self, dataset: Dataset, parameters: NDArrays, config: Config
    ) -> Tuple[float, Dict[str, Scalar], NDArray]:
        """Class-specific evaluation method.

        Parameters:
        -----------
        dataset: Dataset
            Dataset to evaluate the model on.
        model: keras.Model
            Model to evaluate.
        config: Config
            Configuration dictionary.

        Returns:
        --------
        float
            Loss value.
        Dict[str, Scalar]
            Dictionary of metrics.
        NDArray
            Array of predictions.
        """
        raise NotImplementedError


class AEBertoli(NFV2Learner):
    """Autoencoder for the NF-V2 dataset.

    This class implements the Learner API. It is used by the FL client to execute ML
    tasks on the client's local dataset.

    This autoencoder is based on the work of Bertoli et al. (2022-2023), who tested
    Federated Learning on the NF-V2 dataset with a focus on heterogeneity.

    References:
    -----------
      * Bertoli, G., Junior, L., Santos, A., & Saotome, O., Generalizing intrusion
        detection for heterogeneous networks: A stacked-unsupervised federated learning
        approach. in Computers & Security (2023). https://arxiv.org/abs/2209.00721
      * Bertoli, G., fl-unsup-nids, Github (2022), https://github.com/c2dc/fl-unsup-nids
    """

    def __init__(
        self,
        *args,
        val_ratio: int = 0,
        fit_loss_fn: str = "mse",
        eval_loss_fn: str = "mae",
        **kwargs,
    ):
        """Create new AE instance."""

        super().__init__(*args, **kwargs)

        if fit_loss_fn == "mae":
            self.fit_loss_fn = mean_absolute_error
        elif fit_loss_fn == "mse":
            self.fit_loss_fn = mean_squared_error
        elif fit_loss_fn == "rmse":
            self.fit_loss_fn = root_mean_squared_error
        else:
            raise ValueError(f"Invalid fit loss function: {fit_loss_fn}")

        if eval_loss_fn == "mae":
            self.eval_loss_fn = mean_absolute_error
        elif eval_loss_fn == "mse":
            self.eval_loss_fn = mean_squared_error
        elif eval_loss_fn == "rmse":
            self.eval_loss_fn = root_mean_squared_error
        else:
            raise ValueError(f"Invalid fit loss function: {eval_loss_fn}")

        # Rename the loss function according the Learner API
        kwargs["loss_fn"] = self.fit_loss_fn
        super().__init__(*args, **kwargs)

        X_train = self.train_set.X

        if val_ratio == 0:
            X_val = X_train
        else:
            X_train, X_val = train_test_split(
                self.train_set.X,
                test_size=val_ratio,
                random_state=self.seed,
            )

        self.train_set = Dataset(X_train, self.train_set.y, self.train_set.m)
        # fill the validation set with dummy data
        self.val_set = Dataset(X_val, pd.Series(), pd.DataFrame())

    @classmethod
    def create_model(
        cls, num_features, *, loss_fn=None, optimizer=None, learning_rate=None, **_
    ):
        """Create a Keras model for the NF-V2 dataset.

        Args:
            num_features (int): Number of features in the dataset.

        Returns:
            Model: Keras model.
        """
        lr = learning_rate or 0.001

        model = Sequential(
            [
                Input(shape=(num_features,)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(8, activation="relu"),
                Dense(4, activation="relu"),
                Dense(8, activation="relu"),
                Dense(16, activation="relu"),
                Dense(32, activation="relu"),
                Dense(num_features, activation="sigmoid"),
            ]
        )

        if optimizer is None:
            optimizer = Adam(learning_rate=lr)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
        )

        return model

    def fit(self, parameters: NDArrays, config: Config):
        batch_size = int(config["batch_size"])

        self._poison_hook(int(config["round"]))

        self.model.set_weights(parameters)

        return self.model.fit(
            self.train_set.to_sequence(
                batch_size, target=0, seed=self.seed, shuffle=True
            ),
            epochs=int(config["epochs"]),
            shuffle=True,
            # The verbose argument is annotated as `str` in the Keras API, but it
            # actually expects an `int` value for the verbosity level (0, 1, or 2). The
            # string "auto" is the only valid string value for `model.fit()`.
            # `typing.cast` is used to satisfy the type checker, but the value is
            # unchanged.
            verbose=cast(str, self.verbosity),
        )

    def _eval(self, dataset: Dataset, parameters: NDArrays, config: Config):
        batch_size = int(config["batch_size"])

        self.model.set_weights(parameters)

        inferences = self.model.predict(
            self.val_set.to_sequence(batch_size), verbose=cast(str, self.verbosity)
        )
        val_losses = self.eval_loss_fn(self.val_set.X, inferences)
        threshold = np.quantile(val_losses, 0.95)

        # evaluate on test set
        inferences: NDArray = self.model.predict(
            dataset.to_sequence(batch_size), verbose=cast(str, self.verbosity)
        )
        losses: NDArray = self.eval_loss_fn(dataset.X, inferences)

        y_pred = np.array(losses > threshold, dtype=int)

        mean_loss = float(np.mean(losses))
        metrics = metrics_from_preds(dataset.y, y_pred)
        metrics["loss"] = float(mean_loss)

        return mean_loss, metrics


@ray.remote
class MLPPopoola(NFV2Learner):
    """Multi-layer perceptron for the NF-V2 dataset.

    This class implements the Learner API. It is used by the FL client to execute ML
    tasks on the client's local dataset.

    This model is based on the model proposed by Popoola et al. (2021), who used it
    against the NF-V2 dataset, to evaluate new aggregation strategies for FL.

    References:
    -----------
      * S. I. Popoola, G. Gui, B. Adebisi, M. Hammoudeh, and H. Gacanin, “Federated Deep
        Learning for Collaborative Intrusion Detection in Heterogeneous Networks,”
        in 2021 IEEE 94th Vehicular Technology Conference (VTC2021-Fall), Sep.
        2021, pp. 1–6. doi: 10.1109/VTC2021-Fall52928.2021.9625505.
    """

    @classmethod
    def create_model(
        cls, num_features, *, loss_fn=None, optimizer=None, learning_rate=None, **_
    ):
        """Create a Keras model for the NF-V2 dataset.

        Args:
            num_features (int): Number of features in the dataset.

        Returns:
            Model: Keras model.
        """
        lr = learning_rate or 0.0001

        model = keras.Sequential(
            [
                keras.layers.Dense(128, activation="relu", input_shape=(num_features,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=optimizer or Adam(learning_rate=lr),
            loss=loss_fn or keras.losses.BinaryCrossentropy(),
        )

        return model

    def fit(self, parameters: NDArrays, config: Config):
        batch_size = int(config["batch_size"])

        self._poison_hook(int(config["round"]))

        self.model.set_weights(parameters)

        return self.model.fit(
            self.train_set.to_sequence(
                batch_size, target=1, seed=self.seed, shuffle=True
            ),
            epochs=int(config["epochs"]),
            # batch_size=config["batch_size"],
            #    validation_split=self.val_ratio,
            shuffle=True,
            verbose=cast(str, self.verbosity),
        )

    def _eval(self, dataset: Dataset, parameters: NDArrays, config: Config):
        batch_size = int(config["batch_size"])

        self.model.set_weights(parameters)

        loss = self.model.evaluate(
            dataset.to_sequence(batch_size, target=1, seed=self.seed, shuffle=True),
            verbose=cast(str, self.verbosity),
        )

        # Do not shuffle the test set for inference, otherwise we cannot compare y_pred
        # with y_true.
        inferences = self.model.predict(
            dataset.to_sequence(batch_size, target=1), verbose=cast(str, self.verbosity)
        )

        y_pred = np.around(inferences).astype(int).reshape(-1)

        y_true = dataset.y.to_numpy().astype(int)

        metrics = metrics_from_preds(y_true, y_pred)
        metrics["loss"] = loss  # type: ignore

        return loss, metrics, y_pred


if __name__ == "__main__":
    # Instanciate a learner for debugging purposes
    import logging

    import tensorflow as tf
    from trustfids.dataset.nfv2 import load_data

    logger.setLevel(logging.DEBUG)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    train, test = load_data("./exps/data/sampled/cicids.csv.gz", 0.8)
    learner = MLPPopoola(train, test, "tot", poisoning_ins=PoisonIns(["DoS"], PoisonTask(0.0), {3: PoisonTask(0.2)}), verbosity=0)  # type: ignore
    model = learner.create_model(
        cast(Dataset, ray.get(learner.get_train.remote())).X.shape[1]
    )
    learner.fit(model, {"batch_size": 512, "epochs": 10, "round": 1})
    learner.evaluate(model, {"batch_size": 512, "num_rounds": 1, "round": 1})

    # test poisonning
    learner.fit(model, {"batch_size": 512, "epochs": 10, "round": 3})
    learner.evaluate(model, {"batch_size": 512, "num_rounds": 1, "round": 3})
