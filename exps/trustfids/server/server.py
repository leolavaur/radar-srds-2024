# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""


import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import flwr
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.strategy import Strategy
from trustfids.server.strategy import FedXeval
from trustfids.utils.log import logger

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class XevalServer(flwr.server.Server):
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: FedXeval, **kwargs
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy, **kwargs)

        self.strategy: FedXeval = strategy

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        logger.info("Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        logger.info("Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            logger.info(
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        logger.info("FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            fit_results = ([], [])
            if res_fit:
                parameters_prime, _, fit_results = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
            if not fit_results:
                logger.fatal(
                    f"Round {current_round} - No results gathered after `fit_round`"
                )
                quit(1)

            # --------------------------------------------------------------------------
            # At this point, `self.parameters` contains a concatenated list of all
            # client models.
            np_params = parameters_to_ndarrays(self.parameters)
            assert len(np_params) == self.strategy.model_length * len(
                fit_results[0]
            ), f"Server should have {len(fit_results[0])} parameters, but has {len(np_params) / self.strategy.model_length}"
            # --------------------------------------------------------------------------

            # Centralized evaluation if available
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                logger.info(
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Gather cross-evaluation results
            res_xeval = self.cross_evaluate_round(
                server_round=current_round, timeout=timeout
            )

            # Aggregate new model parameters
            if res_fit and res_xeval:
                _, _, fitres = res_fit
                _, _, evalres = res_xeval
                res_agg = self.strategy.aggregate_fit_evaluate(
                    current_round, *fitres, *evalres
                )
                parameters_prime, _ = res_agg  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = (
                        parameters_prime  # Concatenated cluster parameters
                    )
                else:
                    logger.fatal(
                        f"Round {current_round} - No new parameters"
                        + " after `aggregate_fit_evaluate`"
                    )
                    quit(1)

            # --------------------------------------------------------------------------
            # At this point, `self.parameters` contains a concatenated list of all
            # cluster models.
            np_params = parameters_to_ndarrays(self.parameters)
            assert len(np_params) == self.strategy.model_length * len(
                self.strategy.clusters.keys()
            ), f"Server should have {len(self.strategy.clusters.keys())} parameters, but has {len(np_params) / self.strategy.model_length}."
            # --------------------------------------------------------------------------

            # Evaluate new cluster models
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                if evaluate_metrics_fed:
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        logger.info("FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            logger.info("evaluate_round %s: no clients selected, cancel", server_round)
            return None
        logger.info(
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        logger.info(
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def cross_evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_crossevaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            logger.info(
                "cross_evaluate_round %s: no clients selected, cancel", server_round
            )
            return None
        logger.info(
            "cross_evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        logger.info(
            "cross_evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # No aggregation for cross-evaluation

        return None, {}, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            logger.info("fit_round %s: no clients selected, cancel", server_round)
            return None
        logger.info(
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        logger.info(
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
