"""Plotting utilities for Trust-FIDS experiments."""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Scalar
from flwr.common.typing import Metrics as MetricDict
from flwr.server.history import History
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from trustfids.server.strategy import ClientID, JSONMetrics


def plot_moustache(
    dist_metrics: Dict[ClientID, List[Tuple[int, MetricDict]]],
    metric: str,
    title: str,
    mean: bool = True,
    xlabel: str = "Round",
    save_path: Optional[str] = None,
) -> None:
    """Plot the the box plot of a metric over all clients."""

    if not dist_metrics:
        raise ValueError("Metrics dictionary is empty")

    if metric not in dist_metrics.popitem()[1][0][1]:
        raise ValueError(f"Metric `{metric}` not found in distributed metrics")

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(metric.title() if metric != "loss" else "Normalized loss")

    means = []

    for rnd in range(len(dist_metrics.popitem()[1])):
        values = []
        for _, value in dist_metrics.items():
            values.append(value[rnd][1][metric])
        plt.boxplot(values, positions=[rnd], showfliers=False)
        means.append(np.mean(values))

    if mean:
        plt.plot(means, color="red", label="Mean")

    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def plot_mean_metric(
    dist_metrics: Dict[ClientID, List[Tuple[int, MetricDict]]],
    metric: str,
    title: str,
    xlabel: str = "Round",
    save_path: Optional[str] = None,
) -> None:
    """Plot the mean of a metric over all clients."""

    if not dist_metrics:
        raise ValueError("Metrics dictionary is empty")

    if metric not in dist_metrics.popitem()[1][0][1]:
        raise ValueError(f"Metric `{metric}` not found in distributed metrics")

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(metric.title() if metric != "loss" else "Normalized loss")

    means = []

    for rnd in range(len(dist_metrics.popitem()[1])):
        values = []
        for _, value in dist_metrics.items():
            values.append(value[rnd][1][metric])
        means.append(np.mean(values))

    plt.plot(means, color="red", label="Mean")

    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def plot_comparison(
    *result_runs: Tuple[DictConfig, Dict[ClientID, List[Tuple[int, MetricDict]]]],
    metric: str,
    title: str,
    xlabel: str = "Round",
    save_path: Optional[str] = None,
) -> None:
    """Make a comparison plot of the results of multiple runs."""

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(metric.title() if metric != "loss" else "Normalized loss")

    for results in result_runs:
        config, metrics = results
        means = []

        for rnd in range(len(metrics.popitem()[1])):
            values = []
            for _, client_metrics in metrics.items():
                values.append(client_metrics[rnd][1][metric])
            means.append(np.mean(values))

        plt.plot(
            means,
            label=f"{config.baseline.name}",
        )

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
