"""utils for reput plotting
"""

from ast import Set
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Iterator, List, OrderedDict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from omegaconf import DictConfig, OmegaConf
from trustfids.utils.plot_utils import (
    attacker_as_a_dataset,
    check_paths,
    dir_from_multirun,
    extract_xevals,
    load_distribution,
    load_weights,
    load_poisoning_selector,
    markers,
)


def test_unpacker(*in_dirs: str, dataset: str):
    for d in in_dirs:
        print(d)


def plot_weights_in_cluster(
    *in_dirs: str,
    dataset: str,
    out_path: str = "",
    zero_indexed=True,
    attacker_only=False,
    multirun=False,
    title: str = "",
    exploded: bool = True,
    fontsize: int = 12,
    zoomed_scale: bool = True,
    legend_param={},
    small_fig: bool = False,
    labels: Tuple[str, ...] = (),
):
    """plot the weight of each participants from a dataset over rounds. Warning this function
    follow the initial distribution and not the clustering results.

    Args:
        in_dirs (str): input director.y.ies, if multiples directories are specified, labels shall also be specified
        dataset (str): Name of the dataset to observe, e.g. "cicids", "botiot"
        out_path (str, optional): If specified, save the plot in this path. Defaults to "".
        attacker_only (bool, optional): Only plot the attacker reputation.
        title (str, optional): Title for the plot. Defaults to "".
        exploded (bool, optional): True if the score have been further exploded False if it's the normalized weight out of the reputation system.
        fontsize (int, optional): Size of the font for the graph. Default to 12.
        zoomed_scale (bool, optional): Zoom the scale to 0.196-0.202. Only works when none exploded. Default to True
        small_fig (bool, optional): Reduce the size of the fig output. Default to False.
        legend_param (Dict[str, Any], optional): Center the legend on the y axis. Default to none.
        labels (str, optional): Labels for each ploted directories.
    """
    paths = [Path(dir) for dir in in_dirs]
    check_paths(*paths)
    results: Dict[str, List[Tuple[int, float]]] = {}

    if len(paths) >= 1 and len(labels) != 0:
        if len(paths) != len(labels):
            raise ValueError(
                f"The number of path is {len(paths)}, while the number of labels is  {len(labels)}, it should be the same."
            )
        else:
            z_p: Iterator[Tuple[Path, str]] = zip(paths, labels)
    else:
        z_p: Iterator[Tuple[Path, str]] = zip(paths, [""])

    for path, label in z_p:
        distrib = load_distribution(path)
        if dataset not in distrib:
            raise KeyError(
                f"{dataset} is not present in the provided distribution : {distrib.keys()}"
            )
        weights = load_weights(path, exploded)

        # Extract the weights for the observed cluster
        nb_round = len(weights.keys())
        # Handle attacker only.
        ploted_participants: List[str] = []
        if attacker_only:
            for p in distrib[dataset]:
                if "attacker" in p:
                    ploted_participants.append(p)
        else:
            ploted_participants = distrib[dataset]

        for p in ploted_participants:
            l = f"{label}_{p}" if label else p
            results[l] = []
            for i in range(nb_round):
                results[l].append((int(f"{i+1}"), weights[f"r{i+1}"][p]))
    rcParams.update({"font.size": 13})
    if small_fig:
        plt.figure(figsize=(5, 3))
    else:
        plt.figure(figsize=(5, 4))
    rcParams.update({"font.size": fontsize})

    marks = deepcopy(markers)
    nb_plots = len(results)
    i = 0
    for p in results:
        if len(marks) == 0:
            raise IndexError(
                "There are more cluster to plot than the number of available symbols"
            )
        mark = marks.pop()
        r, weight = zip(*results[p])
        plt.plot(
            r,
            weight,
            label=p,
            marker=mark,
            markevery=(i, nb_plots),
            markersize=8,
            linewidth=2,
        )
        i += 1
    if title:
        plt.title(title)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not legend_param:
        plt.legend(loc="center right")
        # plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    else:
        plt.legend(**legend_param)
    if zoomed_scale and not exploded:
        plt.ylim(ymin=0.196, ymax=0.202)
    if zoomed_scale and exploded:
        plt.ylim(ymin=-0.01, ymax=0.35)

    if zero_indexed:
        plt.ylim(ymin=-0.01)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylabel("Weight per participant")
    plt.xlabel("Round number")
    plt.subplot(111).spines["top"].set_visible(False)
    if out_path:
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_multi_weights_in_cluster(
    in_dirs: List[str],
    dataset: str,
    out_path: str = "",
    zero_indexed=False,
    attacker_only=False,
    multirun=False,
    title: str = "",
    exploded: bool = True,
    fontsize: int = 14,
    zoomed_scale: bool = True,
    small_fig: bool = True,
    labels: List[str] = [],
):
    """plot the weight of each participants from a dataset over rounds. Warning this function
    follow the initial distribution and not the clustering results.

    Args:
        in_dirs List(str): input director.y.ies, if multiples directories are specified, labels shall also be specified
        dataset (str): Name of the dataset to observe, e.g. "cicids", "botiot"
        out_path (str, optional): If specified, save the plot in this path. Defaults to "".
        attacker_only (bool, optional): Only plot the attacker reputation.
        title (str, optional): Title for the plot. Defaults to "".
        exploded (bool, optional): True if the score have been further exploded False if it's the normalized weight out of the reputation system.
        fontsize (int, optional): Size of the font for the graph. Default to 12.
        zoomed_scale (bool, optional): Zoom the scale to 0.196-0.202. Only works when none exploded. Default to True
        small_fig (bool, optional): Reduce the size of the fig output. Default to False.
        labels List(str): Labels for each ploted directories.
    """
    paths = [Path(dir) for dir in in_dirs]
    check_paths(*paths)
    nb_subfig = len(paths)

    # Results collection
    results: Dict[str, List[Tuple[int, float]]] = {}
    labels += ["" for _ in range(nb_subfig - len(labels))]

    z_p: Iterator[Tuple[Path, str, int]] = zip(
        paths, labels, [i for i in range(nb_subfig)]
    )

    # Big plot parameters
    rcParams.update({"font.size": 13})
    fig_height: int = 3 if small_fig else 4
    fig, axs = plt.subplots(
        1, nb_subfig, sharex=True, sharey=True, figsize=(5 * nb_subfig, fig_height)
    )
    rcParams.update({"font.size": fontsize})

    for path, label, j in z_p:
        distrib = load_distribution(path)
        if dataset not in distrib:
            raise KeyError(
                f"{dataset} is not present in the provided distribution : {distrib.keys()}"
            )
        weights = load_weights(path, exploded)
        nb_round = len(weights.keys())

        # Handle attacker only.
        ploted_participants: List[str] = []
        if attacker_only:
            for p in distrib[dataset]:
                if "attacker" in p:
                    ploted_participants.append(p)
        else:
            ploted_participants = distrib[dataset]

        for p in ploted_participants:
            results[p] = []
            for i in range(nb_round):
                results[p].append((int(f"{i+1}"), weights[f"r{i+1}"][p]))

        marks = deepcopy(markers)
        nb_plots = len(results)
        i = 0
        for p in results:
            if len(marks) == 0:
                raise IndexError(
                    "There are more cluster to plot than the number of available symbols"
                )
            mark = marks.pop()
            r, weight = zip(*results[p])
            axs[j].plot(
                r,
                weight,
                label=p,
                marker=mark,
                markevery=(i, nb_plots),
                markersize=10,
                linewidth=3,
            )
            i += 1
        axs[j].set_title(
            label,
            fontname="DejaVu Sans Mono",
        )
        # Only keep ticks for the first y axis
        # if j != 0:
        #     axs[j].set_yticklabels([])
        # axs[j].set_yticks([])

    if title:
        fig.suptitle(title)

    if zoomed_scale and not exploded:
        for ax in axs:
            ax.set_ylim(ymin=0.196, ymax=0.202)
    if zoomed_scale and exploded:
        for ax in axs:
            ax.set_ylim(ymin=-0.01, ymax=0.45)
    if zero_indexed:
        for ax in axs:
            ax.set_ylim(ymin=-0.01)
        # grid.
    for ax in axs:
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=len(paths),
    )

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("Weight per participant")
    plt.xlabel("Round number")
    if out_path:
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()
