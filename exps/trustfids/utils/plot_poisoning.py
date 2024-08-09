"""utils for poisoning plotting
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from trustfids.utils.plot_utils import (
    Attack_Scenario,
    b_colors,
    check_paths,
    colors,
    edge_colors,
    hatches,
    load_attack_scenario,
    load_metric,
    markers,
)
from trustfids.utils.stats_utils import get_missrate, get_missrate_list


def boxplot(
    *paths: str,
    title: str = "",
    out_path: str = "",
    include_attacker: bool = False,
    metric="accuracy",
    target: Optional[str] = None,
    dataset: Optional[str] = None,
    font_size: int = 12,
    zoom: str = "",
    labels: List[str] = [],
):
    """Boxplot of a metric for participants from different baselines.

    Args:
        paths (str): Path to extract results from.
        title (str, optional): Name of the plot. Defaults to "".
        out_path (str, optional): file in which the plot should be saved. Defaults to "".
        include_attacker (bool, optional): False if the attackers should be excluded from the plot. Defaults to False.
        metric (str, optional): Metric to plot. Defaults to "accuracy"
        target (str, optional): Name of the targeted class. Defaults to None
        dataset (str, optional): Name of the targeted dataset, must be set if target is set. to None
        fontsize (int, optional): Fontsize used in the graph. Defaults to 12.
        labels (List[str],optional) : labels to add on each box.

    """
    # attacks: Dict[str, Attack_Scenario] = {p: load_attack_scenario(p) for p in paths}
    # Ajout d'un label pour le schéma ?
    if target and dataset:
        mustaches: List[List[float]] = [
            get_missrate_list(p, dataset, target) for p in paths
        ]
    else:
        mustaches: List[List[float]] = [
            load_metric(Path(p), attacker=include_attacker, metric=metric)
            for p in paths
        ]
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(5.5)

    ax.set(
        # axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        ylabel=metric,
    )
    ax.set_xticklabels(labels)
    rcParams.update({"font.size": font_size})

    boxes = ax.boxplot(mustaches, patch_artist=True)
    # ax.set_xticklabels(['Sample1', 'Sample2', 'Sample3', 'Sample4'])
    for box, w1, w2, c1, c2, med, f, color, edge_color, h in zip(
        boxes["boxes"],
        boxes["whiskers"][::2],
        boxes["whiskers"][1::2],
        boxes["caps"][::2],
        boxes["caps"][1::2],
        boxes["medians"],
        boxes["fliers"],
        colors,
        edge_colors,
        hatches,
    ):
        for el in [box, w1, w2, c1, c2]:
            el.set(color=edge_color, linewidth=2)
        med.set(color="deeppink", linewidth=2)
        f.set(marker="o", color="maroon", alpha=0.5)

        # med.set(color="darkorange", linewidth=2)

        box.set(facecolor=color)
        box.set(hatch="/")

    for flier in boxes["fliers"]:
        flier.set(marker="o", color="#e7298a", alpha=0.5)
    if zoom == "high":
        plt.ylim(ymin=0.941, ymax=1.001)
    elif zoom == "low":
        plt.ylim(ymin=-0.001, ymax=0.059)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_attack_success_rate(
    dataset: str = "botiot",
    target: str = "Reconnaissance",
    baselines: List[str] = ["trustfids", "foolsgold", "fedavg"],
    out_path: str = "",
    zero_indexed=False,
    title: str = "",
    fontsize: int = 14,
    small_fig: bool = True,
):
    """plot the attack success rate from multiple baselines over multiple scenarios

    Args:
        dataset (str): Name of the dataset to observe, e.g. "cicids", "botiot"
        out_path (str, optional): If specified, save the plot in this path. Defaults to "".
        title (str, optional): Title for the plot. Defaults to "".
        exploded (bool, optional): True if the score have been further exploded False if it's the normalized weight out of the reputation system.
        fontsize (int, optional): Size of the font for the graph. Default to 12.
        zoomed_scale (bool, optional): Zoom the scale to 0.196-0.202. Only works when none exploded. Default to True
        small_fig (bool, optional): Reduce the size of the fig output. Default to True.
        y_centered_legend (bool, optional): Center the legend on the y axis. Default to False.
        labels (str, optional): Labels for each ploted directories.
    """
    # plot le taux d'empoisonnement pour lone targeted
    baselines = ["foolsgold", "fedavg", "fedavg_c", "trustfids"]
    target_types = ["targeted", "untargeted"]

    scenarios = ["lone", "sybils_min", "sybils"]
    scenarios_labels: Dict[str, str] = {
        "lone": "Lone",
        "sybils_min": "Colluding minority",
        "sybils": "Colluding majority",
    }
    # Big plot parameters
    rcParams.update({"font.size": 13})
    fig_height: int = 3 if small_fig else 4
    fig, axs = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(5 * 6, fig_height))
    rcParams.update({"font.size": fontsize})
    l = 0
    for target_type in target_types:
        z_s = zip(scenarios, [i for i in range(len(scenarios))])
        for scenario, j in z_s:
            marks = deepcopy(markers)
            colors = deepcopy(b_colors)
            target_label: str = "100T" if target_type == "targeted" else "100U"
            axs[j + l * len(scenarios)].set_title(
                f"{scenarios_labels[scenario]} {target_label}",
                fontname="DejaVu Sans Mono",
            )
            k = 0
            for baseline in baselines:
                mark = marks.pop(0)
                col = colors.pop(0)
                # removed path part"exps/results"
                path = f"./{baseline}/{scenario}/{target_type}/"
                x = []
                y = []
                for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    x.append(i * 100)
                    y.append(
                        get_missrate(
                            path + str(i),
                            dataset=dataset,
                            targeted=target_type == "targeted",
                            target=target,
                        )[0]
                    )
                # Plotting
                baseline_names = {
                    "foolsgold": "Foolsgold",
                    "fedavg": "FedAvg",
                    "fedavg_c": "Clust. FedAvg",
                    "trustfids": "RADAR",
                }
                axs[j + l * len(scenarios)].plot(
                    x,
                    y,
                    marker=mark,
                    markevery=(k, len(baselines)),
                    color=col,
                    markersize=10,
                    linewidth=3,
                    label=f"{baseline_names[baseline]}",
                )
                k += 1
        l += 1

    if title:
        fig.suptitle(title)

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
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(baselines),
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

    plt.ylabel("Attack succes rate")  
    plt.xlabel("% of labels flipped")
    if out_path:
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()
