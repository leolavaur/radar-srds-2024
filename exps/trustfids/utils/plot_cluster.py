"""utils for cluster plotting
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Iterable, List, OrderedDict, Tuple

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import cluster
from sklearn.metrics import completeness_score, homogeneity_score

from trustfids.clustering.cluster import build_clusters
from trustfids.utils.plot_utils import (
    Attack_Scenario,
    attacker_as_a_dataset,
    check_paths,
    dir_from_multirun,
    exclude_attackers,
    extract_xevals,
    load_distribution,
    load_attacker_distribution,
    load_attack_scenario,
)


def count_clusters(path: str) -> int:
    """Return the number of clusters for an hydra run

    Args:
        path (str): valid path to an hydra run

    Returns:
        int: nb of clusters
    """
    check_paths(Path(path))
    return len(load_distribution(Path(path)))


def plot_multiple_threshold(
    path: str,
    min_step: float,
    max_step: float,
    step: float,
    title: str = "",
    dtypes: List[str] = ["euclidean", "cosin_sim"],
):
    """plot different fixed threshold for clustering.

    Args:
        path (str): path where the xevals are located.
        min_step (float): min step to test.
        max_step (float): max step to test.
        step (float): step to test.
        title (str): graph title
        dtypes (List[str]): Types of distances to use
    Raises:
        ValueError: _description_
    """
    xevals_all_round, metric = extract_xevals(path)
    nb_clusters = count_clusters(path)
    # Building results
    if min_step > max_step:
        raise ValueError("minstep > maxstep")

    alphas: List[float] = []
    clust_nb: Dict[str, List[int]] = {"baseline": []}
    first_dtype = dtypes[0]
    for dtype in dtypes:
        clust_nb[dtype] = []
        current_step = min_step
        max_dif = -1
        worst_clusters_nb = -1
        while current_step < max_step:
            for round, xevals in xevals_all_round.items():
                clusters = build_clusters(
                    xevals, "xevals", "fixed", current_step, dtype
                )
                if abs(len(clusters) - nb_clusters) > max_dif:
                    worst_clusters_nb = len(clusters)
            if dtype == first_dtype:
                alphas.append(current_step)
                clust_nb["baseline"].append(nb_clusters)
            clust_nb[dtype].append(worst_clusters_nb)
            current_step += step

    # Plotting
    plt.figure()

    # Baseline
    markers = ["o", ".", "v", "*", "+", "^", "p", "D"]
    plt.plot(
        alphas,
        clust_nb.pop("baseline"),
        linestyle="-",
        color="red",
        label=f"Expected number of clusters",
        marker="none",
    )
    # Different distance type
    for dtype in clust_nb:
        lstyle = "-" if dtype == "euclidean" else "--"
        if len(markers) == 0:
            raise IndexError(
                "There are more line to plot than the number of available symbols"
            )
        plt.plot(
            alphas,
            list(clust_nb[dtype]),
            linestyle=lstyle,
            label=f"Distance type : {dtype}",
            marker=markers.pop(0),
        )

    # plt.plot(alphas, clust_nb["cosin_sim"], marker="o")
    plt.title(title)
    plt.xlim(xmin=min_step, xmax=max_step)
    plt.xlabel("Fixed alpha value")
    plt.ylim(ymin=0, ymax=max([max(vals) for vals in clust_nb.values()]) + 1)
    plt.ylabel("Nb of clusters")
    plt.subplot(111).spines["top"].set_visible(False)
    # plt.legend(loc="lower right")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def pair_in_cluster(pair: Tuple[str, str], cluster: List[str]) -> bool:
    """check if pair is in cluster

    Args:
        pair (Tuple[str, str]): pair of participants id.
        cluster (List[str]): list of participants id from a cluster.
    Return:
        presence (bool): True if the pair is in the cluster
    """
    a, b = pair
    return True if (a in cluster) and (b in cluster) else False


def compute_rand_index(
    compared: List[List[str]], ground_truth: List[List[str]]
) -> float:
    """Compute the rand index of two distribution
    Args:
        compared (List[List[str]]): First distribution used for comparison
        ground_truth (List[List[str]]): Second distribution used for comparison
    Return:
        rand_index (float): Rand index of the two distributions
    """
    #  Compute all pair of elements
    participants = []
    for i in ground_truth:
        participants += i
    pairs = [
        (a, b) for idx, a in enumerate(participants) for b in participants[idx + 1 :]
    ]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for pair in pairs:
        truth_grouped = 0
        for cluster in ground_truth:
            truth_grouped += pair_in_cluster(pair, cluster)
        compared_grouped = 0

        for cluster in compared:
            compared_grouped += pair_in_cluster(pair, cluster)
        if truth_grouped and compared_grouped:
            tp += 1
        if not truth_grouped and not compared_grouped:
            tn += 1
        if truth_grouped and not compared_grouped:
            fn += 1
        if not truth_grouped and compared_grouped:
            fp += 1
    rand_index = (tp + tn) / (len(pairs))
    return rand_index


def clustering_metric_wrapper(
    dir: str,
    func,
    split_attacker: bool = False,
) -> Dict[str, List[float]]:
    """Compute the completness values between clustering results and a source of truth
    Args:
         path (str): List of paths to load results from.
         func (function): Function used to compare both partition (tested on homogeneity_score and completness_score)
         split_attacker (bool): Separate attacker from benign in the comparison distribution. Default to False.
    Return:
        rand_indexes (Dict[str, List[float]]) : for every label return the rand index for each round.
    """
    path = Path(dir)
    ground_truth: List[str]
    if split_attacker:
        ground_truth = [
            dataset if "attacker" not in p else "attacker"
            for dataset, participants in load_distribution(path).items()
            for p in participants
        ]
    else:
        ground_truth = [
            dataset
            for dataset, participants in load_distribution(path).items()
            for p in participants
        ]

    # Alphabetical sort so that ground_truth and clustering partition are unaffected by the result.
    ground_truth.sort()

    clusters = json.load(open(path / "clusters.json"))
    distrib = load_distribution(path)
    p_dataset = {
        p: dataset for dataset, participants in distrib.items() for p in participants
    }
    score = {}
    # Clusters id must be unique
    used_cluster_id: Set = set()
    for round in clusters:
        # Attackers grouped with benign are considered misslabelled.
        cluster_result = []
      
        for participants in clusters[round]:
            #
            c_id: str
            count_p_datasets: Counter = Counter(
                [p_dataset[p] for p in participants if "attacker" not in p]
            )
            # Attackers alone in their cluster are correctly classified as attackers
            # otherwise they are missclassified as benign.
            dataset_in_majority: str = (
                count_p_datasets.most_common(1)[0][0]
                if count_p_datasets.total()
                else "attacker"
            )
            while dataset_in_majority in used_cluster_id:
                dataset_in_majority += "_bis"
            c_id = dataset_in_majority
            cluster_result += [c_id for _ in participants]
        cluster_result.sort()
        score[f"{round}"] = func(ground_truth, cluster_result)
    return score


def get_completness_over_round(
    dir: str,
    split_attacker: bool = False,
) -> Dict[str, List[float]]:
    """Compute the completeness values between clustering results and a source of truth
    Args:
         path (str): List of paths to load results from.
         split_attacker (bool): Separate attacker from benign in the comparison distribution. Default to False.

    Return:
        (Dict[str, List[float]]) : completness of the base partition and clustering results.
    """
    return clustering_metric_wrapper(dir, completeness_score, split_attacker)


def get_homogeneity_over_round(
    dir: str,
    split_attacker: bool = False,
) -> Dict[str, List[float]]:
    """Compute the homogeneity values between clustering results and a source of truth
    Args:
         path (str): List of paths to load results from.
         split_attacker (bool): Separate attacker from benign in the comparison distribution. Default to False.

    Return:
        (Dict[str, List[float]]) : homogeneity of the base partition and clustering results.
    """
    return clustering_metric_wrapper(dir, homogeneity_score, split_attacker)


def get_rand_over_round(
    paths: List[Path],
    split_attacker: bool = False,
    no_attackers: bool = False,
    legacy_label: bool = True,
    distribution_label: bool = False,
    target_label: bool = False,
    percent_poisoning_label: bool = False,
) -> Dict[str, Tuple[Iterable[str], Iterable[float]]]:
    """Compute rand index values between clustering results and base distribution
    Args:
        *spaths (str): List of paths to load results from.
         split_attacker (bool): Separate attacker from benign in the comparison distribution. Default to False.
         no_attackers (bool): remove attackers from both partitions. Default to False.
         legacy_label (bool): Display the cross evaluation and the threshold strategy. Default to True.
         distribution_label (bool): Include distribution of participants ("benign","lone", ...) in label. Default to False.
         target_label (bool): Target (targeted or untargeted) if there is attacker in the run. Default to False.
         percent_poisoning_label (bool): Make the label display percents of poisoning. Default to False.
    Return:
        rand_indexes (Dict[str, Tuple[Iterable[str],Iterable[float]]]) : for every label return the rand index for each round.
    """
    check_paths(*paths)

    results: Dict[str, Tuple[Iterable[str], Iterable[float]]] = {}
    for in_dir in paths:
        ground_truth: List[List[str]]
        if split_attacker:
            ground_truth = list(
                attacker_as_a_dataset(OrderedDict(load_distribution(in_dir))).values()
            )
        else:
            ground_truth = list(load_distribution(in_dir).values())
        clusters = json.load(open(in_dir / "clusters.json"))
        if no_attackers:
            clusters = {k: exclude_attackers(clusters[k]) for k in clusters}
            ground_truth = exclude_attackers(ground_truth)
        rand_index: Dict[str, float] = {}
        for round in clusters:
            rand_index[f"{round}"] = compute_rand_index(clusters[round], ground_truth)

        # Naming of the observed run.
        conf = OmegaConf.load(in_dir / ".hydra/config.yaml")
        clustering_metric = (
            conf.archi.strategy.clustering_metric
            if "clustering_metric" in conf.archi.strategy
            else "accuracy"
        )
        reputation_metric = (
            conf.archi.strategy.reputation_metric
            if "reputation_metric" in conf.archi.strategy
            else "accuracy"
        )
        threshold_type = (
            conf.archi.strategy.clustering_kwargs.threshold_type
            if "threshold_type" in conf.archi.strategy.clustering_kwargs
            else "mean"
        )
        distance = (
            conf.archi.strategy.clustering_kwargs.distance_type
            if "distance_type" in conf.archi.strategy.clustering_kwargs
            else "L2_Norm"
        )
        label: str = ""
        if legacy_label and not any(
            [distribution_label, target_label, percent_poisoning_label]
        ):
            label = (
                f"{clustering_metric}_{reputation_metric}_{threshold_type}_{distance}"
            )
        att_scen: Attack_Scenario = load_attack_scenario(in_dir)
        if distribution_label:
            l = load_attacker_distribution(in_dir)
            if l == "byzantine_majority":
                label += "majority of byzantine"
            else:
                label += l
        if target_label:
            target: str = att_scen.attack_type
            if target:
                label += f" {target}"
        if percent_poisoning_label:
            percent: str = (
                f"{str(float(conf.attacker.poisoning)*100).rstrip('0').rstrip('.')} %"
            )
            if att_scen.attack_type and (label != "benign"):
                label += f" {percent}"
        results[label] = (rand_index.keys(), rand_index.values())
    return results


def plot_rand_comparison(
    *in_dirs: str,
    out_path: str = "",
    title: str = "",
    multirun: bool = False,
    split_attacker: bool = False,
    no_attackers: bool = False,
    legend_out: bool = False,
    label_args: Dict[str, bool] = {},
):
    """print rand comparison.
    Args:
        in_dirs (str): input director.y.ies
        out_path (str): file in which the plot can be saved. If left empty plot is shown
        title (str): title for the plot
        multirun (bool): Wether the provided path(s) correspond to an hydra run or multirun. Default to False.
        label_args (Dict[str,bool]): Define what label arguments will be passed to get_rand_over_round.
        split_attacker (bool): Separate attacker from benign in the comparison distribution. Default to False.
        no_attackers (bool): remove attackers from both partitions. Default to False.

    """

    if multirun == True:
        paths = dir_from_multirun(*in_dirs)
    else:
        paths = [Path(dir) for dir in in_dirs]

    nb_plots = len(paths)
    i = 0
    rcParams.update({"font.size": 13})
    results = get_rand_over_round(paths, split_attacker, no_attackers, **label_args)
    plt.figure(figsize=(5, 3))
    markers = ["o", "D", "v", "*", "+", "^", "p", ".", "P", "<", ">", "X"]
    for label, values in results.items():
        round_nb, rand_index = values
        lstyle = "--" if any(m in label for m in ["euclidean"]) else "-"
        if len(markers) == 0:
            raise IndexError(
                "There are more line to plot than the number of available symbols"
            )
        plt.plot(
            round_nb,
            rand_index,
            linestyle=lstyle,
            label=label,
            marker=markers.pop(0),
            markersize=8,
            linewidth=2,
            markevery=(i, nb_plots),
        )
        i += 1
    if title:
        plt.title(title)
    # plt.legend(loc="lower right")
    # plt.legend(loc="middle right")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if legend_out:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="lower right")

    plt.ylim(ymin=0.8, ymax=1.01)
    plt.ylabel("Rand index")
    plt.xlabel("Round number")
    plt.subplot(111).spines["top"].set_visible(False)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()


def create_square_ndarray(xevals: Dict[str, Dict[str, float]]) -> np.ndarray[float]:
    """Create a square NDArray from a xevals dictionary.

    Args:
        xevals (Dict[str, Dict[str, float]]): cross evaluation results
    Returns:
        np.ndarray[float]: cross evaluation square matrice, size is the number of participants.
    """
    # Get the keys from the outer dictionary
    keys = list(xevals.keys())

    # Determine the size of the square ndarray
    size = len(keys)

    # Create an empty square ndarray of the desired size
    square_ndarray = np.empty((size, size), dtype=float)

    # Iterate over the keys and fill the ndarray
    for i, outer_key in enumerate(keys):
        inner_dict = xevals[outer_key]
        for j, inner_key in enumerate(keys):
            square_ndarray[i, j] = inner_dict[inner_key]

    return square_ndarray


def plot_evals_2D(
    path: str,
    round: int = 10,
    out_path: str = "",
    removed_data_set: List[str] = [],
    specified_data_set: List[str] = [],
):
    """Diagonalize the evaluation matrix and choose the eigenvector with biggest eigenvalue for projection

    Args:
        path (str): _description_
        round (int): Round that will be plot for the cluster.
        out_path (str, optional): _description_. Defaults to "".
    """
    distrib = load_distribution(Path(path))
    distrib = attacker_as_a_dataset(OrderedDict(distrib))
    xevals, metric = extract_xevals(path)
    xeval = xevals[f"r{round}"]
    if not specified_data_set:
        specified_data_set = list(distrib.keys())

    nd_xeval = create_square_ndarray(xeval)
    # TODO replace harcoded distance type with omega conf readed distance type.
    # nd_xeval = distance_xevals_matrice(xeval,"cosin_sim")

    # https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad

    eigen_vals, eigen_vecs = np.linalg.eig(nd_xeval)
    # Make a list of (eigenvalue, eigenvector) tuples
    # eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals)) if not np.imag(eigen_vals[i])]
    eigen_pairs = [
        (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
    ]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    projected_xeval = nd_xeval.dot(w)

    markers = ["o", "D", "v", "*", "+", "^", "p", ".", "P", "<", ">", "X"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    i = 0
    for dataset, participants in distrib.items():
        m = markers.pop()
        c = colors.pop()
        for p in participants:
            eval = projected_xeval[i]
            if (dataset not in removed_data_set) and (dataset in specified_data_set):
                plt.scatter(eval[0], eval[1], c=c, label=dataset, marker=m)
            i += 1

    # Make sure labels are unique.
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1, 0.5)
    )
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='lower left')

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"2D projection of evaluations for round {round}")
    plt.show()
