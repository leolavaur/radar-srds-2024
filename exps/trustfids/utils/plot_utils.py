""" utils for plotter, contains functions to extract values from a 
run or a multirun 
#TODO_tests
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, OrderedDict, Tuple, NamedTuple, Optional
from collections import namedtuple
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
from trustfids.dataset.poisoning import (
    parse_poisoning_selector,
    PoisonTask,
    PoisonTasks,
)

markers = ["s", "o", "D", "v", "+", "*", "^", "p", ".", "P", "<", ">", "X"]
colors = [
    "skyblue",
    "moccasin",
    "darkseagreen",
    "lightcoral",
    "palevioletred",
]
# Adjusted from seaborn color palette so that with the following baseline order : [foolsgold,fedavg,fedavg_c, trustfids]
# fedavg and fedavg_c are visually close
b_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

edge_colors = ["steelblue", "goldenrod", "seagreen", "brown", "mediumvioletred"]
hatches = ["/", "\\", "x", ".", "|", "-", "+", "o", "O", "*"]


def load_distribution(path: Path) -> Dict[str, List[str]]:
    """load client_id from distribution.json
    Args:
        distribution_path (Path): path towards the baseline file.
    Return:
        Dict[str,List[str]] : The client initial data distribution with the following format :
        {
        "cicids": [
            "client_000",
            "client_001",
            "client_002",
            "client_003",
            "client_004"
        ],
        ...
        }
    """
    check_paths(path)
    with open(path / "distribution.json") as f:
        return json.load(f)


def load_poisoning_selector(p: Path) -> str:
    """Return the poisoning selector of a run.

    Args:
        p (Path): Path to the run

    Returns:
        str: Poisoining selector if there is one, "" otherwise
    """
    check_paths(p)
    config = OmegaConf.load(p / ".hydra/config.yaml")
    return config.attacker.poisoning if "poisoning" in config.attacker else ""


def load_change_in_poisoning(p: Path):
    s: str = load_poisoning_selector(p)
    # TODO extract the number of round to prevent hardcoded value.
    poison_task: Tuple[PoisonTask, Optional[PoisonTasks]] = parse_poisoning_selector(
        load_poisoning_selector(p), 10
    )
    return poison_task


def load_attacker_distribution(p: Path) -> str:
    """Return a string describing the attack situation.

    Args:
        p (Path): run to analyze

    Returns:
        str: "benign","lone","byzantine_min","byzantine_maj"
    """
    # TODO_test

    participants: List[str] = load_participants(p)

    # extract targeted cluster participants only.
    if (
        not any("attacker" in pa for pa in participants)
        # and load_attack_scenario(p).attack_type == ""
    ):
        return "benign"

    attacker: int = 0
    benign: int = 0
    participants = load_participants_from_dataset(p, extract_attacker_dataset(p))
    for pa in participants:
        if "attacker" in pa:
            attacker += 1
        else:
            benign += 1
    if attacker == 1:
        return "lone"
    elif attacker <= benign:
        return "byzantine_minority"
    else:
        return "byzantine_majority"


def exclude_attackers(partition: List[List[str]]) -> List[List[str]]:
    """Return partition withtout it's attacker

    Args:
        partition (List[List[str]]): partition of pariticipants that may include attackers

    Returns:
        List[List[str]]: Partition of participants without attackers
    """
    newpart: List[List[str]] = []
    for c in partition:
        newpart.append([p for p in c if "attacker" not in p])
    return [c for c in newpart if c]


def load_participants(p: Path, attacker: bool = True) -> List[str]:
    """Return participants from a run.

    Args:
        p (str): path of the run.
        attacker (bool): when False attackers are excluded. Default to True.

    Returns:
        List[str]: list of participants from a run.
    """
    return [
        p
        for dataset_p in load_distribution(Path(p)).values()
        for p in dataset_p
        if (not "attacker" in p) or attacker
    ]


def load_metric(
    p: Path,
    attacker: bool = True,
    metric: str = "accuracy",
    r: int = 10,
    dataset: str = "",
) -> List[float]:
    """Return values of the specified metric

    Args:
        p (Path): run to extract results from
        attacker (bool): when False attackers are excluded. Default to True.
        metric (str, optional): Metric to return. Defaults to "accuracy".
        r (int,optionnal): round number to consult metric. Defaults to 10.
    Returns:
        List[float]: evaluation of
    """
    check_paths(p)

    if dataset:
        participants = load_participants_from_dataset(p, dataset, attacker)
    else:
        participants = load_participants(p, attacker)
    p_metrics = p / "metrics.json"
    results: List[float] = []

    with open(p_metrics) as f:
        metrics = json.load(f)
        for pa in participants:
            results.append(metrics[pa][r - 1][1][metric])

    return results


def load_participants_from_dataset(p: Path, d: str, attacker=True) -> List[str]:
    """Return all the participants from a dataset.

    Args:
        p (str): path of the run.
        d (str): observed dataset.
        attacker (bool): when False attackers are excluded. Default to True.

    Returns:
        List[str]: list of participants of a dataset.
    """
    return [e for e in load_distribution(p)[d] if (not "attacker" in e) or attacker]


class Attack_Scenario(NamedTuple):
    attack_type: str
    target: str
    dataset: str


def load_attack_scenario(p: Path) -> Attack_Scenario:
    """load attack parameter. Doesn't work if multiple labels or datasets are targeted.

    Args:
        p (Path): path of the run from which the attack scenario should be extracted

    Returns:
        Attack_Scenario: Named tupple,
        attack_type can be either "targeted", "untargeted", or "" if there is no attack.
        target : poisonned label if attack_type is "targeted"
        dataset : name of the poisonned dataset, only specified if attack_type is "targeted" or "untargeterd"
    """
    check_paths(Path(p))
    config = OmegaConf.load(Path(p) / ".hydra/config.yaml")
    # Weird conf choice led config.attacker.poison_eval to be True even when there are
    # no ongoing poisonning. For backward compatibility the presence of the "target" keys is checked.
    if (config.attacker.poison_eval == False) or (
        "target" not in config.attacker.keys()
    ):
        return Attack_Scenario("", "", "")
    d = extract_attacker_dataset(p)
    if config.attacker.target:
        return Attack_Scenario("targeted", config.attacker.target[0], d)
    else:
        return Attack_Scenario("untargeted", "", d)


def load_weights(path: Path, exploded=False) -> Dict[str, Dict[str, float]]:
    """return client weights at each round.

    Args:
        path (Path): path to the observed run

    Returns:
        Dict[str,Dict[str,float]]: weight for each client at each round.
        Follow this format :
        {
        "r1": {
            "attacker_000": 0.19608142785689903,
            "client_000": 0.19953571245485555,
            "client_001": 0.20025321289039788
            ...
            },
        "r2": {
            "attacker_000": 0.1962972445324006,
            "client_000": 0.19973897940561902,
            "client_001": 0.19974206939152633,
            ...
            },
        ...
        }
    """
    check_paths(path)
    p = (
        path / "client_weights_exploded.json"
        if exploded
        else path / "client_weights.json"
    )
    with open(p) as f:
        return json.load(f)


def check_paths(*spaths: Path) -> None:
    """Raise exception is the submitted path doesn't exist"""
    for p in spaths:
        if not p.absolute().exists():
            raise ValueError(f"Path {p} does not exist")


def dir_from_multirun(*in_dirs: str) -> List[Path]:
    """Extracts path from a multi-run

    Args:
        in_dir (str):multi-run path

    Returns:
        List[str]: list of hydra subfolder in the specified folder
    """
    dirs: List[Path] = []
    for multipath in in_dirs:
        dirs += [
            p
            for p in Path(multipath).iterdir()
            if p.is_dir() and Path(p / ".hydra").exists()
        ]
    return dirs


def attacker_as_a_dataset(distrib: OrderedDict[str, List[str]]) -> Dict[str, List[str]]:
    """extract attacker from their dataset to mock them as a dataset.

    Args:
        distrib (OrderedDict[str,List[str]]): distribution of participants extracted from distribution.json.

    Returns:
        Dict[str,List[str]]: distribution with attacker as a dataset
    """
    new_distrib: OrderedDict[str, List[str]] = OrderedDict()
    for dataset, participants in distrib.items():
        for p in participants:
            if "attacker" in p:
                if f"{dataset}_attacker" not in new_distrib:
                    new_distrib[f"{dataset}_attacker"] = []
                new_distrib[f"{dataset}_attacker"].append(p)
            else:
                if dataset not in new_distrib:
                    new_distrib[dataset] = []
                new_distrib[dataset].append(p)
    return new_distrib


def extract_xevals(path: str) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], str]:
    """Extract xevals as a dict for accuracy evaluation

    Args:
        path (str): Hydra run output.

    Returns:
        Tuple[Dict[str,Dict[str, Dict[str, float]]],str]: xevals for every round from the run and evaluation metric used for clustering.
            {
                "r1" : xevals_r1,
                "r2" : xevals_r2,
                ...
            }
    """
    check_paths(Path(path))
    p = Path(path)
    xevals = {}

    # redundant, might want to create a common accessible object for reading conf files.
    conf = OmegaConf.load(p / ".hydra/config.yaml")
    clustering_metric = (
        conf.archi.strategy.clustering_metric
        if "clustering_metric" in conf.archi.strategy
        else "accuracy"
    )

    for i in range(10):
        xevals[f"r{i+1}"] = json.load(open(p / f"xeval_{clustering_metric}_{i+1}.json"))
    return (xevals, clustering_metric)


def extract_attacks_stats(path: str):
    metrics = extract_metrics(path)
    return {p: metrics[p][9][1]["attack_stats"] for p in metrics}


def extract_attacker_dataset(path: Path) -> str:
    """Extract the first poisonned dataset encountered

    Args:
        path (str): Run folder.

    Returns:
        str: A poisonned dataset
    """
    distrib = load_distribution(path)
    for dataset, participants in distrib.items():
        for p in participants:
            if "attacker" in p:
                return dataset
    return ""


def extract_metrics(path: str):
    p = Path(path)
    return json.load(open(p / "metrics.json"))
