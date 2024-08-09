"""Utils for extracting stats from a run.   
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, OrderedDict, Tuple

from numpy import extract
from omegaconf import DictConfig, OmegaConf
from trustfids.utils.plot_utils import (
    extract_attacks_stats,
    load_participants_from_dataset,
    load_metric,
)
from statistics import fmean


def get_missrate_list(
    path: str, dataset: str, target: str = "", include_attacker: bool = False
) -> List[float]:
    """_summary_

    Args:
        p (str): run path
        dataset (str): dataset that has been poisonned
        targeted_label (str, optional): Label from the dataset that has been flipped. Defaults to "".

    Returns:
        Tuple[float,float]: first float is missrate, second float is success rate.
    """
    a = extract_attacks_stats(path)
    participants = load_participants_from_dataset(path, dataset)
    results: list[float] = []
    for p in participants:
        if ("attacker" not in p) or include_attacker:
            for attack in a[p]:
                if attack["attack"] == target:
                    results.append(attack["missed"] / attack["count"])
    return results


def get_missrate(
    path: str,
    dataset: str,
    target: str = "",
    targeted: bool = False,
    include_attacker: bool = False,
) -> Tuple[float, float]:
    """Return the missrate on the specified target label.

    Args:
        p (str): run path
        dataset (str): dataset that has been poisonned
        target (str, optional): Label from the dataset that has been flipped. Defaults to "".
        targeted (bool, optional): True if the attack is targeted False if it's untargeed. Defaults to False.

    Returns:
        Tuple[float,float]: first float is missrate, second float is success rate.
    """
    a = extract_attacks_stats(path)
    participants = load_participants_from_dataset(Path(path), dataset)
    if targeted:
        total_count: int = 0
        total_missed: int = 0
        total_correct: int = 0
        for p in participants:
            if ("attacker" not in p) or include_attacker:
                for attack in a[p]:
                    if attack["attack"] == target:
                        total_count += attack["count"]
                        total_correct += attack["correct"]
                        total_missed += attack["missed"]
        if not total_count:
            return 0.0, 0.0
        else:
            return total_missed / total_count, total_correct / total_count
    else:
        return (
            1 - get_mean_accuracy(path, dataset="botiot"),
            get_mean_accuracy(path, dataset="botiot"),
        )

def get_missrate_percent(
    path: str, dataset: str, target: str = "", include_attacker: bool = False
) -> Tuple[str, str]:
    """_summary_
    get_missrate_percent
        Args:
            path (str): _description_
            dataset (str): _description_
            target (str, optional): _description_. Defaults to "".
            include_attacker (bool, optional): True if attacke are counted in the stats. Defaults to False.

        Returns:
            Tuple[str, str]: _description_
    """
    mr, sr = get_missrate(path, dataset, target, include_attacker=include_attacker)
    return f"Miss rate : {mr*100}%", f"Detection rate :{sr*100}%"


def get_attack_success_rate(
    path: str,
    dataset: str,
    target: str = "",
    benign_value: float = 0,
    include_attacker: bool = False,
) -> str:
    """_summary_
    get_missrate_percent
        Args:
            path (str): _description_
            dataset (str): _description_
            target (str, optional): _description_. Defaults to "".
            include_attacker (bool, optional): True if attacke are counted in the stats. Defaults to False.
            benign_value (float, optional): Base missrate from benign participant. Default to 0.
        Returns:
            Tuple[str, str]: _description_
    """
    mr, sr = get_missrate(path, dataset, target, include_attacker=include_attacker)
    return f"Attack success rate : {(mr-benign_value)*100}%"


def get_accuracy(
    path: str, dataset: str, target: str = "", include_attacker: bool = False
) -> Tuple[str, str]:
    """return the accuracy on a targeted label.
    get_missrate_percent
        Args:
            path (str): _description_
            dataset (str): _description_
            target (str, optional): _description_. Defaults to "".
            include_attacker (bool, optional): True if attacke are counted in the stats. Defaults to False.

        Returns:
            Tuple[str, str]: _description_
    """
    mr, sr = get_missrate(path, dataset, target, include_attacker=include_attacker)
    return f"Miss rate : {mr*100}%", f"Detection rate :{sr*100}%"


def get_mean_accuracy(path: str, dataset: str = "") -> float:
    """Return the mean accuracy of benign particiapants

    Args:
        path (str): Path of the run

    Returns:
        float: Mean accuracy
    """
    return fmean(
        load_metric(Path(path), attacker=False, dataset=dataset, metric="accuracy")
    )


def get_mean_missrate(path: str) -> float:
    """Return the mean missrate of benign participants

    Args:
        path (str): Path of the run

    Returns:
        float: Mean missrate
    """
    return fmean(load_metric(Path(path), attacker=False, metric="missrate"))
