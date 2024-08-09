"""Unit test file for dataset.common
"""
import numpy as np
from omegaconf import OmegaConf
from copy import deepcopy
from pathlib import Path
from typing import List

from trustfids.dataset.nfv2 import load_data
from trustfids.dataset.common import Dataset

path_from_exec = "./trustfids/dataset/tests/"


def test_drop_random_class():
    np.random.seed(1123)
    data_path = "./data/reduced/botiot.csv.gz"
    d: Dataset = load_data(data_path, test_ratio=None, n_partitions=None)
    d_bis: Dataset = deepcopy(d)
    d_bis.drop_random_class()
    assert (
        len(getattr(d, "m")["Attack"].unique())
        == len(getattr(d_bis, "m")["Attack"].unique()) + 1
    )

    # Stripped class should be stripped in X, y and m
    assert (
        len(getattr(d_bis, "m").index)
        == len(getattr(d_bis, "y").index)
        == len(getattr(d_bis, "X").index)
    )

    # Benign class should not be stripped.
    d.drop_random_class(class_to_avoid=["Benign", "Reconnaissance"])
    assert "Benign" in getattr(d_bis, "m")["Attack"].unique()
    assert "Reconnaissance" in getattr(d_bis, "m")["Attack"].unique()

    # Choose arbitrarily the class that should be dropped
    dropped_class = getattr(d, "m")["Attack"].unique()[0]
    d.drop_random_class(class_to_drop=dropped_class)
    assert dropped_class not in getattr(d, "m")["Attack"].unique()

    # Check that it works with target from hydra config
    config = OmegaConf.load(Path(path_from_exec) / "config.yaml")
    classes = ["Benign"]
    classes.append(*config.attacker.target)
    d.drop_random_class(classes)

    # Check the removal of multiple classes
    old_length = len(getattr(d_bis, "m")["Attack"].unique())
    dropped_clases: List[str] = d_bis.drop_random_class(nb_class_to_drop=2)
    assert len(dropped_class) == 2
    assert old_length - 2 == len(getattr(d_bis, "m")["Attack"].unique())


def test_drop_all_but_random_class():
    np.random.seed(1123)
    data_path = "./data/reduced/botiot.csv.gz"
    d: Dataset = load_data(data_path, test_ratio=None, n_partitions=None)

    # Drop all but Benign and a random attack class
    kept = d.drop_all_but_random_class()
    assert "Benign" in getattr(d, "m")["Attack"].unique()
    assert len(getattr(d, "m")["Attack"].unique()) == 2

    # Avoid a specific class
    d: Dataset = load_data(data_path, test_ratio=None, n_partitions=None)
    classes_to_avoid = list(set(getattr(d, "m")["Attack"].unique()) - {kept})
    kept_bis = d.drop_all_but_random_class(classes_to_avoid, ["Benign"])
    assert kept == kept_bis
    assert len(getattr(d, "m")["Attack"].unique()) == 2

    # Only keep class_to_keep
    d: Dataset = load_data(data_path, test_ratio=None, n_partitions=None)
    kept_bis = d.drop_all_but_random_class(
        classes_to_keep=["Benign", "Reconnaissance"], classes_to_keep_only=True
    )
    assert len(getattr(d, "m")["Attack"].unique()) == 2
    assert "Reconnaissance" in getattr(d, "m")["Attack"].unique()
