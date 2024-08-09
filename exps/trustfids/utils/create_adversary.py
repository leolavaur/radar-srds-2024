"""Create poisonned dataset.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

SEED = 1138

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def mk_targetted_fn(
    label: str,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Make the targeted attack function."""

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        # Replace label 1 to 0 for designated attack
        logger.info(f"----Injecting {label}...")
        _df = df.copy()
        _df.loc[_df["Attack"] == label, "Label"] = 0
        assert not any(
            _df.loc[_df["Label"] == label, "Label"] == 1
        ), f"No {label} label should be 1 after targeted attack."
        return _df

    return _fn


def mk_untargeted_fn() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Make the untargeted attack function."""

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        # Replace label 1 to 0 for designated attack
        logger.info(f"----Flipping labels...")
        _df = df.copy()
        _df.loc[_df["Label"] == 1, "Label"] = 0
        assert not any(
            _df["Label"] == 1
        ), "No label should be 1 after untargeted attack."
        return _df

    return _fn


def transform(
    datapath: str,
    tgt_fn: Callable[[pd.DataFrame], pd.DataFrame],
    untgt_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """ """
    # Load data
    p = Path(datapath)
    df = pd.read_csv(datapath, low_memory=True)
    srcname = p.name.split(".")[0]

    # Shuffle data
    logger.info(f"--Shuffling dataset...")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info(f"--Splitting dataset...")
    half = len(df) // 2
    benign_set = df[:half].copy()
    attack_set = df[half:].copy()

    # Targeted
    logger.info(f"--Creating targeted attack...")
    save_path = p.parent / "targeted"
    save_path.mkdir(parents=True, exist_ok=True)

    benign_path = save_path / f"{srcname}_benign.csv.gz"
    logger.info(f"--Saving benign set: {benign_path}...")
    benign_set.to_csv(benign_path, compression="gzip", index=False)

    attack_set = tgt_fn(attack_set)
    attack_path = save_path / f"{srcname}_attacked.csv.gz"
    logger.info(f"--Saving attack set: {attack_path}...")
    attack_set.to_csv(attack_path, compression="gzip", index=False)

    # Untargeted
    logger.info(f"--Creating untargeted attack...")
    save_path = p.parent / "untargeted"
    save_path.mkdir(parents=True, exist_ok=True)

    benign_path = save_path / f"{srcname}_benign.csv.gz"
    logger.info(f"--Saving benign set: {benign_path}...")
    benign_set.to_csv(benign_path, compression="gzip", index=False)

    attack_set = untgt_fn(attack_set)
    attack_path = save_path / f"{srcname}_attacked.csv.gz"
    logger.info(f"--Saving attack set: {attack_path}...")
    attack_set.to_csv(attack_path, compression="gzip", index=False)


if __name__ == "__main__":

    for d in Path("./data/").glob("*/*.csv.gz"):
        # only sampled datasets for now
        if "sampled" not in str(d):
            continue

        logger.info(f"Found {d}...")

        if "cicids" in str(d) or "CSE-CIC-IDS2018" in str(d):
            tgt_fn = mk_targetted_fn("DoS")
        elif "botiot" in str(d) or "BoT-IoT" in str(d):
            tgt_fn = mk_targetted_fn("DoS")
        elif "nb15" in str(d) or "UNSW-NB15" in str(d):
            tgt_fn = mk_targetted_fn("Fuzzers")
        elif "toniot" in str(d) or "ToN-IoT" in str(d):
            tgt_fn = mk_targetted_fn("DDoS")
        else:
            raise ValueError("Unknown dataset")

        transform(str(d), tgt_fn, mk_untargeted_fn())

    # sizes = ["sampled", "reduced"]
    # attacks = ["targeted", "untargeted"]
    # # for dsize in sizes :
    # for dsize in sizes:
    #     for att in attacks:

    #         #########
    #         # cicids
    #         #########

    #         dataset = "cicids"
    #         # potential labels : "DoS" : 99,98% , "Brute Force" : 98.62%
    #         label = "DoS"
    #         source = f"./data/{dsize}/{dataset}_{dsize}.csv.gz"

    #         dest = Path(f"./data/{dsize}/{att}/")
    #         dest.mkdir(parents=True, exist_ok=True)
    #         dest = str(dest / f"{dataset}_{dsize}_attacks.csv.gz")
    #         attack(source, dest, att, label)

    #         #########
    #         # botiot
    #         #########

    #         dataset = "botiot"
    #         # potential labels : "DoS" : 90,38% , "Reconaissance" : 73%
    #         label = "DoS"
    #         source = f"./data/{dsize}/{dataset}_{dsize}.csv.gz"
    #         df = pd.read_csv(source, low_memory=True)

    #         dest = Path(f"./data/{dsize}/{att}/")
    #         dest.mkdir(parents=True, exist_ok=True)
    #         dest = str(dest / f"{dataset}_{dsize}_attacks.csv.gz")
    #         attack(source, dest, att, label)

    #         #########
    #         # nb15
    #         #########
    #         dataset = "nb15"
    #         # potential labels : "Fuzzers":100% "Exploits" : 99,96% , "Reconaissance" : 100%

    #         label = "Fuzzers"
    #         source = f"./data/{dsize}/{dataset}_{dsize}.csv.gz"

    #         dest = Path(f"./data/{dsize}/{att}/")
    #         dest.mkdir(parents=True, exist_ok=True)
    #         dest = str(dest / f"{dataset}_{dsize}_attacks.csv.gz")
    #         attack(source, dest, att, label)

    #         #########
    #         # toniot
    #         #########
    #         dataset = "toniot"
    #         # Possible targeted label : "DDoS" 86.49% ,"scanning" 64.49%
    #         label = "DDoS"
    #         source = f"./data/{dsize}/{dataset}_{dsize}.csv.gz"

    #         dest = Path(f"./data/{dsize}/{att}/")
    #         dest.mkdir(parents=True, exist_ok=True)
    #         dest = str(dest / f"{dataset}_{dsize}_attacks.csv.gz")
    #         attack(source, dest, att, label)


# def untargeted_attack(df: pd.DataFrame) -> pd.DataFrame:
#     # Get labels
#     # Change the Attack to it's neighbor ?
#     # Replace the labels ?
#     for i, row in df.iterrows():
#         if row["Label"] == 0:
#             df.at[i, "Label"] = 1
#         else:
#             df.at[i, "Label"] = 0
#     return df


# def attack(
#     src_path: str,
#     dst_path: str,
#     attack: str = "targeted",
#     poisonned_label: str = "Infilteration",
# ) -> None:
#     """ """
#     df = pd.read_csv(source, low_memory=True)
#     if attack == "targeted":
#         df = targeted_attack(df, poisonned_label)
#     elif attack == "untargeted":
#         df = untargeted_attack(df)
#     df.to_csv(dst_path)
