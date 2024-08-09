"""Result analysis utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf

STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "X", "D", "d", "|"]
COLORS = list(mcolors.TABLEAU_COLORS.values())  # type: ignore

Result = Tuple[
    DictConfig,
    Dict[str, List[Tuple[int, Dict[str, Any]]]],
    Dict[str, List[str]],
    Dict[str, List[List[str]]],
    List[Dict[str, Dict[str, float]]],
]


def load_results(*spaths: str, multirun: bool = False) -> List[Result] | Result:
    """Load results from a list of paths.

    Args:
        spaths: List of paths to load results from. multirun: Whether to load the
        subdirectories of the provided path. Only used if
            a single path is provided.

    Returns:
        A tuple or a list of tuples containing the following:
            - Hydra configuration for the run (DictConfig)
            - Per-round metrics for each client, including more precise analysis for the
                last round (Dict[str, List[Tuple[int, Dict[str, Any]]]])
            - Initial distribution of clients (Dict[str, List[str]])
            - Clusters per round (Dict[str, List[List[str]]])
            - Cross-evaluation metrics at each round (List[Dict[str, Dict[str, float]]])

    """
    paths: List[Path] = []

    for p in spaths:
        if not Path(p).absolute().exists():
            raise ValueError(f"Path {p} does not exist")

    if len(spaths) == 0:
        raise ValueError("No paths provided")
    if len(spaths) == 1:
        p = Path(spaths[0])
        if multirun:
            paths = [p for p in Path(spaths[0]).iterdir() if p.is_dir()]
        else:
            paths = [p]
    else:
        paths = [Path(p) for p in spaths]

    returns = []
    for p in paths:
        returns.append(
            (
                (  # Hydra configuration for the run
                    OmegaConf.load(p / ".hydra/config.yaml")
                    if (p / ".hydra/config.yaml").exists()
                    else None
                ),
                (  # Per-round metrics for each client
                    json.loads((p / "metrics.json").read_text())
                    if (p / "metrics.json").exists()
                    else None
                ),
                (  # Initial distribution of clients
                    json.loads((p / "distribution.json").read_text())
                    if (p / "distribution.json").exists()
                    else None
                ),
                (  # Clusters per round
                    json.loads((p / "clusters.json").read_text())
                    if ((p / "clusters.json").exists())
                    else None
                ),
                (  # Cross-evaluation metrics at each round
                    [
                        json.loads((f).read_text())
                        for f in sorted(list(p.glob("xeval_*.json")))
                    ]
                    if len(list(p.glob("xeval_*.json"))) > 0
                    else None
                ),
            )
        )
    if len(returns) == 1:
        return returns[0]
    return returns
