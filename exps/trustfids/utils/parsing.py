"""Trust-FIDS parsing functions."""

import re
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

from omegaconf import DictConfig, ListConfig


def sanitize_resolver(s: str) -> str:
    """OmegaConf sanitizing resovler.

    This resolver is used to sanitize the override keys of Hydra's default config
    groups. Overrides are of the form `path/to/group@package=value`. When using
    `dirname_override` in Hydra's sweep configuration, overrides containing a slash will
    be interpreted as a path, and the resulting output directory will be incorrect.

    See:
    - https://github.com/facebookresearch/hydra/discussions/2409
    - https://omegaconf.readthedocs.io/en/2.1_branch/custom_resolvers.html
    - https://github.com/omry/omegaconf/issues/426

    Parameters:
    -----------
    s : str
        The override string.

    Returns:
    --------
    str
        The sanitized override string.
    """
    assert isinstance(s, str), "The input must be a string."
    return s.replace("/", "_")


class SiloConfig(NamedTuple):
    """Silo configuration."""

    dataset: str
    benign: int
    malicious: int


def parse_silo(silo: DictConfig) -> Tuple[str, int, int]:
    """Parse a silo configuration.

    Parameters:
    -----------
    silo : DictConfig
        The silo configuration object.

    Returns:
    --------
    str
        The name of the dataset.
    int
        The number of benign clients.
    int
        The number of malicious clients.
    """
    # load the dataset name
    dataset = silo.dataset

    # load the number of clients
    clients = silo.clients
    if isinstance(clients, int):
        benign = clients
        malicious = 0
    elif isinstance(clients, str) and re.compile(r"^\d+/\d+$").match(clients):
        benign, malicious = map(int, clients.split("/"))
    elif isinstance(clients, DictConfig | Dict):
        benign = clients.get("benign", 0)
        malicious = clients.get("malicious", 0)
    else:
        raise ValueError(
            "The `fl.silos[].clients` configuration key must be an integer, a string of the form 'benign/malicious', or a dictionary with keys 'benign' and/or 'malicious'."
        )

    return SiloConfig(dataset, benign, malicious)


class ParsingError(Exception):
    """Raised when a parsing error occurs."""

    pass
