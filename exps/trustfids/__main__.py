"""Trust-FIDS experiments."""

import json
import os
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr
import hydra
import ray
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import Metrics as MetricDict
from flwr.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from omegaconf import DictConfig, OmegaConf

from .client.base import IDSClient
from .dataset.common import Dataset
from .dataset.poisoning import PoisonIns, parse_poisoning_selector
from .server.strategy import FedXeval
from .utils.gpu import tf_gpu_count
from .utils.log import logger
from .utils.parsing import ParsingError, parse_silo, sanitize_resolver
from .utils.setup import set_seed
from .utils.typing import NDArrays, Parameters

OmegaConf.register_new_resolver("sanitize", sanitize_resolver)


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def run(cfg: DictConfig) -> None:
    """Run Trust-FIDS experiments.

    This function is the entry point for the Trust-FIDS experiments. It is managed by
    Hydra to parse the configuration and run the experiments.

    Parameters:
    -----------
    cfg : DictConfig
        The configuration object extracted from the YAML configuration file.
    """

    # Disable unnecessary warnings.
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    logger.debug(f"Current WD: {os.getcwd()}")
    logger.debug(f"Orginal WD: {hydra.utils.get_original_cwd()}")
    logger.debug(f"Dataset path: {Path(cfg.dataset.load_data.base_path).absolute()}")

    # Configuration parsing.
    # ----------------------

    if Path(hydra.utils.get_original_cwd()) != Path(__file__).parent.parent.absolute():
        logger.fatal(
            """Running from a different directory than `exps` is unsupported. 
            \tFile path: %s
            \tOrigin path: %s""",
            Path(__file__).parent.parent.absolute(),
            hydra.utils.get_original_cwd(),
        )

    logger.info("Loading configuration values.")

    # Silos and clients.
    silos = []
    for silo in cfg.fl.silos:
        silos.append(parse_silo(silo))
    if len(silos) == 0:
        raise ParsingError("No silos specified.")
    logger.debug(f"Found {len(silos)} silos: {', '.join(s.dataset for s in silos)}.")

    num_benign = sum(s.benign for s in silos)
    num_malicious = sum(s.malicious for s in silos)
    num_clients = num_benign + num_malicious
    logger.debug(f"Found {num_clients} clients ({num_benign}/{num_malicious}).")

    # Hardware configuration.
    if not cfg.hardware.get("auto", False):
        client_resources = {
            "num_cpus": os.cpu_count() * (1 - cfg.hardware.cpu_headroom) // num_clients,
            "num_gpus": tf_gpu_count() // num_clients,
        }
        logger.debug(f"Using manual hardware configuration:\n\t{client_resources}")
    else:
        client_resources = {}

    # Reproducibility.
    # ----------------

    logger.info("Setting up reproducibility.")
    if hasattr(cfg.xp, "seed"):
        seed = cfg.xp.seed
        set_seed(seed)
        logger.debug(f"Using seed {seed}.")
    else:
        logger.warn("No seed specified. Experiment will not be reproducible.")
        seed = None

    # Ray
    # ---
    # Init ray outside of Flower to allow access to the object store.

    logger.info("Initializing Ray.")
    local_mode = cfg.get("ray", {}).get("local_mode", False)

    ray_init_kwargs = {
        "ignore_reinit_error": True,  # do not fail on call to ray.init()
        "include_dashboard": True,
        "local_mode": local_mode,
    }

    ray.init(
        **ray_init_kwargs,  # type: ignore
        **client_resources,
    )

    # Loading datasets.
    # -----------------
    logger.info("Loading datasets.")

    stats: Dict[str, Dict[str, str | Dict[str, Any]]] = {}
    dist: Dict[str, List[str]] = {}

    # datasets_refs: Dict[str, Tuple[ray.ObjectRef, ray.ObjectRef]] = {}
    datasets: Dict[str, Tuple[Dataset, Dataset]] = {}
    # counters for constructing client ids
    benign_counter = 0
    malicious_counter = 0
    for silo in silos:
        logger.debug(f"Loading '{silo.dataset}'.")

        clients: List[str] = []
        clients += [
            f"attacker_{i:03}"
            for i in range(malicious_counter, malicious_counter + silo.malicious)
        ]
        clients += [
            f"client_{i:03}"
            for i in range(benign_counter, benign_counter + silo.benign)
        ]
        benign_counter += silo.benign
        malicious_counter += silo.malicious

        # load the dataset
        _old = os.getcwd()
        os.chdir(Path(__file__).parent)
        silo_sets: List[Tuple[Dataset, Dataset]] = hydra.utils.instantiate(
            cfg.dataset.load_data,
            silo.dataset + ".csv.gz",
            n_partitions=len(clients),
        )
        os.chdir(_old)
        # drop classes where there is too few informations.
        classes_to_avoid: List[str] = ["Benign"] + ["mitm", "ransomware", "worms"]
        attacker_classes_to_avoid: List[str] = ["Benign"] + ["mitm", "ransomware"]

        for client, data in zip(clients, silo_sets):
            train, test = data
            # Class drop
            # ---
            if cfg.fl.drop_class:
                targeted_attack: bool = (
                    True if ("attacker" in client) and cfg.attacker.target else False
                )
                if cfg.fl.drop_all_class_but_one:
                    dropped_classes: List[str] = [
                        # In case of targeted attack on pathological non-IID,
                        # only the targeted and Benign class are kept.
                        train.drop_all_but_random_class(
                            classes_to_avoid=attacker_classes_to_avoid
                            if targeted_attack
                            else classes_to_avoid,
                            classes_to_keep=["Benign"] + [*cfg.attacker.target]
                            if targeted_attack
                            else ["Benign"],
                            classes_to_keep_only=(targeted_attack),
                            client_id=client,
                        )
                    ]
                else:
                    dropped_classes: List[str] = train.drop_random_class(
                        class_to_avoid=attacker_classes_to_avoid
                        + [*cfg.attacker.target]
                        if targeted_attack
                        else classes_to_avoid,
                        nb_class_to_drop=cfg.fl.drop_class_nb,
                        client_id=client,
                    )
                if cfg.fl.drop_different_classes and ("attacker" in client):
                    attacker_classes_to_avoid.append(*dropped_classes)
                if cfg.fl.drop_different_classes and ("attacker" not in client):
                    classes_to_avoid.append(*dropped_classes)

            datasets[client] = (train, test)
            if "Attack" in train.m.columns:
                # store some stats about the datasets
                stats[client] = {
                    "silo_name": silo.dataset,
                    "train": {
                        "num_samples": train.X.shape[0],
                        "classes": train.m["Attack"].value_counts().to_dict(),
                    },
                    "test": {
                        "num_samples": test.X.shape[0],
                        "classes": test.m["Attack"].value_counts().to_dict(),
                    },
                }

        dist[silo.dataset] = clients
        logger.debug(f"Loaded {len(clients)} clients.")

    # FL.
    # ---

    # Setup
    _train, _test = next(iter(datasets.values()))
    lrnr = hydra.utils.instantiate(
        cfg.learner,
        _train,
        _test,
        "tmp",  # client cid
        seed=seed,
        _convert_="partial",
    )

    init_parameters: Parameters = ndarrays_to_parameters(
        typing.cast(NDArrays, ray.get(lrnr.get_parameters.remote()))
    )

    logger.info("Loading FL components.")

    # Strategy and server.
    strat: Optional[Strategy] = None
    serv: Optional[Server] = None

    xeval_kwargs = {
        "distribution": dist,
    }

    if cfg.get("archi", {}).get("strategy") is not None:
        logger.debug(f"Loading strategy {cfg.archi.strategy._target_}.")
        strat = hydra.utils.instantiate(
            cfg.archi.strategy,
            initial_parameters=init_parameters,
            num_rounds=cfg.fl.num_rounds,
            **(xeval_kwargs if isinstance(strat, FedXeval) else {}),
            _convert_="partial",
        )
        if cfg.get("archi", {}).get("server"):
            logger.debug(f"Loading server {cfg.archi.server._target_}.")
            serv = hydra.utils.instantiate(
                cfg.archi.server,
                strategy=strat,
                client_manager=SimpleClientManager(),
                _convert_="partial",
            )

    logger.debug(f"Loading client-side learners {cfg.learner._target_}.")

    # Poisoning instructions

    poisoning_ins: PoisonIns | None = None
    if any("attacker" in client for client in datasets.keys()):
        sel = str(cfg.attacker.poisoning)
        n_rounds: int = cfg.fl.num_rounds
        base, rnds = parse_poisoning_selector(sel, n_rounds)
        poisoning_ins = PoisonIns(
            cfg.attacker.target, base, rnds, cfg.attacker.poison_eval
        )

    lrnrs: Dict[str, ray.ActorID] = {}
    for client in datasets.keys():
        _ins = poisoning_ins if "attacker" in client else None
        lrnrs[client] = hydra.utils.instantiate(
            cfg.learner,
            *datasets[client],
            cid=client,
            seed=seed,
            poisoning_ins=_ins,
            _convert_="partial",
        )

    def client_fn(client_id: str) -> IDSClient:
        return hydra.utils.instantiate(
            cfg.archi.client,
            client_id,
            lrnrs[client_id],
            _convert_="partial",
        )

    logger.info("Starting simulation.")

    hist: History = flwr.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=datasets.keys(),
        client_resources=client_resources,
        server=serv,
        strategy=strat,
        config=flwr.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        keep_initialised=True,
    )

    # Exports.
    # --------
    # Save stats and other data to the current working directory.

    logger.info(f"Exporting data to {Path.cwd()}.")
    logger.debug("Exporting stats to ./stats.json.")
    stats = dict(sorted(stats.items()))
    Path("./stats.json").write_text(json.dumps(stats, indent=4))

    logger.debug("Exporting metrics to ./metrics.json.")
    dist_metrics: Dict[str, List[Tuple[int, MetricDict]]] = {
        cid: [(rnd, json.loads(str(serialized))) for rnd, serialized in ret]
        for cid, ret in hist.metrics_distributed.items()  # type: ignore
    }
    Path("./metrics.json").write_text(
        json.dumps(dict(sorted(dist_metrics.items())), indent=4)
    )

    logger.debug("Exporting cluster distribution to ./distribution.json.")
    Path("./distribution.json").write_text(json.dumps(dist, indent=4))

    if isinstance(strat, FedXeval) and strat.RS is not None:
        logger.debug("Exporting rounds clustering to ./clusters.json.")
        Path("./clusters.json").write_text(
            json.dumps(
                {rnd: info.clusters for rnd, info in strat.RS.hist.rounds.items()},
                indent=4,
            )
        )


if __name__ == "__main__":
    run()
