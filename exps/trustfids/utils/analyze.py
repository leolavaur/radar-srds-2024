"""Analysis toolbox for trustfids experiments.

This module contains functions to analyze the results of experiments run with the
`trustfids` package. The module is expected to be used as a executable script, e.g.:
```bash
python -m trustfids.utils.analyze --help
```
"""
import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
import omegaconf
from omegaconf import DictConfig, ListConfig, OmegaConf
from trustfids.client import create_model, load_siloed_data


class CommandError(Exception):
    """Exception raised when an invalid command is used."""

    pass


def main() -> None:
    # Fix VSCode debugger's parameter passing behavior
    if len(sys.argv) > 1 and " " in sys.argv[1]:
        sys.argv = sys.argv[:1] + sys.argv[1].split(" ")

    parser = argparse.ArgumentParser(description="Trust-FIDS analysis toolbox.")
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    # Subparser for the `plot` command
    plot_parser = subparsers.add_parser("plot", help="Plot results from previous runs.")

    plot_options_group = plot_parser.add_argument_group("plot options")
    plot_options_group.add_argument(
        "-m",
        "--metric",
        type=str,
        help="metric to plot",
        default="accuracy",
        dest="metric",
    )
    plot_options_group.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to save plots to",
        default=None,
        dest="save_path",
    )
    plot_options_group.add_argument(
        "-t",
        "--title",
        type=str,
        help="title of the plot",
        default="",
        dest="title",
    )
    plot_options_group.add_argument(
        "-n",
        "--multirun",
        action="store_true",
        help="treat `path` as a Hydra multirun directory",
        dest="multirun",
    )

    plot_type_group = plot_parser.add_mutually_exclusive_group(required=True)
    plot_type_group.add_argument(
        "--moustache",
        action="store_true",
        help="plot moustache (box) plot",
        dest="moustache",
    )
    plot_type_group.add_argument(
        "--comparison",
        action="store_true",
        help="plot comparison plot",
        dest="comparison",
    )

    plot_parser.add_argument(
        "path",
        type=str,
        nargs="+",
        help="list of paths to load results from",
    )

    # Subparser for the `train` command
    train_parser = subparsers.add_parser("train", help="Train locally a ML model.")

    # list of hydra overrides
    train_parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        help="list of hydra overrides",
        default=[],
        dest="overrides",
    )

    train_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to save metrics to (if ends with `.json`, considered a file path, direcotry otherwise)",
        default=None,
        dest="save_path",
    )

    train_parser.add_argument(
        "silos",
        type=str,
        nargs="*",
        help="list of silos to use for training",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------
    # Plotting commands
    # ----------------------------------------------------------------------------------

    if args.subparser_name == "plot":

        paths = []
        if args.multirun:
            if len(args.path) > 1:
                raise CommandError("Only one multirun directory can be specified.")

            p = Path(args.path[0])

            for run in p.iterdir():
                if run.is_dir():
                    paths.append(run)

        else:
            paths = [Path(p) for p in args.path]

        results: List[Tuple[DictConfig | ListConfig, Any, Path]] = []
        for p in paths:
            if not p.exists():
                raise CommandError(f"Path {p} does not exist.")
            wd = p
            if args.save_path:
                wd = Path(args.save_path)
                if not wd.exists():
                    raise CommandError(f"Path {wd} does not exist.")

            config = OmegaConf.load(p / ".hydra/config.yaml")
            results.append((config, json.load(open(p / "metrics.json", "r")), wd))

        if args.moustache:
            from trustfids.utils.plots import plot_moustache

            for config, metrics, wd in results:

                img_title = args.title if args.title else config.baseline.name

                img_path = (wd / f"{img_title}.png").as_posix()

                plot_moustache(
                    dist_metrics=metrics,
                    metric=args.metric,
                    title=img_title,
                    save_path=img_path,
                    mean=True,
                )

                print(f"Plot saved to {img_path}.")

        elif args.comparison:
            from trustfids.utils.plots import plot_comparison

            wd = results[0][2].parent

            plot_comparison(
                *[(c, m) for c, m, _ in results],  # type: ignore
                metric=args.metric,
                title=args.title if args.title else "Baseline comparison",
                save_path=(wd / "comparison.png").as_posix(),
            )

        else:
            raise CommandError("No valid plot type specified, see group `plot_type`.")

    # ----------------------------------------------------------------------------------
    # Local training commands
    # ----------------------------------------------------------------------------------

    elif args.subparser_name == "train":
        from hydra import compose, initialize
        from hydra.utils import instantiate
        from trustfids.utils.centralized import eval_attack_detection

        initialize(version_base=None, config_path="../conf", job_name="analysis_train")
        cfg = compose(
            config_name="config", overrides=["baseline=fedavg"] + args.overrides
        )

        metrics = eval_attack_detection(cfg, silos=[Path(p) for p in args.silos])

        print(metrics)
        if args.save_path is not None:
            if args.save_path.endswith(".json"):
                out, file = Path(args.save_path).parent, Path(args.save_path).name
            else:
                out, file = Path(args.save_path), "metrics.json"

            if not out.exists():
                out.mkdir(parents=True)

            (out / file).write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
