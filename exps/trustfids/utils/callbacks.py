"""Mattermost integration for TrustFIDS.

This module provides a simple interface to send messages to a Mattermost, using Hydra's
callback mechanism. It provides callback classes for when jobs end.
"""

import json
import textwrap
import time
from pathlib import Path
from typing import Any

import requests
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class MattermostCallback(Callback):
    """Mattermost incomming webhook callback.

    This class implements Hydra's callback mechanism to send messages to a Mattermost
    channel on certain events, such as when a job ends.

    For reference, see: https://hydra.cc/docs/experimental/callbacks/

    TODO: link to the generated files (https://github.com/microsoft/vscode-remote-release/issues/656)
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.jobs = []
        self.timings = {}

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when run starts."""
        pass

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when multirun starts."""
        pass

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when a job starts."""
        id = config.hydra.job.get("id", None) or config.hydra.job.name
        self.timings[id] = time.time()

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Call when a job ends."""
        id = config.hydra.job.get("id", None) or config.hydra.job.name
        t = time.time() - self.timings[id]
        self.jobs.append((job_return, t))

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when run ends."""
        self._send_notification(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when multirun ends."""
        self._send_notification(config, **kwargs)

    def _send_notification(self, config: DictConfig, **kwargs: Any):
        """Send a message to Mattermost.

        Args:
            config: Hydra's configuration object.
        """
        jobs = "\n".join(
            [
                f"| **{j.hydra_cfg.hydra.runtime.choices['archi']}**  | `{j.status}` | {t:.2f} seconds | `{j.overrides}` | `{j.working_dir}` |"
                for j, t in self.jobs
            ]
        )

        r = requests.post(
            url=self.url,
            data=json.dumps(
                {
                    "text": textwrap.dedent(
                        f"""
                        Hey @here!
                        The experiment has finished. The following jobs were run:
                        
                        | Job | Status | Time | Overrides | Results |
                        | --- | ------ | ---- | --------- | ------- |
                        """
                    )
                    + jobs,
                }
            ),
        )
        r.raise_for_status()


class PlotCallback:
    """Plot callback.

    This class implements Hydra's callback mechanism to plot the results of a
    multirun job.

    For reference, see: https://hydra.cc/docs/experimental/callbacks/
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.jobs = []

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Send a message to Mattermost when a job ends.

        Args:
            config: Hydra's configuration object.
            job_return: Hydra's job return object.
        """
        self.jobs.append(job_return)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Send a message to Mattermost when a multirun job ends.

        Args:
            config: Hydra's configuration object.
        """
        self._plot(config, **kwargs)

    def _plot(self, config: DictConfig, **kwargs: Any):
        """Plot the results of a multirun job.

        Args:
            config: Hydra's configuration object.
        """
        results = [
            (j.cfg, json.load(open(Path(j.working_dir) / "metrics.json", "r")))
            for j in self.jobs
        ]


# Flwr documentation
# ------------------
#
# @dataclass
# class JobReturn:
#     overrides: Optional[Sequence[str]] = None
#     cfg: Optional[DictConfig] = None
#     hydra_cfg: Optional[DictConfig] = None
#     working_dir: Optional[str] = None
#     task_name: Optional[str] = None
#     status: JobStatus = JobStatus.UNKNOWN
#     _return_value: Any = None
