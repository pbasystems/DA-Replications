from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from utils.logger import get_logger

try:
    import wandb
except ImportError:  # wandb is an optional integration
    wandb = None


class Reporter:
    def __init__(
        self,
        name: str,
        log_file: str | Path | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        self.logger = get_logger(name, log_file=log_file)
        self.use_wandb = use_wandb
        self._run = None
        cwd = os.environ.get("BASE_WORKING_DIR", str(Path.cwd()))
        self.temp_path = os.path.join(cwd, "temp")
        os.makedirs(self.temp_path, exist_ok=True)

        if self.use_wandb:
            if wandb is None:
                self.logger.warning(
                    "wandb is not installed; continuing without wandb logging."
                )
                self.use_wandb = False
            elif not wandb_project:
                self.logger.warning(
                    "wandb_project not set; continuing without wandb logging."
                )
                self.use_wandb = False
            else:
                try:
                    self._run = wandb.init(
                        project=wandb_project, name=wandb_run_name, config=wandb_config
                    )
                except Exception as exc:  # noqa: BLE001 - wandb init is best-effort
                    self.logger.warning(
                        "Failed to initialise wandb run (%s); continuing without wandb logging.",
                        exc,
                    )
                    self.use_wandb = False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        formatted = " - ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        prefix = f"Step {step} - " if step is not None else ""
        self.logger.info("%s%s", prefix, formatted)

        if self.use_wandb and wandb is not None:
            wandb.log(metrics, step=step)

    def log_model(self, model: torch.nn.Module, step: int | None = None) -> None:
        if self.use_wandb and wandb is not None:
            torch.save(
                model.state_dict(), os.path.join(self.temp_path, f"model_{step}.pth")
            )
            artifact = wandb.Artifact(name=f"model_{step}", type="model")
            artifact.add_file(
                local_path=os.path.join(self.temp_path, f"model_{step}.pth"),
                name=f"model_{step}.pth",
            )
            wandb.log_artifact(artifact)

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def finish(self) -> None:
        if self.use_wandb and wandb is not None:
            wandb.finish()
