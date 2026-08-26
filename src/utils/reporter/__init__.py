from __future__ import annotations

from pathlib import Path
from typing import Any

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

        if self.use_wandb:
            if wandb is None:
                self.logger.warning("wandb is not installed; continuing without wandb logging.")
                self.use_wandb = False
            elif not wandb_project:
                self.logger.warning("wandb_project not set; continuing without wandb logging.")
                self.use_wandb = False
            else:
                try:
                    self._run = wandb.init(
                        project=wandb_project, name=wandb_run_name, config=wandb_config
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to initialise wandb run (%s); continuing without wandb logging.", exc
                    )
                    self.use_wandb = False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        formatted = " - ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        prefix = f"Step {step} - " if step is not None else ""
        self.logger.info("%s%s", prefix, formatted)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def finish(self) -> None:
        if self.use_wandb:
            wandb.finish()
