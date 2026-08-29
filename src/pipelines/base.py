from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from utils.reporter import Reporter


class BasePipeline(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.trainer.get("device", "auto"))

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def create_reporter(self, name_suffix: str = "") -> Reporter:
        logger_cfg = self.cfg.logger
        use_wandb = logger_cfg.get("use_wandb", False)
        wandb_project = logger_cfg.get("wandb_project", None)
        wandb_run_name = logger_cfg.get("wandb_run_name", None)
        if wandb_run_name and name_suffix:
            wandb_run_name = f"{wandb_run_name}_{name_suffix}"
        elif name_suffix and not wandb_run_name:
            wandb_run_name = f"{self.cfg.get('task_name', 'experiment')}_{name_suffix}"

        wandb_config = cast(
            "dict[str, Any]",
            OmegaConf.to_container(self.cfg, resolve=True) if use_wandb else None,
        )

        return Reporter(
            name=f"{self.cfg.get('task_name', 'pipeline')}{f'.{name_suffix}' if name_suffix else ''}",
            log_file=logger_cfg.get("log_file", None),
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_config=wandb_config,
            wandb_run_name=wandb_run_name,
        )

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the pipeline experiment."""
