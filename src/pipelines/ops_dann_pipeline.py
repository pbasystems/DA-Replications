from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from ops_dann.Dataset import NCMAPSSDataset
from ops_dann.Loss import DomainLoss, RULLoss
from ops_dann.Loss.Score import Score
from ops_dann.Tester import Tester
from ops_dann.Trainer import Trainer
from src.pipelines.base import BasePipeline


class OPSDANNPipeline(BasePipeline):
    def _build_datasets(self) -> tuple[NCMAPSSDataset, NCMAPSSDataset]:
        ds_cfg = self.cfg.dataset
        h5_path = Path(ds_cfg.h5_path)
        source_fc = ds_cfg.source_fc
        target_fc = ds_cfg.target_fc

        source_dataset = NCMAPSSDataset(
            h5_path=h5_path,
            flight_class=source_fc,
            window_size=ds_cfg.get("window_size", 50),
            stride=ds_cfg.get("stride", 1),
            downsample_factor=ds_cfg.get("downsample_factor", 10),
            phase_threshold=ds_cfg.get("phase_threshold", 0.5),
            median_filter_length=ds_cfg.get("median_filter_length", 51),
        )

        target_dataset = NCMAPSSDataset(
            h5_path=h5_path,
            flight_class=target_fc,
            window_size=ds_cfg.get("window_size", 50),
            stride=ds_cfg.get("stride", 1),
            downsample_factor=ds_cfg.get("downsample_factor", 10),
            phase_threshold=ds_cfg.get("phase_threshold", 0.5),
            median_filter_length=ds_cfg.get("median_filter_length", 51),
            feature_stats=source_dataset.feature_stats,
        )

        return source_dataset, target_dataset

    def run_trial(
        self,
        seed: int,
        source_dataset: NCMAPSSDataset,
        target_dataset: NCMAPSSDataset,
        reporter: Any,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        batch_size = self.cfg.dataset.batch_size
        shuffle = self.cfg.dataset.get("shuffle_train", True)

        source_dataloader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=shuffle
        )
        target_dataloader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=shuffle
        )

        model = instantiate(self.cfg.model).to(self.device)

        opt_cfg = self.cfg.optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.get("lr", 0.05),
            momentum=opt_cfg.get("momentum", 0.5),
        )

        loss_cfg = self.cfg.loss
        rul_loss = RULLoss()
        domain_loss = DomainLoss()
        score_fn = Score(
            alpha_over=loss_cfg.get("score_a1", 13.0),
            alpha_under=loss_cfg.get("score_a2", 10.0),
        )

        trainer = Trainer(
            model=model,
            source_dataloader=source_dataloader,
            target_dataloader=target_dataloader,
            optimizer=optimizer,
            rul_loss=rul_loss,
            domain_loss=domain_loss,
            tradeoff=self.cfg.trainer.get("tradeoff", 1.0),
            device=self.device,
            num_phases=self.cfg.trainer.get("num_phases", 3),
            reporter=reporter,
        )

        trainer.train(epochs=self.cfg.trainer.epochs)

        tester = Tester(
            model=model,
            score_loss=score_fn,
            device=self.device,
            num_phases=self.cfg.trainer.get("num_phases", 3),
        )
        return tester.evaluate(target_dataloader)

    def run(self) -> dict[str, Any]:
        source_dataset, target_dataset = self._build_datasets()
        trials = self.cfg.get("trials", 1)

        if trials == 1:
            reporter = self.create_reporter()
            seed = self.cfg.get("seed", 42)
            metrics = self.run_trial(seed, source_dataset, target_dataset, reporter)
            reporter.log_metrics(metrics)
            reporter.finish()
            score_mean = metrics["score"] / metrics["n_samples"]
            print(
                f"\nResults: RMSE={metrics['rmse_cycles']:.2f}  "
                f"MAE={metrics['mae_cycles']:.2f}  s-score={score_mean:.2f}"
            )
            return metrics
        else:
            main_reporter = self.create_reporter()
            results = []
            for trial_idx in range(trials):
                seed = trial_idx
                trial_reporter = self.create_reporter(name_suffix=f"trial_{trial_idx}")
                metrics = self.run_trial(
                    seed, source_dataset, target_dataset, trial_reporter
                )
                trial_reporter.finish()
                score_mean = metrics["score"] / metrics["n_samples"]
                print(
                    f"Trial {trial_idx + 1}/{trials}: "
                    f"RMSE={metrics['rmse_cycles']:.2f}  s-score={score_mean:.2f}"
                )
                results.append({"rmse": metrics["rmse_cycles"], "score": score_mean})

            rmses = np.array([r["rmse"] for r in results])
            scores = np.array([r["score"] for r in results])

            summary = {
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "trials": trials,
            }

            source_fc = self.cfg.dataset.source_fc
            target_fc = self.cfg.dataset.target_fc
            print(f"\nFC{source_fc} -> FC{target_fc} over {trials} trials:")
            print(f"RMSE:    {summary['rmse_mean']:.2f} +/- {summary['rmse_std']:.2f}")
            print(
                f"s-score: {summary['score_mean']:.2f} +/- {summary['score_std']:.2f}"
            )

            main_reporter.log_metrics(summary)
            main_reporter.finish()
            return summary
