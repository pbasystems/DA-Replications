from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from lstm_dann.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from lstm_dann.Loss import ClassificationLoss, RegressionLoss
from lstm_dann.Loss.Score import Score
from lstm_dann.Tester import Tester
from lstm_dann.Trainer import Trainer
from src.pipelines.base import BasePipeline
from utils.seed import seed_everything


def compute_dataset_feature_stats(root_dir: str | Path, fd: str) -> FeatureStats:
    train_df = pd.read_csv(
        Path(root_dir) / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test_df = pd.read_csv(
        Path(root_dir) / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    values = pd.concat(
        [train_df[FEATURE_COLS], test_df[FEATURE_COLS]], axis=0
    ).to_numpy()
    return FeatureStats(min=values.min(axis=0), max=values.max(axis=0))


def last_window_per_engine(dataset: CMAPSSDataset) -> Subset:
    unit_numbers = dataset.unit_numbers
    _, last_pos_reversed = np.unique(unit_numbers[::-1], return_index=True)
    last_indices = len(unit_numbers) - 1 - last_pos_reversed
    return Subset(dataset, sorted(last_indices))


class LSTMDANNPipeline(BasePipeline):
    def _build_datasets(self) -> tuple[CMAPSSDataset, CMAPSSDataset, DataLoader]:
        ds_cfg = self.cfg.dataset
        data_dir = Path(ds_cfg.data_dir)
        source_fd = ds_cfg.source_fd
        target_fd = ds_cfg.target_fd
        window_size = ds_cfg.window_size
        r_early = ds_cfg.r_early

        if ds_cfg.get("combine_splits_for_stats", True):
            source_stats = compute_dataset_feature_stats(data_dir, source_fd)
            target_stats = compute_dataset_feature_stats(data_dir, target_fd)
        else:
            source_stats = None
            target_stats = None

        source_dataset = CMAPSSDataset(
            root_dir=data_dir,
            fd=source_fd,
            split="train",
            window_size=window_size,
            r_early=r_early,
            feature_stats=source_stats,
        )

        target_dataset = CMAPSSDataset(
            root_dir=data_dir,
            fd=target_fd,
            split="train",
            window_size=window_size,
            r_early=r_early,
            feature_stats=target_stats,
        )

        target_test_dataset = CMAPSSDataset(
            root_dir=data_dir,
            fd=target_fd,
            split="test",
            window_size=window_size,
            r_early=r_early,
            feature_stats=target_stats or target_dataset.feature_stats,
        )

        target_test_subset = last_window_per_engine(target_test_dataset)
        target_test_dataloader = DataLoader(
            target_test_subset,
            batch_size=len(target_test_subset),
            shuffle=False,
        )

        return source_dataset, target_dataset, target_test_dataloader

    def run_trial(
        self,
        seed: int,
        source_dataset: CMAPSSDataset,
        target_dataset: CMAPSSDataset,
        target_test_dataloader: DataLoader,
        reporter: Any,
    ) -> dict[str, Any]:
        seed_everything(seed)

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
        lr_source_reg = opt_cfg.lr_source_reg
        lr_domain_class = opt_cfg.lr_domain_class
        l2_reg = opt_cfg.l2_reg

        regression_optimizer = torch.optim.SGD(
            [
                {"params": model.feature_extractor.parameters(), "lr": lr_source_reg},
                {
                    "params": model.regressor.parameters(),
                    "lr": lr_source_reg,
                    "weight_decay": l2_reg,
                },
            ]
        )
        domain_optimizer = torch.optim.SGD(
            [
                {"params": model.feature_extractor.parameters(), "lr": lr_domain_class},
                {
                    "params": model.classifier.parameters(),
                    "lr": lr_domain_class,
                    "weight_decay": l2_reg,
                },
            ]
        )

        milestones = list(opt_cfg.get("scheduler_milestones", [100]))
        gamma = opt_cfg.get("scheduler_gamma", 0.1)
        regression_scheduler = MultiStepLR(
            regression_optimizer, milestones=milestones, gamma=gamma
        )
        domain_scheduler = MultiStepLR(
            domain_optimizer, milestones=milestones, gamma=gamma
        )

        loss_cfg = self.cfg.loss
        regression_loss = RegressionLoss(p=loss_cfg.get("regression_p", 1))
        classification_loss = ClassificationLoss()
        score_fn = Score(
            a_1=loss_cfg.get("score_a1", 13.0), a_2=loss_cfg.get("score_a2", 10.0)
        )

        trainer = Trainer(
            model=model,
            regression_optimizer=regression_optimizer,
            domain_optimizer=domain_optimizer,
            regression_scheduler=regression_scheduler,
            domain_scheduler=domain_scheduler,
            source_dataloader=source_dataloader,
            target_dataloader=target_dataloader,
            regression_loss=regression_loss,
            classification_loss=classification_loss,
            score_loss=score_fn,
            device=self.device,
            max_grad_norm=self.cfg.trainer.get("max_grad_norm", 1.0),
            reporter=reporter,
        )

        trainer.train(epochs=self.cfg.trainer.epochs)

        tester = Tester(model=model, score_loss=score_fn, device=self.device)
        return tester.evaluate(target_test_dataloader)

    def run(self) -> dict[str, Any]:
        source_dataset, target_dataset, target_test_dataloader = self._build_datasets()
        trials = self.cfg.get("trials", 1)

        if trials == 1:
            reporter = self.create_reporter()
            seed = self.cfg.get("seed", 42)
            metrics = self.run_trial(
                seed, source_dataset, target_dataset, target_test_dataloader, reporter
            )
            reporter.log_metrics(metrics)
            reporter.finish()
            print(
                f"\nResults: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  Score={metrics['score']:.2f}"
            )
            return metrics
        else:
            main_reporter = self.create_reporter()
            results = []
            for trial_idx in range(trials):
                seed = trial_idx
                trial_reporter = self.create_reporter(name_suffix=f"trial_{trial_idx}")
                metrics = self.run_trial(
                    seed,
                    source_dataset,
                    target_dataset,
                    target_test_dataloader,
                    trial_reporter,
                )
                trial_reporter.finish()
                print(
                    f"Trial {trial_idx + 1}/{trials}: "
                    f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  Score={metrics['score']:.2f}"
                )
                results.append(metrics)

            rmses = np.array([r["rmse"] for r in results])
            maes = np.array([r["mae"] for r in results])
            scores = np.array([r["score"] for r in results])

            summary = {
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std()),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "trials": trials,
            }

            source_fd = self.cfg.dataset.source_fd
            target_fd = self.cfg.dataset.target_fd
            print(f"\n{source_fd} -> {target_fd} over {trials} trials:")
            print(f"RMSE:  {summary['rmse_mean']:.2f} +/- {summary['rmse_std']:.2f}")
            print(f"MAE:   {summary['mae_mean']:.2f} +/- {summary['mae_std']:.2f}")
            print(f"Score: {summary['score_mean']:.2f} +/- {summary['score_std']:.2f}")

            main_reporter.log_metrics(summary)
            main_reporter.finish()
            return summary
