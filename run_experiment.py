import os
from pathlib import Path
from this import s
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from lstm_dann.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from lstm_dann.EarlyStopper import EarlyStopping
from lstm_dann.Loss import ClassificationLoss, RegressionLoss
from lstm_dann.Loss.Score import Score
from lstm_dann.Model import LSTM_DANN
from lstm_dann.Tester import Tester
from lstm_dann.Trainer import Trainer
from utils.reporter import Reporter
from utils.seed import seed_everything

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = Path.cwd() / "Data" / "CMAPSS"
TEMP_DIR = Path.cwd() / "temp"
os.environ["BASE_WORKING_DIR"] = str(Path.cwd())
N_TRIALS = 1
EPOCHS = 1


def split_by_engine(dataset, val_ratio: float = 0.10, seed: int = 42):
    """
    Splits a CMAPSSDataset into train and validation subsets at the engine/unit level
    to prevent temporal leakage across consecutive time windows of the same engine.
    """
    unique_units = np.unique(dataset.unit_numbers)
    rng = np.random.RandomState(seed)
    shuffled_units = rng.permutation(unique_units)

    n_val = round(len(unique_units) * val_ratio)
    val_units = set(shuffled_units[:n_val])
    train_units = set(shuffled_units[n_val:])

    train_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in train_units]
    val_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in val_units]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def compute_dataset_feature_stats(root_dir, fd):
    train_df = pd.read_csv(
        Path(root_dir) / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test_df = pd.read_csv(
        Path(root_dir) / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    values = pd.concat([train_df[FEATURE_COLS], test_df[FEATURE_COLS]], axis=0).to_numpy()
    return FeatureStats(min=values.min(axis=0), max=values.max(axis=0))


def last_window_per_engine(dataset):
    unit_numbers = dataset.unit_numbers
    _, last_pos_reversed = np.unique(unit_numbers[::-1], return_index=True)
    last_indices = len(unit_numbers) - 1 - last_pos_reversed
    return Subset(dataset, sorted(last_indices))


def run_trial(
    seed,
    source_train_dataset,
    source_val_dataset,
    target_train_dataset,
    target_val_dataset,
    target_test_dataloader,
    config: DictConfig,
):
    seed_everything(config.seed)

    source_train_dataloader = DataLoader(
        source_train_dataset, batch_size=config.lstm_dann.batch_size, shuffle=True
    )
    source_val_dataloader = DataLoader(
        source_val_dataset, batch_size=config.lstm_dann.batch_size, shuffle=False
    )
    target_train_dataloader = DataLoader(
        target_train_dataset, batch_size=config.lstm_dann.batch_size, shuffle=True
    )
    target_val_dataloader = DataLoader(
        target_val_dataset, batch_size=config.lstm_dann.batch_size, shuffle=False
    )

    model = LSTM_DANN(
        input_size=config.lstm_dann.Model.input_size,
        lstm_hidden_size=config.lstm_dann.Model.lstm_hidden_size,
        f_size=config.lstm_dann.Model.f_size,
        num_layers=config.lstm_dann.Model.num_layers,
        lstm_dropout=config.lstm_dann.Model.lstm_dropout,
        regressor_dropout=config.lstm_dann.Model.regressor_dropout,
        classifier_dropout=config.lstm_dann.Model.classifier_dropout,
        alpha=config.lstm_dann.Model.alpha,
        regressor_first_hidden_size=config.lstm_dann.Model.regressor_first_hidden_size,
        regressor_second_hidden_size=config.lstm_dann.Model.regressor_second_hidden_size,
        classifier_first_hidden_size=config.lstm_dann.Model.classifier_first_hidden_size,
        classifier_second_hidden_size=config.lstm_dann.Model.classifier_second_hidden_size,
    ).to(DEVICE)

    regression_optimizer = torch.optim.SGD(
        [
            {
                "params": model.feature_extractor.parameters(),
                "lr": config.lstm_dann.optimizer.lr_source_reg,
            },
            {
                "params": model.regressor.parameters(),
                "lr": config.lstm_dann.optimizer.lr_source_reg,
                "weight_decay": config.lstm_dann.regularization,
            },
        ]
    )
    domain_optimizer = torch.optim.SGD(
        [
            {
                "params": model.feature_extractor.parameters(),
                "lr": config.lstm_dann.optimizer.lr_domain_class,
            },
            {
                "params": model.classifier.parameters(),
                "lr": config.lstm_dann.optimizer.lr_domain_class,
                "weight_decay": config.lstm_dann.regularization,
            },
        ]
    )
    regression_scheduler = MultiStepLR(regression_optimizer, milestones=[100], gamma=0.1)
    domain_scheduler = MultiStepLR(domain_optimizer, milestones=[100], gamma=0.1)

    score_fn = Score(a_1=13, a_2=10)
    reporter = Reporter(
        name=f"LSTM_DANN.Experiments.{seed + 1}",
        use_wandb=True,
        wandb_project="DA-Replications|LSTM-DANN|CMAPSS",
        wandb_run_name=f"{config.lstm_dann.source_fd}-to-{config.lstm_dann.target_fd} | Trial {seed + 1}",
        wandb_config=cast("dict[str, Any]", OmegaConf.to_container(config.lstm_dann)),
    )

    early_stopper = EarlyStopping(patience=20)

    trainer = Trainer(
        model=model,
        regression_optimizer=regression_optimizer,
        domain_optimizer=domain_optimizer,
        regression_scheduler=regression_scheduler,
        domain_scheduler=domain_scheduler,
        source_train_dataloader=source_train_dataloader,
        target_train_dataloader=target_train_dataloader,
        source_val_dataloader=source_val_dataloader,
        target_val_dataloader=target_val_dataloader,
        regression_loss=RegressionLoss(p=1),
        classification_loss=ClassificationLoss(),
        score_loss=score_fn,
        device=DEVICE,
        early_stopper=early_stopper,
        reporter=reporter,
    )
    trainer.train(epochs=EPOCHS)

    tester = Tester(model=model, score_loss=score_fn, device=DEVICE, reporter=reporter)
    metrics = tester.evaluate(target_test_dataloader)
    reporter.finish()
    return metrics


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    source_stats = compute_dataset_feature_stats(DATASET_PATH, cfg.lstm_dann.source_fd)
    target_stats = compute_dataset_feature_stats(DATASET_PATH, cfg.lstm_dann.target_fd)

    source_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=cfg.lstm_dann.source_fd,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=source_stats,
    )
    target_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=cfg.lstm_dann.target_fd,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )
    target_test_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=cfg.lstm_dann.target_fd,
        split="test",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )
    source_train_subset, source_val_subset = split_by_engine(
        source_dataset, val_ratio=0.10, seed=cfg.seed
    )
    target_train_subset, target_val_subset = split_by_engine(
        target_dataset, val_ratio=0.10, seed=cfg.seed
    )
    target_test_subset = last_window_per_engine(target_test_dataset)
    target_test_dataloader = DataLoader(
        target_test_subset, batch_size=len(target_test_subset), shuffle=False
    )

    results = []
    for trial in range(N_TRIALS):
        metrics = run_trial(
            seed=trial,
            source_train_dataset=source_train_subset,
            source_val_dataset=source_val_subset,
            target_train_dataset=target_train_subset,
            target_val_dataset=target_val_subset,
            target_test_dataloader=target_test_dataloader,
            config=cfg,
        )
        print(
            f"Trial {trial + 1}/{N_TRIALS}: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  Score={metrics['score']:.2f}"
        )
        results.append(metrics)

    rmses = np.array([r["rmse"] for r in results])
    maes = np.array([r["mae"] for r in results])
    scores = np.array([r["score"] for r in results])

    print(f"\n{cfg.lstm_dann.source_fd} -> {cfg.lstm_dann.target_fd} over {N_TRIALS} trials:")
    print(f"RMSE:  {rmses.mean():.2f} +/- {rmses.std():.2f} ")
    print(f"MAE:   {maes.mean():.2f} +/- {maes.std():.2f}")
    print(f"Score: {scores.mean():.2f} +/- {scores.std():.2f}")
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(TEMP_DIR / "results.csv", "a") as f:
        f.write(
            f"{cfg.lstm_dann.source_fd},{cfg.lstm_dann.target_fd},{rmses.mean():.2f},{rmses.std():.2f},{maes.mean():.2f},{maes.std():.2f},{scores.mean():.2f},{scores.std():.2f}\n"
        )
    print("Results saved to results.csv")


if __name__ == "__main__":
    main()
