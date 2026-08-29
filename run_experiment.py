from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from lstm_dann.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from lstm_dann.Loss import ClassificationLoss, RegressionLoss
from lstm_dann.Loss.Score import Score
from lstm_dann.Model import LSTM_DANN
from lstm_dann.Tester import Tester
from lstm_dann.Trainer import Trainer
from utils.reporter import Reporter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = Path.cwd() / "Data" / "CMAPSS"
SOURCE_FD, TARGET_FD = "FD001", "FD002"
N_TRIALS = 10
EPOCHS = 200


def compute_dataset_feature_stats(root_dir, fd):
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


def last_window_per_engine(dataset):
    unit_numbers = dataset.unit_numbers
    _, last_pos_reversed = np.unique(unit_numbers[::-1], return_index=True)
    last_indices = len(unit_numbers) - 1 - last_pos_reversed
    return Subset(dataset, sorted(last_indices))


def run_trial(seed, source_dataset, target_dataset, target_test_dataloader):
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_dataloader = DataLoader(source_dataset, batch_size=256, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=256, shuffle=True)

    model = LSTM_DANN(
        input_size=24,
        hidden_size=64,
        f_size=32,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        classifier_dropout=0.3,
        alpha=0.8,
    ).to(DEVICE)

    l2_reg, lr_source_reg, lr_domain_class = 0.01, 0.01, 0.01

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
    regression_scheduler = MultiStepLR(
        regression_optimizer, milestones=[100], gamma=0.1
    )
    domain_scheduler = MultiStepLR(domain_optimizer, milestones=[100], gamma=0.1)

    score_fn = Score(a_1=13, a_2=10)
    reporter = Reporter(name=f"trial-{seed}", use_wandb=False)

    trainer = Trainer(
        model=model,
        regression_optimizer=regression_optimizer,
        domain_optimizer=domain_optimizer,
        regression_scheduler=regression_scheduler,
        domain_scheduler=domain_scheduler,
        source_dataloader=source_dataloader,
        target_dataloader=target_dataloader,
        regression_loss=RegressionLoss(p=1),
        classification_loss=ClassificationLoss(),
        score_loss=score_fn,
        device=DEVICE,
        reporter=reporter,
    )
    trainer.train(epochs=EPOCHS)

    tester = Tester(model=model, score_loss=score_fn, device=DEVICE)
    return tester.evaluate(target_test_dataloader)


def main():
    source_stats = compute_dataset_feature_stats(DATASET_PATH, SOURCE_FD)
    target_stats = compute_dataset_feature_stats(DATASET_PATH, TARGET_FD)

    source_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=SOURCE_FD,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=source_stats,
    )
    target_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=TARGET_FD,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )
    target_test_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=TARGET_FD,
        split="test",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )
    target_test_subset = last_window_per_engine(target_test_dataset)
    target_test_dataloader = DataLoader(
        target_test_subset, batch_size=len(target_test_subset), shuffle=False
    )

    results = []
    for trial in range(N_TRIALS):
        metrics = run_trial(
            trial, source_dataset, target_dataset, target_test_dataloader
        )
        print(
            f"Trial {trial + 1}/{N_TRIALS}: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  Score={metrics['score']:.2f}"
        )
        results.append(metrics)

    rmses = np.array([r["rmse"] for r in results])
    scores = np.array([r["score"] for r in results])

    print(f"\n{SOURCE_FD} -> {TARGET_FD} over {N_TRIALS} trials:")
    print(f"RMSE:  {rmses.mean():.2f} +/- {rmses.std():.2f} ")
    print(f"Score: {scores.mean():.2f} +/- {scores.std():.2f}")


if __name__ == "__main__":
    main()
