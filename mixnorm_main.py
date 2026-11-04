from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from MixNorm.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from MixNorm.EarlyStopper import EarlyStopping
from MixNorm.Loss import RegressionLoss
from MixNorm.Loss.Score import Score
from MixNorm.Model import MixNorm
from MixNorm.Tester import Tester
from MixNorm.Trainer import Trainer
from utils.reporter import Reporter
from utils.seed import seed_everything


def compute_dataset_feature_stats(root_dir, fd):
    train_df = pd.read_csv(
        Path(root_dir) / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test_df = pd.read_csv(
        Path(root_dir) / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    values = pd.concat([train_df[FEATURE_COLS], test_df[FEATURE_COLS]], axis=0).to_numpy()
    return FeatureStats(min=values.min(axis=0), max=values.max(axis=0))


def split_by_engine(dataset, val_ratio: float = 0.10, seed: int = 42):
    unique_units = np.unique(dataset.unit_numbers)
    rng = np.random.RandomState(seed)
    shuffled_units = rng.permutation(unique_units)

    n_val = round(len(unique_units) * val_ratio)
    val_units = set(shuffled_units[:n_val])
    train_units = set(shuffled_units[n_val:])

    train_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in train_units]
    val_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in val_units]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def last_window_per_engine(dataset):
    unit_numbers = dataset.unit_numbers
    _, last_pos_reversed = np.unique(unit_numbers[::-1], return_index=True)
    last_indices = len(unit_numbers) - 1 - last_pos_reversed
    return Subset(dataset, sorted(last_indices))


def main(
    source_fd: str = "FD001",
    target_fd: str = "FD002",
    epochs: int = 20,
    seed: int = 42,
    batch_size: int = 256,
    lr: float = 0.01,
    weight_decay: float = 0.01,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = Path.cwd() / "Data" / "CMAPSS"
    source_stats = compute_dataset_feature_stats(root_dir, source_fd)
    target_stats = compute_dataset_feature_stats(root_dir, target_fd)

    source_dataset = CMAPSSDataset(
        root_dir=root_dir,
        fd=source_fd,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=source_stats,
    )
    target_test_dataset = CMAPSSDataset(
        root_dir=root_dir,
        fd=target_fd,
        split="test",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )

    source_train_subset, source_val_subset = split_by_engine(
        source_dataset, val_ratio=0.10, seed=seed
    )
    target_test_subset = last_window_per_engine(target_test_dataset)

    source_train_dataloader = DataLoader(
        source_train_subset, batch_size=batch_size, shuffle=True
    )
    source_val_dataloader = DataLoader(
        source_val_subset, batch_size=batch_size, shuffle=False
    )
    target_test_dataloader = DataLoader(
        target_test_subset, batch_size=len(target_test_subset), shuffle=False
    )

    print(
        f"Data loaded: Source ({source_fd}) train={len(source_train_subset)}, "
        f"val={len(source_val_subset)} | Target ({target_fd}) test_engines={len(target_test_subset)}"
    )

    model = MixNorm(
        input_size=24,
        lstm_hidden_size=128,
        f_size=64,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        regressor_first_hidden_size=32,
        regressor_second_hidden_size=0,
        mixing_scale=0.05,
        moving_speed=0.001,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.5)], gamma=0.1)
    score_fn = Score(a_1=13.0, a_2=10.0)
    early_stopper = EarlyStopping(patience=15)
    reporter = Reporter(name="MixNorm.Main", use_wandb=False)

    trainer = Trainer(
        model=model,
        source_train_dataloader=source_train_dataloader,
        source_val_dataloader=source_val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        regression_loss=RegressionLoss(p=1),
        score_loss=score_fn,
        device=device,
        early_stopper=early_stopper,
        reporter=reporter,
    )

    print(f"\n--- Training Mix-Norm on Source Domain ({source_fd}) for {epochs} epochs ---")
    trainer.train(epochs=epochs)

    tester = Tester(model=model, score_loss=score_fn, device=device, reporter=reporter)

    print("\n--- Evaluating Target Domain with Mix-Norm Test-Time Adaptation ---")
    sweep_scales = [0.01, 0.05, 0.10, 0.20, 0.50]
    comparison = tester.compare_settings(target_test_dataloader, mixing_scales=sweep_scales)

    print("\n=== Summary Comparison: Mix-Norm on CMAPSS ===")
    print(f"Task: {source_fd} -> {target_fd}")
    print(f"{'Method / Configuration':<38} | {'RMSE':<8} | {'MAE':<8} | {'Score':<10}")
    print("-" * 74)
    for name, m in comparison.items():
        print(f"{name:<38} | {m['rmse']:<8.2f} | {m['mae']:<8.2f} | {m['score']:<10.2f}")

    return comparison


if __name__ == "__main__":
    main()
