from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from AdaBN.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from AdaBN.EarlyStopper import EarlyStopping
from AdaBN.Loss import RegressionLoss
from AdaBN.Loss.Score import Score
from AdaBN.Model import AdaBN
from AdaBN.Tester import Tester
from AdaBN.Trainer import Trainer
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
    epochs: int = 50,
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
    target_dataset = CMAPSSDataset(
        root_dir=root_dir,
        fd=target_fd,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
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
    target_train_dataloader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=False
    )
    target_test_dataloader = DataLoader(
        target_test_subset, batch_size=len(target_test_subset), shuffle=False
    )

    print(
        f"Data loaded: Source ({source_fd}) train={len(source_train_subset)}, "
        f"val={len(source_val_subset)} | Target ({target_fd}) test_engines={len(target_test_subset)}"
    )

    # Initialize AdaBN model: FeatureExtractor + Regressor with BatchNorm
    model = AdaBN(
        input_size=24,
        lstm_hidden_size=128,
        f_size=64,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        regressor_first_hidden_size=32,
        regressor_second_hidden_size=0,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.5)], gamma=0.1)
    score_fn = Score(a_1=13.0, a_2=10.0)
    early_stopper = EarlyStopping(patience=15)
    reporter = Reporter(name="AdaBN.Main", use_wandb=False)

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

    print(f"\n--- Training AdaBN on Source Domain ({source_fd}) for {epochs} epochs ---")
    trainer.train(epochs=epochs)

    tester = Tester(model=model, score_loss=score_fn, device=device, reporter=reporter)

    # 1. Baseline: Evaluate on Target without Domain Adaptation (Source BN statistics)
    print("\n--- Evaluating Target Domain BEFORE Adaptation (Source BN Stats) ---")
    metrics_before = tester.evaluate(target_test_dataloader)
    print(
        f"Before AdaBN:  RMSE = {metrics_before['rmse']:.2f} | "
        f"MAE = {metrics_before['mae']:.2f} | Score = {metrics_before['score']:.2f}"
    )

    # 2. Test-Time Domain Adaptation using AdaBN on Target Test Data
    print("\n--- Adapting BN statistics using Target Test Data (AdaBN Test-Time DA) ---")
    metrics_after_test_adapt = tester.evaluate(
        target_test_dataloader, adapt_dataloader=target_test_dataloader
    )
    print(
        f"After AdaBN (Test set adapted):  RMSE = {metrics_after_test_adapt['rmse']:.2f} | "
        f"MAE = {metrics_after_test_adapt['mae']:.2f} | Score = {metrics_after_test_adapt['score']:.2f}"
    )

    # 3. Test-Time Domain Adaptation using Target Unlabeled Train Data
    print("\n--- Adapting BN statistics using Target Train Data (AdaBN) ---")
    metrics_after_train_adapt = tester.evaluate(
        target_test_dataloader, adapt_dataloader=target_train_dataloader
    )
    print(
        f"After AdaBN (Train set adapted): RMSE = {metrics_after_train_adapt['rmse']:.2f} | "
        f"MAE = {metrics_after_train_adapt['mae']:.2f} | Score = {metrics_after_train_adapt['score']:.2f}"
    )

    print("\n=== Summary Comparison ===")
    print(f"Task: {source_fd} -> {target_fd}")
    print(f"{'Method':<30} | {'RMSE':<8} | {'MAE':<8} | {'Score':<10}")
    print("-" * 65)
    print(f"{'Source Only (No DA)':<30} | {metrics_before['rmse']:<8.2f} | {metrics_before['mae']:<8.2f} | {metrics_before['score']:<10.2f}")
    print(f"{'AdaBN (Test Adapt)':<30} | {metrics_after_test_adapt['rmse']:<8.2f} | {metrics_after_test_adapt['mae']:<8.2f} | {metrics_after_test_adapt['score']:<10.2f}")
    print(f"{'AdaBN (Train Adapt)':<30} | {metrics_after_train_adapt['rmse']:<8.2f} | {metrics_after_train_adapt['mae']:<8.2f} | {metrics_after_train_adapt['score']:<10.2f}")

    return {
        "before": metrics_before,
        "after_test_adapt": metrics_after_test_adapt,
        "after_train_adapt": metrics_after_train_adapt,
    }


if __name__ == "__main__":
    main()
