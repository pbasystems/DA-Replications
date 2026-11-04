from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from TTN.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from TTN.EarlyStopper import EarlyStopping
from TTN.Loss import RegressionLoss
from TTN.Loss.Score import Score
from TTN.Model import TTN
from TTN.Tester import Tester
from TTN.Trainer import Trainer
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
    pretrain_epochs: int = 20,
    posttrain_epochs: int = 5,
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

    # Initialize TTN Model (FeatureExtractor + Regressor with TTN1d layers)
    model = TTN(
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
    scheduler = MultiStepLR(optimizer, milestones=[int(pretrain_epochs * 0.5)], gamma=0.1)
    score_fn = Score(a_1=13.0, a_2=10.0)
    early_stopper = EarlyStopping(patience=15)
    reporter = Reporter(name="TTN.Main", use_wandb=False)

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

    # Phase 1: Pre-training on Source Domain (CBN mode)
    print(f"\n--- [Phase 1] Pre-training on Source Domain ({source_fd}) for {pretrain_epochs} epochs ---")
    trainer.train(epochs=pretrain_epochs)

    tester = Tester(model=model, score_loss=score_fn, device=device, reporter=reporter)

    # Evaluate Baseline CBN: Conventional Batch Normalization (alpha=0, source running stats)
    metrics_cbn = tester.evaluate(target_test_dataloader, mode="cbn")
    print(
        f"1. CBN (No DA, alpha=0):        RMSE = {metrics_cbn['rmse']:.2f} | "
        f"MAE = {metrics_cbn['mae']:.2f} | Score = {metrics_cbn['score']:.2f}"
    )

    # Evaluate Baseline TBN: Transductive Batch Normalization (alpha=1, AdaBN test batch stats)
    metrics_tbn = tester.evaluate(target_test_dataloader, mode="tbn")
    print(
        f"2. TBN (AdaBN, alpha=1):        RMSE = {metrics_tbn['rmse']:.2f} | "
        f"MAE = {metrics_tbn['mae']:.2f} | Score = {metrics_tbn['score']:.2f}"
    )

    # Phase 2: Post-training (Algorithms 1 & 2)
    print(f"\n--- [Phase 2] TTN Post-training (Prior A + Alpha Optimization) ---")
    priors = trainer.obtain_prior(source_train_dataloader, max_samples=1024)
    print(f"Obtained Prior A for {len(priors)} TTN layers.")
    for idx, p in enumerate(priors):
        print(f"  Layer {idx + 1} Prior A: mean = {p.mean().item():.4f}, std = {p.std().item():.4f}")

    # Evaluate Prior A directly
    model.set_ttn_mode("ttn")
    for layer, p in zip(model.get_ttn_layers(), priors):
        layer.set_alpha(p)
    metrics_prior = tester.evaluate(target_test_dataloader, mode="ttn")
    print(
        f"3. TTN with Prior A:           RMSE = {metrics_prior['rmse']:.2f} | "
        f"MAE = {metrics_prior['mae']:.2f} | Score = {metrics_prior['score']:.2f}"
    )

    # Optimize Alpha
    print(f"Optimizing Alpha for {posttrain_epochs} epochs...")
    opt_alphas = trainer.post_train(
        source_train_dataloader, epochs=posttrain_epochs, lr=1e-3, lambda_mse=1.0
    )
    for idx, a in enumerate(opt_alphas):
        print(f"  Layer {idx + 1} Optimized Alpha: mean = {a.mean().item():.4f}, std = {a.std().item():.4f}")

    metrics_opt = tester.evaluate(target_test_dataloader, mode="ttn")
    print(
        f"4. TTN (Post-trained alpha):   RMSE = {metrics_opt['rmse']:.2f} | "
        f"MAE = {metrics_opt['mae']:.2f} | Score = {metrics_opt['score']:.2f}"
    )

    # Evaluate constant alpha sweep (as in paper Table 10)
    sweep_results = {}
    for const_a in [0.1, 0.2, 0.3, 0.5, 0.7]:
        metrics_a = tester.evaluate(target_test_dataloader, mode="ttn", alpha=const_a)
        sweep_results[const_a] = metrics_a

    print("\n=== Summary Comparison: TTN on CMAPSS ===")
    print(f"Task: {source_fd} -> {target_fd}")
    print(f"{'Method / Configuration':<35} | {'RMSE':<8} | {'MAE':<8} | {'Score':<10}")
    print("-" * 70)
    print(f"{'CBN (Source Only, alpha=0)':<35} | {metrics_cbn['rmse']:<8.2f} | {metrics_cbn['mae']:<8.2f} | {metrics_cbn['score']:<10.2f}")
    print(f"{'TBN (AdaBN, alpha=1)':<35} | {metrics_tbn['rmse']:<8.2f} | {metrics_tbn['mae']:<8.2f} | {metrics_tbn['score']:<10.2f}")
    print(f"{'TTN (Prior A)':<35} | {metrics_prior['rmse']:<8.2f} | {metrics_prior['mae']:<8.2f} | {metrics_prior['score']:<10.2f}")
    print(f"{'TTN (Post-Trained Optimized alpha)':<35} | {metrics_opt['rmse']:<8.2f} | {metrics_opt['mae']:<8.2f} | {metrics_opt['score']:<10.2f}")
    print("-" * 70)
    for const_a, m in sweep_results.items():
        print(f"{f'TTN (Constant alpha={const_a:.1f})':<35} | {m['rmse']:<8.2f} | {m['mae']:<8.2f} | {m['score']:<10.2f}")

    return {
        "cbn": metrics_cbn,
        "tbn": metrics_tbn,
        "ttn_prior": metrics_prior,
        "ttn_opt": metrics_opt,
        "sweep": sweep_results,
    }


if __name__ == "__main__":
    main()
