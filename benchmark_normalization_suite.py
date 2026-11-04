import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from AdaBN.Dataset import COLUMN_NAMES, FEATURE_COLS, CMAPSSDataset, FeatureStats
from AdaBN.EarlyStopper import EarlyStopping
from AdaBN.Loss import RegressionLoss
from AdaBN.Loss.Score import Score
from AdaBN.Model import AdaBN
from AlphaBN.Model.AlphaBNLayer import convert_bn_to_alpha_bn
from MixNorm.Model.MixNormLayer import convert_bn_to_mixnorm
from TTN.Model.TTNLayer import convert_bn_to_ttn
from TTN.Trainer import Trainer as TTNTrainer
from utils.seed import seed_everything


def compute_dataset_feature_stats(root_dir: Path, fd: str) -> FeatureStats:
    train_df = pd.read_csv(root_dir / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(root_dir / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES)
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


def build_base_model(
    input_size: int = 24,
    lstm_hidden_size: int = 128,
    f_size: int = 64,
    lstm_dropout: float = 0.5,
    regressor_dropout: float = 0.3,
    regressor_first_hidden_size: int = 32,
) -> AdaBN:
    return AdaBN(
        input_size=input_size,
        lstm_hidden_size=lstm_hidden_size,
        f_size=f_size,
        num_layers=1,
        lstm_dropout=lstm_dropout,
        regressor_dropout=regressor_dropout,
        regressor_first_hidden_size=regressor_first_hidden_size,
        regressor_second_hidden_size=0,
    )


def train_model_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    domain_name: str = "",
    max_epochs: int = 100,
    patience: int = 15,
    lr: float = 0.01,
    weight_decay: float = 0.01,
) -> nn.Module:
    model = model.to(device)
    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-4)
    criterion = RegressionLoss(p=1)
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-3)

    print(f"--> Training on {domain_name} (Max Epochs: {max_epochs}, Early Stopping Patience: {patience})...")

    for epoch in range(1, max_epochs + 1):
        # 1. Training
        model.train()
        for m in model.modules():
            if isinstance(m, nn.RNNBase):
                m.flatten_parameters()

        train_loss = 0.0
        train_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_samples += len(y)

        train_loss /= max(train_samples, 1)

        # 2. Validation
        model.eval()
        val_sq_err = 0.0
        val_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                pred = model(x)
                val_sq_err += ((pred - y) ** 2).sum().item()
                val_samples += len(y)

        val_rmse = (val_sq_err / max(val_samples, 1)) ** 0.5
        scheduler.step(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:6.2f} | "
                f"Val RMSE: {val_rmse:6.2f} (Best: {early_stopper.best_score:6.2f}) | "
                f"LR: {current_lr:.5f}"
            )

        # Check Early Stopping
        if early_stopper.step(val_rmse, model):
            print(f"    [Early Stopping Triggered] at Epoch {epoch}. Best Val RMSE: {early_stopper.best_score:.2f}")
            break

    early_stopper.restore_best_weights(model)
    model = model.to(device)
    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()
    print(f"    Restored best model weights with Val RMSE = {early_stopper.best_score:.2f}\n")
    return model


@torch.no_grad()
def evaluate_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, score_fn: Score) -> dict[str, float]:
    model = model.to(device)
    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_score = 0.0
    n_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        outputs = model(inputs)

        total_sq_error += ((outputs - labels) ** 2).sum().item()
        total_abs_error += (outputs - labels).abs().sum().item()
        if score_fn is not None:
            total_score += score_fn(outputs, labels).item()
        n_samples += labels.size(0)

    rmse = (total_sq_error / max(n_samples, 1)) ** 0.5
    mae = total_abs_error / max(n_samples, 1)
    return {"rmse": rmse, "mae": mae, "score": total_score}


def evaluate_cbn(base_model: nn.Module, test_loader: DataLoader, device: torch.device, score_fn: Score) -> dict[str, float]:
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    return evaluate_predictions(model, test_loader, device, score_fn)


def evaluate_adabn(base_model: nn.Module, test_loader: DataLoader, device: torch.device, score_fn: Score) -> dict[str, float]:
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()  # Calculate batch statistics on target batch
    return evaluate_predictions(model, test_loader, device, score_fn)


def evaluate_alpha_bn(base_model: nn.Module, test_loader: DataLoader, device: torch.device, score_fn: Score, alpha: float) -> dict[str, float]:
    model = copy.deepcopy(base_model)
    convert_bn_to_alpha_bn(model, default_alpha=alpha)
    model = model.to(device)
    model.eval()
    for m in model.modules():
        if hasattr(m, "set_alpha"):
            m.set_alpha(alpha)
    return evaluate_predictions(model, test_loader, device, score_fn)


def evaluate_ttn(base_model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, score_fn: Score) -> dict[str, float]:
    model = copy.deepcopy(base_model)
    convert_bn_to_ttn(model)
    model = model.to(device)
    model.eval()

    # Calculate Prior A via gradient similarity
    trainer = TTNTrainer(
        model=model,
        source_train_dataloader=train_loader,
        source_val_dataloader=train_loader,
        regression_loss=RegressionLoss(p=1),
        score_loss=score_fn,
        device=device,
    )
    priors = trainer.obtain_prior(train_loader, max_samples=1024)

    # Set prior alpha on all TTN layers
    ttn_layers = [m for m in model.modules() if hasattr(m, "set_alpha")]
    for layer, p in zip(ttn_layers, priors):
        layer.set_alpha(p)
        if hasattr(layer, "set_ttn_mode"):
            layer.set_ttn_mode("ttn")

    return evaluate_predictions(model, test_loader, device, score_fn)


def evaluate_mixnorm(base_model: nn.Module, test_loader: DataLoader, device: torch.device, score_fn: Score, mixing_scale: float) -> dict[str, float]:
    model = copy.deepcopy(base_model)
    convert_bn_to_mixnorm(model, mixing_scale=mixing_scale)
    model = model.to(device)
    model.eval()
    for m in model.modules():
        if hasattr(m, "reset_online_stats"):
            m.reset_online_stats()
        if hasattr(m, "set_online_update"):
            m.set_online_update(True)
        if hasattr(m, "set_mixing_scale"):
            m.set_mixing_scale(mixing_scale)

    return evaluate_predictions(model, test_loader, device, score_fn)


def main():
    parser = argparse.ArgumentParser(description="Test-time Normalization Adaptation Benchmark with Early Stopping")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs per domain before early stopping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without validation improvement)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for SGD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--data_dir", type=str, default="Data/CMAPSS", help="Path to CMAPSS directory")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    tasks = [
        ("FD002", "FD001", "FD002 -> FD001 (Multi-condition -> Single-condition)"),
        ("FD004", "FD003", "FD004 -> FD003 (Multi-condition -> Single-condition)"),
        ("FD001", "FD002", "FD001 -> FD002 (Single-condition -> Multi-condition)"),
        ("FD003", "FD004", "FD003 -> FD004 (Single-condition -> Multi-condition)"),
    ]

    print("==================================================================")
    print("   Test-Time Normalization Benchmark on C-MAPSS Turbofan RUL      ")
    print("==================================================================")
    print(f"Device: {device} | Max Epochs: {args.max_epochs} | Patience: {args.patience} | Batch Size: {args.batch_size} | Seed: {args.seed}\n")

    # 1. Precompute stats & build DataLoaders for each domain
    all_domains = sorted(list({t[0] for t in tasks} | {t[1] for t in tasks}))
    print(f"Domains needed across tasks: {all_domains}\n")

    score_fn = Score()
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    print("--- Loading & Preprocessing C-MAPSS Datasets ---")
    for fd in all_domains:
        stats = compute_dataset_feature_stats(data_dir, fd)
        train_ds = CMAPSSDataset(
            root_dir=data_dir,
            fd=fd,
            split="train",
            window_size=30,
            max_rul=125,
            feature_stats=stats,
        )
        test_ds = CMAPSSDataset(
            root_dir=data_dir,
            fd=fd,
            split="test",
            window_size=30,
            max_rul=125,
            feature_stats=stats,
        )

        train_sub, val_sub = split_by_engine(train_ds, val_ratio=0.10, seed=args.seed)
        test_sub = last_window_per_engine(test_ds)

        train_loaders[fd] = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
        val_loaders[fd] = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        test_loaders[fd] = DataLoader(test_sub, batch_size=len(test_sub), shuffle=False)

        print(f"Domain {fd}: Train={len(train_sub)} windows | Val={len(val_sub)} windows | Test={len(test_sub)} engines")

    # 2. Train Base Models for each unique domain with Early Stopping (acts as source models & oracle models)
    trained_models = {}
    print("\n--- Training Base Models with Early Stopping (Source & Oracle) ---")
    for fd in all_domains:
        base_model = build_base_model()
        trained_model = train_model_with_early_stopping(
            model=base_model,
            train_loader=train_loaders[fd],
            val_loader=val_loaders[fd],
            device=device,
            domain_name=fd,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        trained_models[fd] = trained_model

    # 3. Evaluate All 4 Transfer Tasks
    records = []
    print("\n==================================================================")
    print("                    EVALUATION ACROSS TASKS                       ")
    print("==================================================================")

    for src_fd, tgt_fd, task_desc in tasks:
        print(f"\n>>> Task: {task_desc}")
        src_model = trained_models[src_fd]
        oracle_model = trained_models[tgt_fd]
        tgt_test_loader = test_loaders[tgt_fd]
        src_train_loader = train_loaders[src_fd]

        # 1. Source Only (CBN)
        res_cbn = evaluate_cbn(src_model, tgt_test_loader, device, score_fn)
        # 2. AdaBN / Pure T-BN
        res_adabn = evaluate_adabn(src_model, tgt_test_loader, device, score_fn)
        # 3. Alpha-BN (alpha = 0.50)
        res_abn_50 = evaluate_alpha_bn(src_model, tgt_test_loader, device, score_fn, alpha=0.50)
        # 4. Alpha-BN (alpha = 0.20)
        res_abn_20 = evaluate_alpha_bn(src_model, tgt_test_loader, device, score_fn, alpha=0.20)
        # 5. TTN (Prior A)
        res_ttn = evaluate_ttn(src_model, src_train_loader, tgt_test_loader, device, score_fn)
        # 6. Mix-Norm (m = 0.10)
        res_mn_10 = evaluate_mixnorm(src_model, tgt_test_loader, device, score_fn, mixing_scale=0.10)
        # 7. Mix-Norm (m = 0.50)
        res_mn_50 = evaluate_mixnorm(src_model, tgt_test_loader, device, score_fn, mixing_scale=0.50)
        # 8. Target Oracle Upper Bound (Trained on target train, tested on target test)
        res_oracle = evaluate_cbn(oracle_model, tgt_test_loader, device, score_fn)

        results_map = {
            "Source Only (CBN)": res_cbn,
            "AdaBN / Pure T-BN": res_adabn,
            "Alpha-BN (a=0.50)": res_abn_50,
            "Alpha-BN (a=0.20)": res_abn_20,
            "TTN (Prior A)": res_ttn,
            "Mix-Norm (m=0.10)": res_mn_10,
            "Mix-Norm (m=0.50)": res_mn_50,
            "Target Oracle": res_oracle,
        }

        # Print quick summary for current task
        task_rows = []
        for method_name, m_res in results_map.items():
            task_rows.append([method_name, f"{m_res['rmse']:.2f}", f"{m_res['mae']:.2f}", f"{m_res['score']:.1f}"])
            records.append({
                "Task": task_desc.split(" (")[0],
                "Method": method_name,
                "RMSE": m_res["rmse"],
                "MAE": m_res["mae"],
                "Score": m_res["score"],
            })

        print(tabulate(task_rows, headers=["Method", "RMSE", "MAE", "Score"], tablefmt="simple_outline"))

    # 4. Overall Benchmark Summary
    df_results = pd.DataFrame(records)
    print("\n==================================================================")
    print("                    FINAL BENCHMARK SUMMARY                       ")
    print("==================================================================")

    avg_df = df_results.groupby("Method")[["RMSE", "MAE", "Score"]].mean().reset_index()
    # Sort with Oracle at the end, others by RMSE ascending
    avg_df["sort_key"] = avg_df["Method"].apply(lambda m: 999 if m == "Target Oracle" else 0)
    avg_df = avg_df.sort_values(by=["sort_key", "RMSE"]).drop(columns=["sort_key"])

    print("\n--- Average Across All 4 Benchmark Tasks ---")
    print(tabulate(avg_df.values, headers=avg_df.columns, floatfmt=".2f", tablefmt="github"))

    out_csv = Path("benchmark_normalization_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved detailed results to {out_csv.resolve()}\n")


if __name__ == "__main__":
    main()
