import os
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
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

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = Path.cwd() / "Data" / "CMAPSS"
TEMP_DIR = Path.cwd() / "temp"
os.environ["BASE_WORKING_DIR"] = str(Path.cwd())
N_TRIALS = 10
EPOCHS = 200


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
    seed: int,
    source_train_dataset,
    source_val_dataset,
    target_test_dataloader,
    config: DictConfig,
    pretrain_epochs: int = 20,
    posttrain_epochs: int = 5,
):
    seed_everything(seed)
    cfg_model = getattr(config, "TTN", getattr(config, "AdaBN", getattr(config, "lstm_dann", None)))

    source_train_dataloader = DataLoader(
        source_train_dataset, batch_size=cfg_model.batch_size, shuffle=True
    )
    source_val_dataloader = DataLoader(
        source_val_dataset, batch_size=cfg_model.batch_size, shuffle=False
    )

    model = TTN(
        input_size=cfg_model.Model.input_size,
        lstm_hidden_size=cfg_model.Model.lstm_hidden_size,
        f_size=cfg_model.Model.f_size,
        num_layers=cfg_model.Model.num_layers,
        lstm_dropout=cfg_model.Model.lstm_dropout,
        regressor_dropout=cfg_model.Model.regressor_dropout,
        regressor_first_hidden_size=cfg_model.Model.regressor_first_hidden_size,
        regressor_second_hidden_size=cfg_model.Model.regressor_second_hidden_size,
    ).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg_model.optimizer.lr_source_reg,
        weight_decay=cfg_model.regularization,
    )
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    score_fn = Score(a_1=13.0, a_2=10.0)

    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    reporter = Reporter(
        name=f"TTN.Experiments.{seed + 1}",
        use_wandb=use_wandb,
        wandb_project="DA-Replications|TTN|CMAPSS",
        wandb_run_name=f"{cfg_model.source_fd}-to-{cfg_model.target_fd} | Trial {seed + 1}",
        wandb_config=cast("dict[str, Any]", OmegaConf.to_container(cfg_model)),
    )

    early_stopper = EarlyStopping(patience=20)

    trainer = Trainer(
        model=model,
        source_train_dataloader=source_train_dataloader,
        source_val_dataloader=source_val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        regression_loss=RegressionLoss(p=1),
        score_loss=score_fn,
        device=DEVICE,
        early_stopper=early_stopper,
        reporter=reporter,
    )

    # 1. Pre-training on source
    trainer.train(epochs=pretrain_epochs)

    # 2. Post-training (Prior A + Optimization)
    trainer.post_train(source_train_dataloader, epochs=posttrain_epochs, lr=1e-3, lambda_mse=1.0)

    tester = Tester(model=model, score_loss=score_fn, device=DEVICE, reporter=reporter)

    metrics_cbn = tester.evaluate(target_test_dataloader, mode="cbn")
    metrics_tbn = tester.evaluate(target_test_dataloader, mode="tbn")
    metrics_ttn = tester.evaluate(target_test_dataloader, mode="ttn")

    reporter.finish()
    return {
        "cbn": metrics_cbn,
        "tbn": metrics_tbn,
        "ttn": metrics_ttn,
    }


@hydra.main(config_path="configs", config_name="config_ttn", version_base="1.1")
def main(cfg: DictConfig):
    cfg_model = getattr(cfg, "TTN", getattr(cfg, "AdaBN", getattr(cfg, "lstm_dann", None)))
    source_fd = cfg_model.source_fd
    target_fd = cfg_model.target_fd

    source_stats = compute_dataset_feature_stats(DATASET_PATH, source_fd)
    target_stats = compute_dataset_feature_stats(DATASET_PATH, target_fd)

    source_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=source_fd,
        split="train",
        window_size=30,
        r_early=125,
        feature_stats=source_stats,
    )
    target_test_dataset = CMAPSSDataset(
        root_dir=DATASET_PATH,
        fd=target_fd,
        split="test",
        window_size=30,
        r_early=125,
        feature_stats=target_stats,
    )

    source_train_subset, source_val_subset = split_by_engine(
        source_dataset, val_ratio=0.10, seed=cfg.seed
    )
    target_test_subset = last_window_per_engine(target_test_dataset)
    target_test_dataloader = DataLoader(
        target_test_subset, batch_size=len(target_test_subset), shuffle=False
    )

    n_trials = getattr(cfg, "n_trials", N_TRIALS)
    pretrain_epochs = getattr(cfg, "pretrain_epochs", 20)
    posttrain_epochs = getattr(cfg, "posttrain_epochs", 5)

    print(f"\n================ TTN Experiment: {source_fd} -> {target_fd} ================")
    print(f"Trials: {n_trials} | Device: {DEVICE}")

    results_cbn, results_tbn, results_ttn = [], [], []

    for trial in range(n_trials):
        metrics = run_trial(
            seed=trial,
            source_train_dataset=source_train_subset,
            source_val_dataset=source_val_subset,
            target_test_dataloader=target_test_dataloader,
            config=cfg,
            pretrain_epochs=pretrain_epochs,
            posttrain_epochs=posttrain_epochs,
        )
        cbn, tbn, ttn = metrics["cbn"], metrics["tbn"], metrics["ttn"]
        print(
            f"Trial {trial + 1}/{n_trials}: "
            f"CBN RMSE={cbn['rmse']:.2f} | TBN RMSE={tbn['rmse']:.2f} | TTN RMSE={ttn['rmse']:.2f}"
        )
        results_cbn.append(cbn)
        results_tbn.append(tbn)
        results_ttn.append(ttn)

    def summarize(res_list, name):
        rmses = np.array([r["rmse"] for r in res_list])
        maes = np.array([r["mae"] for r in res_list])
        scores = np.array([r["score"] for r in res_list])
        print(f"\n--- {name} ({source_fd} -> {target_fd} over {n_trials} trials) ---")
        print(f"RMSE:  {rmses.mean():.2f} +/- {rmses.std():.2f}")
        print(f"MAE:   {maes.mean():.2f} +/- {maes.std():.2f}")
        print(f"Score: {scores.mean():.2f} +/- {scores.std():.2f}")
        return rmses, maes, scores

    cbn_rmse, _, cbn_score = summarize(results_cbn, "CBN (Conventional BN, No DA)")
    tbn_rmse, _, tbn_score = summarize(results_tbn, "TBN (Transductive BN / AdaBN)")
    ttn_rmse, _, ttn_score = summarize(results_ttn, "TTN (Test-Time Normalization)")

    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(TEMP_DIR / "ttn_results.csv", "a") as f:
        f.write(
            f"{source_fd},{target_fd},"
            f"{cbn_rmse.mean():.2f},{cbn_rmse.std():.2f},{cbn_score.mean():.2f},"
            f"{tbn_rmse.mean():.2f},{tbn_rmse.std():.2f},{tbn_score.mean():.2f},"
            f"{ttn_rmse.mean():.2f},{ttn_rmse.std():.2f},{ttn_score.mean():.2f}\n"
        )
    print("\nResults appended to temp/ttn_results.csv")


if __name__ == "__main__":
    main()
