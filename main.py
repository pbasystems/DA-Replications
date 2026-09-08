from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

import wandb
from lstm_dann.Dataset import CMAPSSDataset
from lstm_dann.Loss import ClassificationLoss, RegressionLoss
from lstm_dann.Loss.Score import Score
from lstm_dann.Model import LSTM_DANN
from lstm_dann.Tester import Tester
from lstm_dann.Trainer import Trainer
from utils.reporter import Reporter

load_dotenv()
wandb.login()

l2_reg = 0.01
lr_source_reg = 0.01
lr_domain_class = 0.01

reporter = Reporter(
    name="LSTM_DANN.Trainer",
    use_wandb=True,
    wandb_project="DA-Replications|LSTM-DANN|CMAPSS",
    wandb_run_name="FD001-to-FD002",
    wandb_config={
        "source_fd": "FD001",
        "target_fd": "FD002",
        "hidden_size": 64,
        "f_size": 32,
        "num_layers": 1,
        "lstm_dropout": 0.5,
        "regressor_dropout": 0.3,
        "classifier_dropout": 0.3,
        "alpha": 0.8,
        "lr_source_reg": lr_source_reg,
        "lr_domain_class": lr_domain_class,
        "l2_reg": l2_reg,
    },
)


def last_window_per_engine(dataset):
    unit_numbers = dataset.unit_numbers
    _, last_pos_reversed = np.unique(unit_numbers[::-1], return_index=True)
    last_indices = len(unit_numbers) - 1 - last_pos_reversed
    return Subset(dataset, sorted(last_indices))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path.cwd()
DATASET_PATH = ROOT_DIR / "Data" / "CMAPSS"
source_dataset = CMAPSSDataset(
    root_dir=DATASET_PATH, fd="FD001", split="train", window_size=30, r_early=125
)
target_dataset = CMAPSSDataset(
    root_dir=DATASET_PATH, fd="FD002", split="train", window_size=30, r_early=125
)

target_test_dataset = CMAPSSDataset(
    root_dir=DATASET_PATH,
    fd="FD002",
    split="test",
    window_size=30,
    r_early=125,
    feature_stats=target_dataset.feature_stats,
)

target_test_subset = last_window_per_engine(target_test_dataset)
target_test_dataloader = DataLoader(
    target_test_subset, batch_size=len(target_test_subset), shuffle=False
)


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
)

model = model.to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0001,
)
regression_loss = RegressionLoss(p=1)
classification_loss = ClassificationLoss()
score_fn = Score(a_1=13, a_2=10)


regression_optimizer = optim.SGD(
    [
        {"params": model.feature_extractor.parameters(), "lr": lr_source_reg},
        {
            "params": model.regressor.parameters(),
            "lr": lr_source_reg,
            "weight_decay": l2_reg,
        },
    ]
)

domain_optimizer = optim.SGD(
    [
        {"params": model.feature_extractor.parameters(), "lr": lr_domain_class},
        {
            "params": model.classifier.parameters(),
            "lr": lr_domain_class,
            "weight_decay": l2_reg,
        },
    ]
)

regression_scheduler = MultiStepLR(regression_optimizer, milestones=[100], gamma=0.1)
domain_scheduler = MultiStepLR(domain_optimizer, milestones=[100], gamma=0.1)

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
    device=device,
    reporter=reporter,
)
trainer.train(epochs=200)
tester = Tester(model=model, score_loss=score_fn, device=device)
metrics = tester.evaluate(target_test_dataloader)
reporter.log_metrics(metrics)
reporter.finish()
