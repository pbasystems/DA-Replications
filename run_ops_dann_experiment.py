from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ops_dann.Dataset import NCMAPSSDataset
from ops_dann.Loss import DomainLoss, RULLoss
from ops_dann.Loss.Score import Score
from ops_dann.Model import OPSDANNHard
from ops_dann.Tester import Tester
from ops_dann.Trainer import Trainer
from utils.reporter import Reporter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = Path.cwd() / "Data" / "NCMAPSS" / "N-CMAPSS_DS03-012.h5"

SOURCE_FC, TARGET_FC = 1, 3
EPOCHS = 15
N_TRIALS = 10


def run_trial(seed, source_dataset, target_dataset):
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_dataloader = DataLoader(source_dataset, batch_size=256, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=256, shuffle=True)

    model = OPSDANNHard(input_channels=18, num_phases=3).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

    score_fn = Score()
    reporter = Reporter(name=f"trial-{seed}", use_wandb=False)

    trainer = Trainer(
        model=model,
        source_dataloader=source_dataloader,
        target_dataloader=target_dataloader,
        optimizer=optimizer,
        rul_loss=RULLoss(),
        domain_loss=DomainLoss(),
        tradeoff=1.0,
        device=DEVICE,
        reporter=reporter,
    )
    trainer.train(epochs=EPOCHS)

    tester = Tester(model=model, score_loss=score_fn, device=DEVICE)
    return tester.evaluate(target_dataloader)


def main():
    source_dataset = NCMAPSSDataset(H5_PATH, flight_class=SOURCE_FC)
    target_dataset = NCMAPSSDataset(
        H5_PATH, flight_class=TARGET_FC, feature_stats=source_dataset.feature_stats
    )

    results = []
    for trial in range(N_TRIALS):
        metrics = run_trial(trial, source_dataset, target_dataset)
        score_mean = metrics["score"] / metrics["n_samples"]
        print(
            f"Trial {trial + 1}/{N_TRIALS}: RMSE={metrics['rmse_cycles']:.2f}  s-score={score_mean:.2f}"
        )
        results.append({"rmse": metrics["rmse_cycles"], "score": score_mean})

    rmses = np.array([r["rmse"] for r in results])
    scores = np.array([r["score"] for r in results])

    print(f"\nFC{SOURCE_FC} -> FC{TARGET_FC} over {N_TRIALS} trials:")
    print(f"RMSE:    {rmses.mean():.2f} +/- {rmses.std():.2f}")
    print(f"s-score: {scores.mean():.2f} +/- {scores.std():.2f}")


if __name__ == "__main__":
    main()
