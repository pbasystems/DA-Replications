from pathlib import Path

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

source_dataset = NCMAPSSDataset(H5_PATH, flight_class=SOURCE_FC)
target_dataset = NCMAPSSDataset(
    H5_PATH, flight_class=TARGET_FC, feature_stats=source_dataset.feature_stats
)

source_dataloader = DataLoader(source_dataset, batch_size=256, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=256, shuffle=True)

model = OPSDANNHard(input_channels=18, num_phases=3).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

rul_loss = RULLoss()
domain_loss = DomainLoss()
score_fn = Score()

reporter = Reporter(name="ops_dann.Trainer", use_wandb=False)

trainer = Trainer(
    model=model,
    source_dataloader=source_dataloader,
    target_dataloader=target_dataloader,
    optimizer=optimizer,
    rul_loss=rul_loss,
    domain_loss=domain_loss,
    tradeoff=1.0,
    device=DEVICE,
    reporter=reporter,
)
trainer.train(epochs=EPOCHS)

tester = Tester(model=model, score_loss=score_fn, device=DEVICE)
metrics = tester.evaluate(target_dataloader)
reporter.log_metrics(metrics)
reporter.finish()
