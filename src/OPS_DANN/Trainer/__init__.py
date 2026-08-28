import math

import torch
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        source_dataloader,
        target_dataloader,
        optimizer,
        rul_loss,
        domain_loss,
        tradeoff: float,
        device,
        num_phases: int = 3,
        reporter=None,
    ):
        self.model = model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.optimizer = optimizer
        self.rul_loss = rul_loss
        self.domain_loss = domain_loss
        self.tradeoff = tradeoff
        self.device = device
        self.num_phases = num_phases
        self.reporter = reporter

    def _phase_averaged_domain_loss(self, domain_pred, domain_true, phase):
        losses = []
        for phase_id in range(self.num_phases):
            mask = phase == phase_id
            if mask.any():
                losses.append(self.domain_loss(domain_pred[mask], domain_true[mask]))
        return (
            torch.stack(losses).mean() if losses else torch.tensor(0.0, device=domain_pred.device)
        )

    def train(self, epochs: int):
        self.model.train()
        steps_per_epoch = min(len(self.source_dataloader), len(self.target_dataloader))
        total_steps = steps_per_epoch * epochs
        base_lr = self.optimizer.param_groups[0]["lr"]
        global_step = 0

        for epoch in range(epochs):
            total_rul_loss, total_domain_loss = 0.0, 0.0

            lp_epoch = epoch / max(epochs - 1, 1)
            new_lr = base_lr / (1 + 10 * lp_epoch) ** 0.75
            for group in self.optimizer.param_groups:
                group["lr"] = new_lr

            bar = tqdm(
                zip(self.source_dataloader, self.target_dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for (source_x, source_y, source_z, _), (target_x, target_y, target_z, _) in bar:
                source_x, source_y, source_z = (
                    source_x.to(self.device),
                    source_y.to(self.device),
                    source_z.to(self.device),
                )
                target_x, target_z = target_x.to(self.device), target_z.to(self.device)
                source_y = source_y.unsqueeze(1)

                lp = global_step / total_steps
                self.model.grl.alpha = 2 / (1 + math.exp(-10 * lp)) - 1

                self.optimizer.zero_grad()

                rul_pred, source_domain_pred = self.model(source_x, source_z)
                _, target_domain_pred = self.model(target_x, target_z)

                rul_loss_value = self.rul_loss(rul_pred, source_y)

                domain_pred = torch.cat([source_domain_pred, target_domain_pred], dim=0)
                domain_true = torch.cat(
                    [
                        torch.zeros(source_domain_pred.size(0), 1, device=self.device),
                        torch.ones(target_domain_pred.size(0), 1, device=self.device),
                    ],
                    dim=0,
                )
                phase = torch.cat([source_z, target_z], dim=0)
                domain_loss_value = self._phase_averaged_domain_loss(
                    domain_pred, domain_true, phase
                )

                loss = rul_loss_value + self.tradeoff * domain_loss_value
                loss.backward()
                self.optimizer.step()

                total_rul_loss += rul_loss_value.item()
                total_domain_loss += domain_loss_value.item()
                global_step += 1
                bar.set_postfix(
                    rul=f"{rul_loss_value.item():.4f}", domain=f"{domain_loss_value.item():.4f}"
                )

            total_rul_loss /= steps_per_epoch
            total_domain_loss /= steps_per_epoch

            if self.reporter is not None:
                self.reporter.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "rul_loss": total_rul_loss,
                        "domain_loss": total_domain_loss,
                        "rho": self.model.grl.alpha,
                        "lr": new_lr,
                    },
                    step=epoch + 1,
                )
