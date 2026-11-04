from __future__ import annotations
import copy
import torch
from torch import nn
from tqdm import tqdm

from TTN.EarlyStopper import EarlyStopping
from TTN.Model.TTNLayer import TTN1d
from utils.reporter import Reporter


def _clip_optimizer_grads(optimizer, max_norm=1.0):
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def augment_time_series(x: torch.Tensor, noise_std: float = 0.15, scale_range: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """
    Simulates domain shift on multivariate time series:
    adds Gaussian noise and applies feature-wise scaling jitter.
    """
    scale = torch.empty(x.size(0), 1, x.size(2), device=x.device).uniform_(*scale_range)
    noise = torch.randn_like(x) * noise_std
    return x * scale + noise


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        source_train_dataloader,
        source_val_dataloader,
        optimizer=None,
        scheduler=None,
        regression_optimizer=None,
        regression_scheduler=None,
        regression_loss=None,
        score_loss=None,
        device=None,
        max_grad_norm=1.0,
        early_stopper: EarlyStopping | None = None,
        reporter: Reporter | None = None,
        **kwargs,
    ):
        self.model = model
        self.source_train_dataloader = source_train_dataloader
        self.source_val_dataloader = source_val_dataloader
        self.optimizer = optimizer or regression_optimizer
        self.scheduler = scheduler or regression_scheduler
        self.regression_loss = regression_loss or nn.L1Loss()
        self.score_loss = score_loss
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = max_grad_norm
        self.reporter = reporter
        self.mse_loss = nn.MSELoss()
        self.early_stopper = early_stopper

    def train(self, epochs: int):
        """
        Pre-training on source domain with Conventional Batch Normalization (CBN).
        Accumulates source running statistics mu_s and sigma_s^2.
        """
        if hasattr(self.model, "set_ttn_mode"):
            self.model.set_ttn_mode("cbn")

        for epoch in range(epochs):
            self.model.train()
            total_regression_loss = 0.0

            pbar = tqdm(
                self.source_train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [TTN Pre-train]",
                leave=False,
            )
            for batch in pbar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss_val = self.regression_loss(outputs, labels)
                loss_val.backward()
                _clip_optimizer_grads(self.optimizer, self.max_grad_norm)
                self.optimizer.step()

                total_regression_loss += loss_val.item()
                pbar.set_postfix(loss=f"{loss_val.item():.4f}")

            total_regression_loss /= len(self.source_train_dataloader)
            train_score = (
                self.score_loss(outputs, labels).item() if self.score_loss is not None else 0.0
            )

            validation_loss, val_score = self.validate()

            if self.reporter is not None:
                self.reporter.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train/regression_loss": total_regression_loss,
                        "train/score": train_score,
                        "val/regression_loss": validation_loss,
                        "val/score": val_score,
                    }
                )

            if self.early_stopper is not None and self.early_stopper.step(
                validation_loss, self.model
            ):
                if self.reporter is not None:
                    self.reporter.info(f"Early stopping at epoch {epoch + 1}")
                self.early_stopper.restore_best_weights(self.model)
                break

            if self.scheduler is not None:
                self.scheduler.step()

        if self.early_stopper is not None:
            self.early_stopper.restore_best_weights(self.model)

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        if hasattr(self.model, "set_ttn_mode"):
            self.model.set_ttn_mode("cbn")

        total_sq_error = 0.0
        total_score = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.source_val_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                outputs = self.model(inputs)
                total_sq_error += ((outputs - labels) ** 2).sum().item()
                if self.score_loss is not None:
                    total_score += self.score_loss(outputs, labels).item()
                n_samples += labels.size(0)

        rmse = (total_sq_error / max(n_samples, 1)) ** 0.5
        return rmse, total_score

    def obtain_prior(
        self,
        dataloader=None,
        max_samples: int = 1024,
        noise_std: float = 0.15,
    ) -> list[torch.Tensor]:
        """
        Algorithm 1: Obtain Prior A.
        Computes cosine similarity between gradients of clean vs augmented data on affine parameters (gamma, beta).
        Prior A = (1 - (s_gamma + s_beta)/2)^2.
        """
        loader = dataloader or self.source_train_dataloader
        self.model.eval()
        if hasattr(self.model, "set_ttn_mode"):
            self.model.set_ttn_mode("cbn")

        ttn_layers: list[TTN1d] = (
            self.model.get_ttn_layers()
            if hasattr(self.model, "get_ttn_layers")
            else [m for m in self.model.modules() if isinstance(m, TTN1d)]
        )

        # Freeze all parameters except affine parameters of TTN1d
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in ttn_layers:
            if layer.weight is not None:
                layer.weight.requires_grad = True
            if layer.bias is not None:
                layer.bias.requires_grad = True

        sim_gamma = [torch.zeros(layer.num_features, device=self.device) for layer in ttn_layers]
        sim_beta = [torch.zeros(layer.num_features, device=self.device) for layer in ttn_layers]
        n_batches_seen = 0
        total_samples = 0

        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(1)

            # 1. Gradient from clean inputs
            self.model.zero_grad(set_to_none=True)
            out_clean = self.model(inputs)
            loss_clean = self.regression_loss(out_clean, labels)
            loss_clean.backward()
            grad_gamma_clean = [
                l.weight.grad.clone() if l.weight.grad is not None else torch.zeros_like(l.weight)
                for l in ttn_layers
            ]
            grad_beta_clean = [
                l.bias.grad.clone() if l.bias.grad is not None else torch.zeros_like(l.bias)
                for l in ttn_layers
            ]

            # 2. Gradient from augmented inputs
            aug_inputs = augment_time_series(inputs, noise_std=noise_std)
            self.model.zero_grad(set_to_none=True)
            out_aug = self.model(aug_inputs)
            loss_aug = self.regression_loss(out_aug, labels)
            loss_aug.backward()
            grad_gamma_aug = [
                l.weight.grad.clone() if l.weight.grad is not None else torch.zeros_like(l.weight)
                for l in ttn_layers
            ]
            grad_beta_aug = [
                l.bias.grad.clone() if l.bias.grad is not None else torch.zeros_like(l.bias)
                for l in ttn_layers
            ]

            # Cosine similarity per channel
            for i, layer in enumerate(ttn_layers):
                g_c = grad_gamma_clean[i]
                g_a = grad_gamma_aug[i]
                norm_c = torch.norm(g_c, p=2).clamp(min=1e-8)
                norm_a = torch.norm(g_a, p=2).clamp(min=1e-8)
                sim_gamma[i] += (g_c * g_a) / (norm_c * norm_a)

                b_c = grad_beta_clean[i]
                b_a = grad_beta_aug[i]
                norm_bc = torch.norm(b_c, p=2).clamp(min=1e-8)
                norm_ba = torch.norm(b_a, p=2).clamp(min=1e-8)
                sim_beta[i] += (b_c * b_a) / (norm_bc * norm_ba)

            n_batches_seen += 1
            total_samples += inputs.size(0)
            if total_samples >= max_samples:
                break

        priors = []
        for i, layer in enumerate(ttn_layers):
            s_g = sim_gamma[i] / max(n_batches_seen, 1)
            s_b = sim_beta[i] / max(n_batches_seen, 1)

            # Min-max normalize per layer
            s_g_min, s_g_max = s_g.min(), s_g.max()
            span_g = (s_g_max - s_g_min).clamp(min=1e-8)
            norm_sg = (s_g - s_g_min) / span_g

            s_b_min, s_b_max = s_b.min(), s_b.max()
            span_b = (s_b_max - s_b_min).clamp(min=1e-8)
            norm_sb = (s_b - s_b_min) / span_b

            dist = 1.0 - 0.5 * (norm_sg + norm_sb)
            prior = (dist ** 2).clamp(min=0.0, max=1.0)
            layer.set_prior(prior)
            priors.append(prior)

        return priors

    def post_train(
        self,
        dataloader=None,
        epochs: int = 10,
        lr: float = 1e-3,
        lambda_mse: float = 1.0,
        noise_std: float = 0.15,
        max_prior_samples: int = 1024,
    ) -> list[torch.Tensor]:
        """
        Algorithm 2: Post-train TTN.
        1. Obtains prior A and initializes alpha with A.
        2. Optimizes alpha parameters with frozen model weights on augmented data.
        Loss = L_task + lambda_mse * ||alpha - A||^2.
        """
        loader = dataloader or self.source_train_dataloader

        # Step 1: Obtain Prior A
        priors = self.obtain_prior(loader, max_samples=max_prior_samples, noise_std=noise_std)

        # Step 2: Freeze all model weights except alpha parameters
        for p in self.model.parameters():
            p.requires_grad = False

        ttn_layers: list[TTN1d] = (
            self.model.get_ttn_layers()
            if hasattr(self.model, "get_ttn_layers")
            else [m for m in self.model.modules() if isinstance(m, TTN1d)]
        )

        alpha_params = []
        for layer in ttn_layers:
            layer.alpha.requires_grad = True
            alpha_params.append(layer.alpha)

        if hasattr(self.model, "set_ttn_mode"):
            self.model.set_ttn_mode("ttn")

        optimizer = torch.optim.Adam(alpha_params, lr=lr)

        pbar = tqdm(range(epochs), desc="TTN Post-train (alpha optimization)", leave=False)
        for _ in pbar:
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                aug_inputs = augment_time_series(inputs, noise_std=noise_std)
                optimizer.zero_grad()
                outputs = self.model(aug_inputs)
                task_loss = self.regression_loss(outputs, labels)

                # MSE constraint with Prior A
                mse_prior = sum(
                    torch.mean((layer.alpha - layer.prior_A) ** 2) for layer in ttn_layers
                )
                total_loss = task_loss + lambda_mse * mse_prior

                total_loss.backward()
                optimizer.step()

                # Clamp alpha to [0, 1]
                with torch.no_grad():
                    for layer in ttn_layers:
                        layer.alpha.data.clamp_(0.0, 1.0)

                pbar.set_postfix(
                    task_loss=f"{task_loss.item():.4f}", mse=f"{mse_prior.item():.4f}"
                )

        return [layer.alpha.data.clone() for layer in ttn_layers]
