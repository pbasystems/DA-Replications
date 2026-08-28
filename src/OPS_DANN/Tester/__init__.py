import torch


class Tester:
    def __init__(self, model, score_loss, device, num_phases: int = 3):
        self.model = model
        self.score_loss = score_loss
        self.device = device
        self.num_phases = num_phases

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        total_sq_error, total_abs_error, total_score, n_samples = 0.0, 0.0, 0.0, 0
        total_sq_error_norm = 0.0

        for x, y, z, scale in dataloader:
            x, y, z, scale = (
                x.to(self.device),
                y.to(self.device),
                z.to(self.device),
                scale.to(self.device),
            )
            y = y.unsqueeze(1)
            scale = scale.unsqueeze(1)

            rul_pred, _ = self.model(x, z)

            rul_pred_cycles = rul_pred * scale
            y_cycles = y * scale

            total_sq_error += ((rul_pred_cycles - y_cycles) ** 2).sum().item()
            total_abs_error += (rul_pred_cycles - y_cycles).abs().sum().item()
            total_score += self.score_loss(rul_pred_cycles, y_cycles).item()
            total_sq_error_norm += ((rul_pred - y) ** 2).sum().item()
            n_samples += y.size(0)

        self.model.train()
        return {
            "rmse_cycles": (total_sq_error / n_samples) ** 0.5,
            "mae_cycles": total_abs_error / n_samples,
            "score": total_score,
            "score_mean": total_score / n_samples,
            "rmse_normalized": (total_sq_error_norm / n_samples) ** 0.5,
            "n_samples": n_samples,
        }
