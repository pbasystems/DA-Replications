import torch


class Tester:
    def __init__(self, model, score_loss, device):
        self.model = model
        self.score_loss = score_loss
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        total_sq_error = 0.0
        total_abs_error = 0.0
        total_score = 0.0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(
                1
            )  # (batch,) -> (batch, 1) to match regressor output

            regression_outputs, _ = self.model(inputs)

            total_sq_error += ((regression_outputs - labels) ** 2).sum().item()
            total_abs_error += (regression_outputs - labels).abs().sum().item()
            total_score += self.score_loss(regression_outputs, labels).item()
            n_samples += labels.size(0)

        self.model.train()

        return {
            "rmse": (total_sq_error / n_samples) ** 0.5,
            "mae": total_abs_error / n_samples,
            "score": total_score,
            "n_samples": n_samples,
        }
