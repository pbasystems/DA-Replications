import torch
from tqdm import tqdm



def _clip_optimizer_grads(optimizer, max_norm=1.0):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, max_norm)


class Trainer:
    def __init__(self, model, source_dataloader, target_dataloader, regression_optimizer, domain_optimizer,
                 regression_scheduler, domain_scheduler, regression_loss, classification_loss, score_loss, device,
                 max_grad_norm=1.0,reporter=None):
        self.model = model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.regression_optimizer = regression_optimizer
        self.domain_optimizer = domain_optimizer
        self.regression_scheduler = regression_scheduler
        self.domain_scheduler = domain_scheduler
        self.regression_loss = regression_loss
        self.classification_loss = classification_loss
        self.score_loss = score_loss
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.reporter = reporter

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            total_regression_loss = 0.0
            total_classification_loss = 0.0

            regression_bar = tqdm(
                self.source_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [regression]",
                leave=False,
            )
            for batch in regression_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                self.regression_optimizer.zero_grad()
                regression_outputs, _ = self.model(inputs)
                regression_loss_value = self.regression_loss(regression_outputs, labels)
                regression_loss_value.backward()
                _clip_optimizer_grads(self.regression_optimizer, self.max_grad_norm)
                self.regression_optimizer.step()

                total_regression_loss += regression_loss_value.item()
                regression_bar.set_postfix(loss=f"{regression_loss_value.item():.4f}")

            total_regression_loss /= len(self.source_dataloader)

            n_batches = 0
            classification_bar = tqdm(
                zip(self.source_dataloader, self.target_dataloader),
                total=min(len(self.source_dataloader), len(self.target_dataloader)),
                desc=f"Epoch {epoch + 1}/{epochs} [domain]",
                leave=False,
            )
            for source_batch, target_batch in classification_bar:
                source_inputs, _ = source_batch
                target_inputs, _ = target_batch
                source_inputs, target_inputs = source_inputs.to(self.device), target_inputs.to(self.device)

                self.domain_optimizer.zero_grad()
                _, source_classification_outputs = self.model(source_inputs)
                _, target_classification_outputs = self.model(target_inputs)

                source_labels = torch.zeros(source_classification_outputs.size(0), 1).to(self.device)
                target_labels = torch.ones(target_classification_outputs.size(0), 1).to(self.device)

                source_classification_loss = self.classification_loss(source_classification_outputs, source_labels)
                target_classification_loss = self.classification_loss(target_classification_outputs, target_labels)
                classification_loss_value = source_classification_loss + target_classification_loss

                classification_loss_value.backward()
                _clip_optimizer_grads(self.domain_optimizer, self.max_grad_norm)
                self.domain_optimizer.step()

                total_classification_loss += classification_loss_value.item()
                n_batches += 1
                classification_bar.set_postfix(loss=f"{classification_loss_value.item():.4f}")

            total_classification_loss /= n_batches

            if self.regression_scheduler is not None:
                self.regression_scheduler.step()
            if self.domain_scheduler is not None:
                self.domain_scheduler.step()

            self.reporter.log_metrics({
                "epoch": epoch + 1,
                "regression_loss": total_regression_loss,
                "classification_loss": total_classification_loss,
            })
 
