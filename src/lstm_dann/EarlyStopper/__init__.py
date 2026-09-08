import copy


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.epochs_no_improve = 0
        self.best_state_dict = None
        self.early_stop = False

    def step(self, current_val_rmse: float, model) -> bool:
        if current_val_rmse < self.best_score - self.min_delta:
            self.best_score = current_val_rmse
            self.epochs_no_improve = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
