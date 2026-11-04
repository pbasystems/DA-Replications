# What is AdaBN

Adaptive Batch Normalization (AdaBN) is a parameter-free domain adaptation method introduced in the paper:
**"Revisiting Batch Normalization For Practical Domain Adaptation"** by Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, Xiaodi Hou (arXiv:1603.04779).

The key hypothesis of AdaBN is:
- **Domain-invariant representation / task knowledge** is stored in the weight matrices ($W, b$) of each layer.
- **Domain-specific information** is captured primarily by the statistics ($\mu, \sigma^2$) of the Batch Normalization layers.

Unlike domain-adversarial methods (such as DANN) that require a domain classifier and Gradient Reversal Layer (GRL), AdaBN:
1. Does not require any domain classifier or discriminator.
2. Does not require any adversarial training, MMD loss, or optimization on the target domain.
3. Operates at **test time**: the model is trained purely on the labeled source domain. At test time, target domain data is forwarded through the network with learnable parameters frozen to re-estimate the batch normalization mean $\mu_T$ and variance $\sigma_T^2$ for all BN layers.
4. Evaluation on target domain samples is then conducted using these target-specific BN statistics.

# Architecture in this Implementation

The model in `src/AdaBN` consists strictly of:
- **FeatureExtractor**:
  - `nn.LSTM` (input sequence: $T_w=30$, $d=24$)
  - `nn.BatchNorm1d(lstm_hidden_size)`
  - `nn.ReLU()`
  - `nn.Dropout(p=lstm_dropout)`
  - `nn.Linear(lstm_hidden_size, f_size)`
  - `nn.BatchNorm1d(f_size)`
- **Regressor**:
  - `nn.Linear(f_size, regressor_first_hidden_size)`
  - `nn.BatchNorm1d(regressor_first_hidden_size)`
  - `nn.ReLU()`
  - `nn.Dropout(p=regressor_dropout)`
  - (optional 2nd hidden layer with `BatchNorm1d`)
  - `nn.Linear(regressor_first_hidden_size, 1)`

# Test-Time Domain Adaptation Procedure (Algorithm 1)

1. **Source Training**: Train the FeatureExtractor + Regressor network on source domain data using `RegressionLoss` ($L_1$ loss) and early stopping on source validation RMSE.
2. **Domain Adaptation**:
   - Freeze all network weights (`torch.no_grad()`).
   - Reset running statistics of all `BatchNorm1d` layers and set them to calculate target statistics over target domain data (`target_test_dataloader` or `target_train_dataloader`).
   - Forward pass target data through the model to establish $\mu_T$ and $\sigma_T^2$.
   - Switch model back to evaluation mode (`model.eval()`).
3. **Target Evaluation**: Compute RMSE, MAE, and NASA CMAPSS score on target engines using the adapted BN statistics.

# Comparison with LSTM-DANN
| Aspect | LSTM-DANN | AdaBN |
|---|---|---|
| Domain Classifier | Yes (MLP) | None |
| Gradient Reversal (GRL) | Yes ($\alpha$) | None |
| Training Complexity | 2 optimizers (regression + domain) | 1 optimizer (source regression) |
| Target Data at Training | Required during training batches | Not needed during training |
| Target Adaptation | Joint adversarial representation | Parameter-free test-time BN statistic modulation |
