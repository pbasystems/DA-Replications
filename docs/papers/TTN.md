# What is TTN (Test-Time Normalization)

**Test-Time Normalization (TTN)** is a domain-shift aware batch normalization strategy for test-time adaptation introduced in:
**"TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation"** by Hyesu Lim, Byeonggeun Kim, Jaegul Choo, Sungha Choi (ICLR 2023, arXiv:2302.05155).

## Core Motivation: The Trade-off Between CBN and TBN
- **Conventional Batch Normalization (CBN)**: Normalizes test features using running statistics $(\mu_s, \sigma_s^2)$ accumulated on the source training domain. When domain shift is present, source statistics cause wrong standardization and performance drops.
- **Transductive Batch Normalization (TBN / AdaBN)**: Normalizes test features using statistics $(\mu, \sigma^2)$ calculated directly from test batches. While effective under large batch sizes, TBN suffers from high variance and severe degradation when test batches are small or non-i.i.d.
- **TTN**: Formulates an adaptive interpolation between CBN and TBN based on the domain-shift sensitivity of each layer and channel.

## Mathematical Formulation (Equation 2)

For each batch normalization layer, TTN standardizes input $z$ using blended statistics:
$$\tilde{\mu} = \alpha \mu + (1 - \alpha) \mu_s$$
$$\tilde{\sigma}^2 = \alpha \sigma^2 + (1 - \alpha) \sigma_s^2 + \alpha (1 - \alpha) (\mu - \mu_s)^2$$
$$\hat{z} = \frac{z - \tilde{\mu}}{\sqrt{\tilde{\sigma}^2 + \epsilon}}, \quad z_{out} = \gamma \odot \hat{z} + \beta$$
where $\alpha \in [0, 1]$ is a channel-wise mixing weight:
- $\alpha = 0 \implies$ pure **CBN** (conventional source running statistics).
- $\alpha = 1 \implies$ pure **TBN** (transductive test batch statistics / AdaBN).
- $0 < \alpha < 1 \implies$ optimal trade-off balancing source knowledge and target adaptation.

## Two-Stage Post-Training Phase

1. **Stage 1 (Algorithm 1): Obtain Prior $A$**:
   - Generates simulated domain shift on source data via time-series perturbations ($x \to x'$).
   - Computes gradients of affine parameters $(\nabla \gamma, \nabla \beta)$ for clean and shifted inputs.
   - Computes gradient distance score:
     $$d = 1 - \frac{1}{2} (s_\gamma + s_\beta)$$
   - Defines prior $A = d^2$, and initializes $\alpha \leftarrow A$.
2. **Stage 2 (Algorithm 2): Optimize $\alpha$**:
   - Freezes all model weights $(\theta, \gamma, \beta)$ except $\alpha$.
   - Optimizes $\alpha$ on augmented source minibatches with task loss and prior regularization:
     $$\mathcal{L} = \mathcal{L}_{task}(f_\alpha(x'), y) + \lambda \|\alpha - A\|^2$$
   - Clamps $\alpha \in [0, 1]$.

## Comparison with AdaBN and LSTM-DANN

| Property | LSTM-DANN | AdaBN | TTN (This Implementation) |
|---|---|---|---|
| Domain Classifier | Yes (MLP + GRL) | None | None |
| Source Pre-training | Joint adversarial | Pure source regression | Pure source regression |
| Target Adaptation | Adversarial feature alignment | Replaces BN with test statistics ($\alpha=1$) | Interpolates source and test statistics via $\alpha$ (Eq. 2) |
| Sensitivity Aware | No | No | Yes (via Prior $A$ and channel-wise $\alpha$) |
| Small Batch Robustness | Moderate | Low (sensitive to batch size) | High (balances CBN & TBN smoothly) |
