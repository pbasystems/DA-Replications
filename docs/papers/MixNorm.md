# What is Mix-Norm (MixNorm)

**Mix-Norm** is a test-time adaptation framework introduced in:
**"MIXNORM: Test-Time Adaptation through Online Normalization Estimation"** by Xuefeng Hu, Gokhan Uzunbas, Shen Yan, Rui Wang, Avi Schwarzschild, Margret Keuper (arXiv:2110.11478).

## Core Motivation
- **The Pitfall of Test-Time Batch Normalization (T-BN / AdaBN)**:
  T-BN relies solely on current mini-batch statistics ($\mu_{local}, \sigma^2_{local}$). When mini-batch size $B$ is small or individual batches fluctuate, batch statistics exhibit high variance and estimation error.
- **The Online Continuous Shift Problem**:
  In real-world deployment, target data arrives sequentially as an online data stream. Completely resetting statistics per batch loses historical target distribution information.
- **The Solution (Mix-Norm)**:
  Mix-Norm decomposes normalization estimation into two complementary components:
  1. **Global Statistics** ($\mu_{global}, \sigma^2_{global}$): Maintained and updated online via Exponential Moving Average (EMA) over the incoming target stream, anchored at the source model's running statistics.
  2. **Local Statistics** ($\mu_{local}, \sigma^2_{local}$): Estimated on the current test mini-batch (and optionally augmented views).
  3. **Adaptive Mixing**: Blends global and local statistics via mixing scale $m$:
     $$\mu_{mixed} = (1 - m) \mu_{global} + m \mu_{local}$$
     $$\sigma^2_{mixed} = (1 - m) \sigma^2_{global} + m \sigma^2_{local}$$

---

## Mathematical Formulation (Algorithms 1 & 2)

### 1. Online Global Statistics Update
For each incoming batch $B$:
$$\tau = \min\left(0.99, \max\left(10^{-4}, \tau_{max} \cdot 10^{-3/B}\right)\right)$$
$$\mu_{global} \leftarrow (1 - \tau) \mu_{global} + \tau \mu_{local}$$
$$\sigma^2_{global} \leftarrow (1 - \tau) \sigma^2_{global} + \tau \sigma^2_{local}$$

- $\tau_{max} \approx 0.9$: controls maximum moving speed.
- The adaptive factor $10^{-3/B}$ automatically scales the momentum $\tau$ depending on the current batch size $B$. When $B$ is small, $\tau$ decreases to avoid noisy drift; when $B$ is large, $\tau$ increases to rapidly absorb accurate target information.

### 2. Normalization Mixing
$$\hat{x} = \frac{x - \mu_{mixed}}{\sqrt{\sigma^2_{mixed} + \epsilon}}$$
$$y = \gamma \odot \hat{x} + \beta$$

- $m = 0.0 \implies$ pure **Global / Online EMA** adaptation (or Source Only if $\tau = 0$).
- $m = 1.0 \implies$ pure **Local Batch Normalization** (T-BN / AdaBN).
- $0 < m < 1 \implies$ **Mix-Norm**, combining global stability with local batch responsiveness.

---

## Comprehensive Comparison Across Normalization Adaptations

| Method | Source | Test Statistics Used | Variance Mixing Formulation | Parameter Updates | Online Streaming Support |
|---|---|---|---|---|---|
| **Conventional BN (CBN)** | Ioffe & Szegedy (2015) | Static Source Running Stats | None | None | No |
| **AdaBN / T-BN** | Li et al. (2016) | Target Mini-Batch Stats | None | None | No |
| **Alpha-BN** | You et al. (2021) | Static Source + Target Batch | $\alpha \sigma_s + (1-\alpha) \sigma_t$ (Std Dev linear blend) | None (Scalar $\alpha$) | No |
| **TTN** | Lim et al. (ICLR 2023) | Static Source + Target Batch | Law of Total Variance with cross-term: $\alpha \sigma_t^2 + (1-\alpha)\sigma_s^2 + \alpha(1-\alpha)(\mu_t-\mu_s)^2$ | Channel-wise $\alpha$ optimized via post-training | No |
| **Mix-Norm** | Hu et al. (2021) | Online EMA Target Stream + Target Batch | $(1-m)\sigma^2_{global} + m \sigma^2_{local}$ | None (Online EMA update of buffers) | **Yes** (Continuous Online Adaptation) |
