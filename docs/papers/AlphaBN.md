# What is Alpha-BN (α-BN)

**Alpha-BN (α-BN)** is a test-time batch statistics calibration method introduced in:
**"Test-Time Batch Statistics Calibration for Covariate Shift"** by Fuming You, Jingjing Li, Zhou Zhao (arXiv:2110.04065).

## Core Motivation
- **The Pitfall of Test-Time Normalization (T-BN / AdaBN)**:
  T-BN recalculates batch statistics on the target domain during inference. While this mitigates covariate shift, replacing source statistics with target statistics causes a **mismatch with source-trained model parameters**, which can perturb the learned discriminative structures.
- **The Solution (α-BN)**:
  Mixes the source running statistics and the target batch statistics during inference via a scalar hyperparameter $\alpha \in [0, 1]$. This simultaneously preserves the source model's discriminative representations and adapts to the target domain shift.

## Mathematical Formulation (Equations 3 & 4)

Given source running statistics $\{\mu_s, \sigma_s\}$ and target batch statistics $\{\mu_t, \sigma_t\}$:
$$\mu = \alpha \mu_s + (1 - \alpha) \mu_t$$
$$\sigma = \alpha \sigma_s + (1 - \alpha) \sigma_t$$
$$\hat{x} = \frac{x - \mu}{\sigma + \epsilon}, \quad y = \gamma \odot \hat{x} + \beta$$

- $\alpha = 1.0 \implies$ pure **Source (CBN)** (fixed source running statistics).
- $\alpha = 0.0 \implies$ pure **T-BN (AdaBN)** (target batch statistics only).
- $0 < \alpha < 1 \implies$ **calibrated mixture** (preserves source discriminative structure while adapting to shift).

## Key Characteristics
1. **Training-Free Post-Processing**:
   α-BN requires zero additional training or optimization at test time.
2. **Minimal Computational Overhead**:
   Inference time increases by less than $1\%$ compared to vanilla inference.
3. **Parameter Robustness**:
   The paper shows $\alpha \in [0.5, 0.9]$ achieves robust performance across domain generalization benchmarks.

## Comparison Across Normalization Methods

| Property | Conventional BN (CBN) | AdaBN / T-BN | TTN (ICLR 2023) | Alpha-BN (2021) |
|---|---|---|---|---|
| Source Statistics Weight | $1.0$ | $0.0$ | $1 - \alpha$ | $\alpha$ |
| Target Statistics Weight | $0.0$ | $1.0$ | $\alpha$ | $1 - \alpha$ |
| Variance Mixing | None | None | Law of Total Variance (with cross-term) | Direct std mixing: $\alpha \sigma_s + (1-\alpha)\sigma_t$ |
| Granularity | Layer-wise | Layer-wise | Channel-wise learnable $\alpha$ | Scalar $\alpha \in [0, 1]$ |
| Post-Training | None | None | 2-stage (Prior $A$ + optimization) | None (Training-free) |
