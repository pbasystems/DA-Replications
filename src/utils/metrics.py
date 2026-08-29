from __future__ import annotations

import torch


def rmse_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item())


def mae_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return float(torch.mean(torch.abs(y_pred - y_true)).item())


def saxena_cmapss_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    a1: float = 13.0,
    a2: float = 10.0,
) -> torch.Tensor:
    """
    Compute asymmetric NASA C-MAPSS score (Saxena et al., 2008 / Costa et al., 2019).
    Equation:
      d = y_pred - y_true
      s = exp(-d / a1) - 1   if d < 0 (early prediction / underestimation)
      s = exp(d / a2) - 1    if d >= 0 (late prediction / overestimation)
    """
    residue = y_pred - y_true
    score = torch.where(
        residue < 0,
        torch.exp(-residue / a1) - 1.0,
        torch.exp(residue / a2) - 1.0,
    )
    return torch.sum(score)


def nejjar_cmapss_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha_over: float = 1.0 / 10.0,
    alpha_under: float = 1.0 / 13.0,
) -> torch.Tensor:
    """
    Compute asymmetric N-CMAPSS score (Nejjar et al., 2023).
    Equation:
      d = y_pred - y_true
      s = exp(alpha_over * |d|)   if d >= 0
      s = exp(alpha_under * |d|)  if d < 0
    """
    residue = y_pred - y_true
    alpha = torch.where(residue >= 0, alpha_over, alpha_under)
    return torch.sum(torch.exp(alpha * torch.abs(residue)))
