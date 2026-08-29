import math

import torch

from lstm_dann.Loss.Score import Score as LSTMScore
from ops_dann.Loss.Score import Score as OPSScore
from utils.metrics import (
    mae_score,
    nejjar_cmapss_score,
    rmse_score,
    saxena_cmapss_score,
)


def test_rmse_and_mae():
    y_pred = torch.tensor([10.0, 20.0, 30.0])
    y_true = torch.tensor([12.0, 20.0, 26.0])
    # errors: -2, 0, +4 -> sq: 4, 0, 16 -> mean: 20/3
    assert math.isclose(rmse_score(y_pred, y_true), math.sqrt(20.0 / 3.0), rel_tol=1e-5)
    # abs errors: 2, 0, 4 -> mean: 6/3 = 2.0
    assert math.isclose(mae_score(y_pred, y_true), 2.0, rel_tol=1e-5)


def test_saxena_cmapss_score_properties():
    # 1. Zero error gives zero score
    y = torch.tensor([50.0, 100.0])
    score_zero = saxena_cmapss_score(y, y)
    assert math.isclose(score_zero.item(), 0.0, abs_tol=1e-6)

    # 2. Asymmetry: late prediction (y_pred > y_true) is penalized more heavily than early (y_pred < y_true)
    y_true = torch.tensor([50.0])
    y_early = torch.tensor([40.0])  # d = -10 -> exp(10/13) - 1 ≈ 1.158
    y_late = torch.tensor([60.0])  # d = +10 -> exp(10/10) - 1 = e - 1 ≈ 1.718

    s_early = saxena_cmapss_score(y_early, y_true).item()
    s_late = saxena_cmapss_score(y_late, y_true).item()

    assert s_late > s_early
    assert math.isclose(s_late, math.e - 1.0, rel_tol=1e-4)

    # 3. Matches LSTMScore module
    lstm_score_fn = LSTMScore(a_1=13.0, a_2=10.0)
    assert math.isclose(
        lstm_score_fn(y_late, y_true).item(),
        saxena_cmapss_score(y_late, y_true).item(),
        rel_tol=1e-5,
    )


def test_nejjar_cmapss_score_properties():
    # 1. Zero error gives N (since exp(0) = 1 per sample)
    y = torch.tensor([50.0, 100.0])
    score_zero = nejjar_cmapss_score(y, y)
    assert math.isclose(score_zero.item(), 2.0, rel_tol=1e-5)

    # 2. Asymmetry: late prediction penalized more than early
    y_true = torch.tensor([50.0])
    y_early = torch.tensor([40.0])  # d = -10 -> exp(10/13) ≈ 2.158
    y_late = torch.tensor([60.0])  # d = +10 -> exp(10/10) = e ≈ 2.718

    s_early = nejjar_cmapss_score(y_early, y_true).item()
    s_late = nejjar_cmapss_score(y_late, y_true).item()

    assert s_late > s_early
    assert math.isclose(s_late, math.e, rel_tol=1e-4)

    # 3. Matches OPSScore module
    ops_score_fn = OPSScore(alpha_over=1.0 / 10.0, alpha_under=1.0 / 13.0)
    assert math.isclose(
        ops_score_fn(y_late, y_true).item(),
        nejjar_cmapss_score(y_late, y_true).item(),
        rel_tol=1e-5,
    )
