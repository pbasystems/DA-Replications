import torch

from common.grl import GradientReversal as CommonGRL
from lstm_dann.Model.GRL import GradientReversal as LSTM_GRL
from ops_dann.Model.GRL import GradientReversal as OPS_GRL


def test_lstm_grl_forward_and_backward():
    alpha = 0.75
    grl = LSTM_GRL(alpha=alpha)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Forward should be identity
    out = grl(x)
    assert torch.equal(out, x)

    # Backward should negate and scale by alpha
    loss = (out * torch.tensor([[2.0, 3.0], [4.0, 5.0]])).sum()
    loss.backward()

    expected_grad = -alpha * torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    if x.grad is not None:
        assert torch.allclose(x.grad, expected_grad)


def test_ops_grl_dynamic_alpha():
    grl = OPS_GRL(alpha=0.0)
    x = torch.tensor([5.0, 10.0], requires_grad=True)

    # Initially alpha=0
    out0 = grl(x)
    loss0 = (out0 * torch.tensor([1.0, 2.0])).sum()
    loss0.backward()
    if x.grad is not None:
        assert torch.allclose(x.grad, torch.tensor([-0.0, -0.0]))

    # Reset gradient and update alpha dynamically before forward pass
    if x.grad is not None:
        x.grad.zero_()
    grl.alpha = 0.5
    out = grl(x)
    loss = (out * torch.tensor([1.0, 2.0])).sum()
    loss.backward()

    expected_grad = -0.5 * torch.tensor([1.0, 2.0])
    if x.grad is not None:
        assert torch.allclose(x.grad, expected_grad)


def test_common_grl_forward_backward():
    alpha = 0.5
    grl = CommonGRL(alpha=alpha)
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    out = grl(x)
    assert torch.equal(out, x)

    loss = out.sum()
    loss.backward()
    if x.grad is not None:
        assert torch.allclose(x.grad, torch.tensor([-alpha, -alpha, -alpha]))
