import torch

from lstm_dann.Model import LSTM_DANN
from ops_dann.Model import OPSDANNHard


def test_lstm_dann_forward_pass():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = LSTM_DANN(
        input_size=input_size,
        hidden_size=64,
        f_size=f_size,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        classifier_dropout=0.3,
        alpha=0.8,
    )

    x = torch.randn(batch_size, window_size, input_size)
    reg_out, cls_out = model(x)

    assert reg_out.shape == (batch_size, 1)
    assert cls_out.shape == (batch_size, 1)


def test_ops_dann_hard_forward_pass():
    batch_size = 16
    channels = 18
    window_size = 50
    num_phases = 3

    model = OPSDANNHard(input_channels=channels, num_phases=num_phases)

    x = torch.randn(batch_size, window_size, channels)
    phase = torch.randint(0, num_phases, (batch_size,))

    rul_out, domain_out = model(x, phase)

    assert rul_out.shape == (batch_size, 1)
    assert domain_out.shape == (batch_size, 1)
