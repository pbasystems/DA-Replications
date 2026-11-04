import torch
from torch import nn
from AlphaBN.Model import AlphaBN
from AlphaBN.Model.AlphaBNLayer import AlphaBN1d, convert_bn_to_alpha_bn
from AlphaBN.Tester import Tester
from AlphaBN.Loss.Score import Score


def test_alpha_bn_forward_pass():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = AlphaBN(
        input_size=input_size,
        lstm_hidden_size=64,
        f_size=f_size,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        regressor_first_hidden_size=32,
    )

    x = torch.randn(batch_size, window_size, input_size)
    out = model(x)
    assert out.shape == (batch_size, 1)


def test_alpha_bn_layer_calibration():
    num_features = 8
    layer = AlphaBN1d(num_features=num_features, default_alpha=0.7)

    # Set mock source stats
    mu_s = torch.zeros(num_features)
    var_s = torch.ones(num_features)
    layer.running_mean.copy_(mu_s)
    layer.running_var.copy_(var_s)

    x = torch.randn(32, num_features) + 5.0

    # 1. Source mode (alpha=1.0): uses source running stats
    layer.eval()
    layer.set_alpha(1.0)
    out_source = layer(x)

    # 2. T-BN mode (alpha=0.0): uses target batch stats
    layer.set_alpha(0.0)
    out_tbn = layer(x)

    # 3. Alpha-BN mode (alpha=0.7): calibrated blend
    layer.set_alpha(0.7)
    out_alpha_bn = layer(x)

    # Verify that Alpha-BN output is distinct from pure source and pure TBN
    assert not torch.allclose(out_alpha_bn, out_source)
    assert not torch.allclose(out_alpha_bn, out_tbn)

    # Verify exact math: mu = 0.7 * mu_s + 0.3 * mu_t
    mu_t = x.mean(dim=0)
    sigma_s = torch.sqrt(var_s + layer.eps)
    sigma_t = torch.sqrt(x.var(dim=0, unbiased=False) + layer.eps)
    expected_mu = 0.7 * mu_s + 0.3 * mu_t
    expected_sigma = 0.7 * sigma_s + 0.3 * sigma_t
    expected_out = (x - expected_mu) / expected_sigma
    assert torch.allclose(out_alpha_bn, expected_out, atol=1e-5)


def test_convert_bn_to_alpha_bn():
    seq = nn.Sequential(
        nn.Linear(16, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
    )
    converted = convert_bn_to_alpha_bn(seq, default_alpha=0.7)
    assert isinstance(converted[1], AlphaBN1d)
    assert converted[1].alpha == 0.7


def test_alpha_bn_device_safety():
    num_features = 4
    layer = AlphaBN1d(num_features=num_features, default_alpha=0.5)
    layer.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)
    x = torch.randn(8, num_features, device=device)

    out = layer(x)
    assert out.shape == (8, num_features)
    assert out.device == x.device
