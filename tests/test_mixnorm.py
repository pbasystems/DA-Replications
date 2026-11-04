import torch
from torch import nn
from MixNorm.Model import MixNorm
from MixNorm.Model.MixNormLayer import MixNorm1d, convert_bn_to_mixnorm


def test_mixnorm_forward_pass():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = MixNorm(
        input_size=input_size,
        lstm_hidden_size=64,
        f_size=f_size,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        regressor_first_hidden_size=32,
        mixing_scale=0.05,
    )

    x = torch.randn(batch_size, window_size, input_size)
    out = model(x)
    assert out.shape == (batch_size, 1)


def test_mixnorm_layer_mechanics():
    num_features = 8
    layer = MixNorm1d(
        num_features=num_features,
        mixing_scale=0.1,
        moving_speed=0.01,
        adaptive_tau=False,
    )

    # Set mock source stats
    mu_s = torch.zeros(num_features)
    var_s = torch.ones(num_features)
    layer.running_mean.copy_(mu_s)
    layer.running_var.copy_(var_s)
    layer.reset_online_stats()

    layer.eval()

    x = torch.randn(32, num_features) + 5.0

    # 1. Source Mode: online_update=False, m=0.0 -> uses static source stats
    layer.set_online_update(False)
    layer.set_mixing_scale(0.0)
    out_source = layer(x)
    expected_source = (x - mu_s) / torch.sqrt(var_s + layer.eps)
    assert torch.allclose(out_source, expected_source, atol=1e-5)

    # 2. Pure Local / T-BN mode: m=1.0 -> uses pure batch stats
    layer.set_mixing_scale(1.0)
    out_local = layer(x)
    mu_local = x.mean(dim=0)
    var_local = x.var(dim=0, unbiased=False)
    expected_local = (x - mu_local) / torch.sqrt(var_local + layer.eps)
    assert torch.allclose(out_local, expected_local, atol=1e-5)

    # 3. MixNorm mode: online_update=True, m=0.1
    layer.reset_online_stats()
    layer.set_online_update(True)
    layer.set_mixing_scale(0.1)

    out_mix = layer(x)
    # Ensure distinct from pure source and pure local
    assert not torch.allclose(out_mix, out_source)
    assert not torch.allclose(out_mix, out_local)

    # Verify global statistics were updated via EMA
    tau = layer.moving_speed
    expected_global_mean = (1.0 - tau) * mu_s + tau * mu_local
    assert torch.allclose(layer.global_mean, expected_global_mean, atol=1e-5)


def test_convert_bn_to_mixnorm():
    seq = nn.Sequential(
        nn.Linear(16, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
    )
    converted = convert_bn_to_mixnorm(seq, mixing_scale=0.05)
    assert isinstance(converted[1], MixNorm1d)
    assert converted[1].mixing_scale == 0.05
