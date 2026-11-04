import torch
from AdaBN.Model import AdaBN
from AdaBN.Tester import Tester
from AdaBN.Loss.Score import Score


def test_adabn_forward_pass():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = AdaBN(
        input_size=input_size,
        lstm_hidden_size=64,
        f_size=f_size,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        regressor_first_hidden_size=32,
    )

    x = torch.randn(batch_size, window_size, input_size)
    reg_out = model(x)

    # AdaBN only outputs regression (no classification or GRL)
    assert reg_out.shape == (batch_size, 1)


def test_adabn_test_time_adaptation():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = AdaBN(
        input_size=input_size,
        lstm_hidden_size=64,
        f_size=f_size,
        num_layers=1,
    )

    # Source forward to set initial running stats
    source_x = torch.randn(batch_size, window_size, input_size) + 1.0
    model.train()
    _ = model(source_x)

    bn_layer = model.feature_extractor.bn_proj
    initial_mean = bn_layer.running_mean.clone()

    # Target data with distinct distribution
    target_x = torch.randn(batch_size, window_size, input_size) + 10.0
    target_loader = [(target_x, torch.zeros(batch_size))]

    tester = Tester(model=model, score_loss=Score(), device=torch.device("cpu"))
    tester.adapt(target_loader)

    adapted_mean = bn_layer.running_mean.clone()

    # Verify that BN running stats have adapted to the target domain
    assert not torch.allclose(initial_mean, adapted_mean)
