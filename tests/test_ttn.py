import torch
from TTN.Model import TTN
from TTN.Model.TTNLayer import TTN1d, convert_bn_to_ttn
from TTN.Loss.Score import Score
from TTN.Loss import RegressionLoss
from TTN.Trainer import Trainer
from TTN.Tester import Tester


def test_ttn_forward_pass():
    batch_size = 16
    window_size = 30
    input_size = 24
    f_size = 32

    model = TTN(
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


def test_ttn_layer_modes():
    num_features = 8
    layer = TTN1d(num_features=num_features, initial_alpha=0.5)

    # Set mock source stats
    layer.running_mean.copy_(torch.zeros(num_features))
    layer.running_var.copy_(torch.ones(num_features))

    x = torch.randn(32, num_features) + 5.0

    # 1. CBN mode: uses running stats
    layer.set_mode("cbn")
    layer.eval()
    out_cbn = layer(x)

    # 2. TBN mode: uses batch stats
    layer.set_mode("tbn")
    out_tbn = layer(x)

    # 3. TTN mode: alpha=0 should match CBN, alpha=1 should match TBN
    layer.set_mode("ttn")
    layer.set_alpha(0.0)
    out_ttn_0 = layer(x)
    assert torch.allclose(out_cbn, out_ttn_0, atol=1e-5)

    layer.set_alpha(1.0)
    out_ttn_1 = layer(x)
    assert torch.allclose(out_tbn, out_ttn_1, atol=1e-5)

    # Intermediate alpha
    layer.set_alpha(0.5)
    out_ttn_mid = layer(x)
    assert not torch.allclose(out_ttn_mid, out_cbn)
    assert not torch.allclose(out_ttn_mid, out_tbn)


def test_obtain_prior_and_post_train():
    batch_size = 16
    window_size = 30
    input_size = 24

    model = TTN(
        input_size=input_size,
        lstm_hidden_size=32,
        f_size=16,
        num_layers=1,
    )

    dummy_data = [
        (torch.randn(batch_size, window_size, input_size), torch.full((batch_size,), 50.0))
        for _ in range(3)
    ]

    trainer = Trainer(
        model=model,
        source_train_dataloader=dummy_data,
        source_val_dataloader=dummy_data,
        regression_loss=RegressionLoss(p=1),
        score_loss=Score(),
        device=torch.device("cpu"),
    )

    # Test Algorithm 1: Obtain Prior A
    priors = trainer.obtain_prior(dummy_data, max_samples=32)
    assert len(priors) > 0
    for p in priors:
        assert torch.all(p >= 0.0) and torch.all(p <= 1.0)

    # Test Algorithm 2: Post-train alpha
    alphas = trainer.post_train(dummy_data, epochs=1, lr=1e-2)
    for a in alphas:
        assert torch.all(a >= 0.0) and torch.all(a <= 1.0)
