import hydra
import torch
from omegaconf import DictConfig

from src.pipelines import get_pipeline


def test_hydra_config_loading():
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="config")
        assert isinstance(cfg, DictConfig)
        assert cfg.pipeline == "lstm_dann"
        assert "dataset" in cfg
        assert "model" in cfg
        assert "trainer" in cfg
        assert "optimizer" in cfg
        assert "loss" in cfg


def test_all_experiment_configs():
    experiments = [
        "lstm_dann_cmapss_single",
        "lstm_dann_cmapss_10trials",
        "ops_dann_ncmapss_single",
        "ops_dann_ncmapss_10trials",
    ]
    for exp in experiments:
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            cfg = hydra.compose(config_name="config", overrides=[f"experiment={exp}"])
            assert isinstance(cfg, DictConfig)
            assert cfg.task_name == exp
            pipeline = get_pipeline(cfg)
            assert pipeline is not None


def test_seed_everything_reproducibility():
    from utils.seed import seed_everything

    seed_everything(123)
    t1 = torch.randn(5, 5)

    seed_everything(123)
    t2 = torch.randn(5, 5)

    assert torch.equal(t1, t2)
