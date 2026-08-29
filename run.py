import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines import get_pipeline


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Resolved Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    pipeline = get_pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
