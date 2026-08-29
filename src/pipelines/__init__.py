from src.pipelines.base import BasePipeline
from src.pipelines.lstm_dann_pipeline import LSTMDANNPipeline
from src.pipelines.ops_dann_pipeline import OPSDANNPipeline

PIPELINE_REGISTRY = {
    "lstm_dann": LSTMDANNPipeline,
    "ops_dann": OPSDANNPipeline,
}


def get_pipeline(cfg) -> BasePipeline:
    pipeline_name = cfg.get("pipeline", "lstm_dann").lower()
    if pipeline_name not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{pipeline_name}'. Available: {list(PIPELINE_REGISTRY.keys())}"
        )
    pipeline_cls = PIPELINE_REGISTRY[pipeline_name]
    return pipeline_cls(cfg)
