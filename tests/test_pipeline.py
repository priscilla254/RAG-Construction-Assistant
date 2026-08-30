from rag_assistant.config import Config
from rag_assistant.pipeline import RetrievalPipeline


def test_config_loads():
    config = Config.from_yaml("config.yaml")
    assert config.retrieval.k > 0
    assert config.chunking.chunk_size > config.chunking.chunk_overlap


def test_pipeline_init():
    config = Config.from_yaml("config.yaml")
    pipeline = RetrievalPipeline(config=config, retriever=None, generator=None)  # type: ignore[arg-type]
    assert pipeline.config is config
