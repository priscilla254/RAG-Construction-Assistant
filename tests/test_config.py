import os

from rag_assistant.chunker import DocumentChunker
from rag_assistant.config import (
    ChunkingConfig,
    Config,
    EmbeddingConfig,
    GenerationConfig,
    RetrievalConfig,
)
from rag_assistant.embedder import Embedder


def test_embedding_config_loads_model_and_batch_size(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_dir: data/raw
  cleaned_dir: data/cleaned
  chunks_path: data/chunks.jsonl
  chroma_dir: data/chroma_db
  manifest_path: manifest.csv
chunking:
  chunk_size: 400
  chunk_overlap: 60
embedding:
  model_name: "BAAI/bge-small-en-v1.5"
  batch_size: 64
retrieval:
  k: 5
  vector_weight: 0.7
  keyword_weight: 0.3
generation:
  provider: groq
  model_name: llama-3.1-8b-instant
  temperature: 0.1
  max_tokens: 1024
""",
        encoding="utf-8",
    )
    config = Config.from_yaml(config_path)
    assert config.embedding.model_name == "BAAI/bge-small-en-v1.5"
    assert config.embedding.batch_size == 64
    assert config.generation.show_all_sources is True


def test_show_all_sources_can_be_turned_off(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_dir: data/raw
  cleaned_dir: data/cleaned
  chunks_path: data/chunks.jsonl
  chroma_dir: data/chroma_db
  manifest_path: manifest.csv
chunking:
  chunk_size: 400
  chunk_overlap: 60
embedding:
  model_name: "BAAI/bge-small-en-v1.5"
  batch_size: 64
retrieval:
  k: 5
  vector_weight: 0.7
  keyword_weight: 0.3
generation:
  provider: groq
  model_name: llama-3.1-8b-instant
  temperature: 0.1
  max_tokens: 1024
  show_all_sources: false
""",
        encoding="utf-8",
    )
    config = Config.from_yaml(config_path)
    assert config.generation.show_all_sources is False


def test_dataclass_defaults_hold_k_model_and_chunk_size():
    assert ChunkingConfig().chunk_size == 400
    assert ChunkingConfig().chunk_overlap == 60
    assert EmbeddingConfig().model_name == "BAAI/bge-small-en-v1.5"
    assert EmbeddingConfig().batch_size == 64
    assert RetrievalConfig().k == 5
    assert RetrievalConfig().max_per_source == 3
    assert GenerationConfig().model_name == "openai/gpt-oss-20b"
    assert GenerationConfig().max_tokens == 2048


def test_missing_yaml_keys_fall_back_to_dataclass_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_dir: data/raw
  cleaned_dir: data/cleaned
  chunks_path: data/chunks.jsonl
  chroma_dir: data/chroma_db
  manifest_path: manifest.csv
chunking: {}
embedding: {}
retrieval: {}
generation: {}
""",
        encoding="utf-8",
    )
    config = Config.from_yaml(config_path)
    assert config.chunking.chunk_size == 400
    assert config.embedding.model_name == "BAAI/bge-small-en-v1.5"
    assert config.retrieval.k == 5
    assert config.retrieval.max_per_source == 3
    assert config.generation.model_name == "openai/gpt-oss-20b"


def test_chunker_and_embedder_from_config_use_yaml_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_dir: data/raw
  cleaned_dir: data/cleaned
  chunks_path: data/chunks.jsonl
  chroma_dir: data/chroma_db
  manifest_path: manifest.csv
chunking:
  chunk_size: 120
  chunk_overlap: 10
embedding:
  model_name: unit-test-model
  batch_size: 8
retrieval:
  k: 3
generation:
  model_name: unit-test-llm
""",
        encoding="utf-8",
    )
    config = Config.from_yaml(config_path)
    chunker = DocumentChunker.from_config(config)
    embedder = Embedder.from_config(config)
    assert chunker.chunk_size == 120
    assert chunker.chunk_overlap == 10
    assert embedder.model_name == "unit-test-model"
    assert embedder.batch_size == 8
    assert config.retrieval.k == 3
    assert config.generation.model_name == "unit-test-llm"


def test_parse_secrets_text_reads_toml_and_dotenv():
    from rag_assistant.config import parse_secrets_text

    toml_values = parse_secrets_text('GROQ_API_KEY = "gsk_toml"\n')
    assert toml_values["GROQ_API_KEY"] == "gsk_toml"
    env_values = parse_secrets_text("GROQ_API_KEY=gsk_env\n")
    assert env_values["GROQ_API_KEY"] == "gsk_env"


def test_apply_streamlit_secrets_file_sets_env(tmp_path, monkeypatch):
    from rag_assistant.config import apply_streamlit_secrets_file

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = tmp_path / "secrets.toml"
    path.write_text("GROQ_API_KEY=gsk_from_file\n", encoding="utf-8")
    apply_streamlit_secrets_file(path)
    assert os.environ["GROQ_API_KEY"] == "gsk_from_file"


def test_apply_streamlit_secrets_file_does_not_override_env(tmp_path, monkeypatch):
    from rag_assistant.config import apply_streamlit_secrets_file

    monkeypatch.setenv("GROQ_API_KEY", "already-set")
    path = tmp_path / "secrets.toml"
    path.write_text('GROQ_API_KEY = "from-file"\n', encoding="utf-8")
    apply_streamlit_secrets_file(path)
    assert os.environ["GROQ_API_KEY"] == "already-set"
