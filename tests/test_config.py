from rag_assistant.config import Config


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
