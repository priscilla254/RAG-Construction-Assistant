"""Load project settings from config.yaml and environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int


@dataclass
class RetrievalConfig:
    k: int
    vector_weight: float
    keyword_weight: float


@dataclass
class GenerationConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int


@dataclass
class PathsConfig:
    raw_dir: Path
    cleaned_dir: Path
    chunks_path: Path
    chroma_dir: Path
    manifest_path: Path


@dataclass
class Config:
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        load_dotenv()
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_path = config_path.resolve()
        root = config_path.parent
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        def _p(value: str) -> Path:
            p = Path(value)
            return p if p.is_absolute() else root / p

        return cls(
            paths=PathsConfig(
                raw_dir=_p(raw["paths"]["raw_dir"]),
                cleaned_dir=_p(raw["paths"]["cleaned_dir"]),
                chunks_path=_p(raw["paths"]["chunks_path"]),
                chroma_dir=_p(raw["paths"]["chroma_dir"]),
                manifest_path=_p(raw["paths"]["manifest_path"]),
            ),
            chunking=ChunkingConfig(**raw["chunking"]),
            embedding=EmbeddingConfig(**raw["embedding"]),
            retrieval=RetrievalConfig(**raw["retrieval"]),
            generation=GenerationConfig(**raw["generation"]),
        )
