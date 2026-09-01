"""Load project settings from config.yaml and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def parse_secrets_text(text: str) -> dict[str, str]:
    """Read GROQ-style KEY = value from TOML or dotenv `KEY=value` lines."""
    values: dict[str, str] = {}
    if tomllib is not None:
        try:
            loaded = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, str):
                    values[str(key)] = value
    if "GROQ_API_KEY" in values:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def apply_streamlit_secrets_file(path: Path) -> None:
    """Set GROQ_API_KEY from secrets.toml without using st.secrets.

    Streamlit Cloud writes App settings → Secrets to this file. Accessing
    st.secrets on invalid TOML (dotenv lines, a bare key) prints a parse
    error and never exposes the key. .env already wins if the var is set.
    """
    if os.environ.get("GROQ_API_KEY", "").strip():
        return
    if not path.is_file():
        return
    values = parse_secrets_text(path.read_text(encoding="utf-8"))
    key = (values.get("GROQ_API_KEY") or "").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key


@dataclass
class ChunkingConfig:
    chunk_size: int = 400
    chunk_overlap: int = 60


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64


@dataclass
class RetrievalConfig:
    k: int = 5
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    # After global hybrid, at most this many chunks from one PDF before
    # later ranks from other PDFs are considered. 0 disables the cap.
    # If other PDFs cannot fill k, overflow from the dominant PDF is used.
    max_per_source: int = 3


@dataclass
class GenerationConfig:
    provider: str = "groq"
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = 0.1
    max_tokens: int = 2048
    # True: list every retrieved passage (debug). False: cited [n] only (demo).
    show_all_sources: bool = True


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
        apply_streamlit_secrets_file(root / ".streamlit" / "secrets.toml")
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
