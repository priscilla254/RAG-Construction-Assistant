"""
Turn chunk text into dense vectors.

The public surface is two methods so a later model swap can treat queries
and documents differently without touching callers:

  embed_texts(list[str]) -> list[list[float]]   # index-time, batched
  embed_query(str) -> list[float]               # search-time

Callers must pass Chunk.embedding_text at index time, not Chunk.text,
so the Part/section/clause prefix is in the vector.

Vectors are L2-normalised. Changing model_name after the Chroma
collection exists requires a full rebuild -- VectorStore enforces that.
"""

from __future__ import annotations

from rag_assistant.chunker import Chunk
from rag_assistant.config import Config, EmbeddingConfig

# Official BGE retrieval instruction. Applied to queries only; document
# text is encoded as-is. MiniLM and other symmetric models skip this.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        defaults = EmbeddingConfig()
        self.model_name = model_name or defaults.model_name
        self.batch_size = defaults.batch_size if batch_size is None else batch_size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.device = _resolve_device(device)
        self._model = None

    @classmethod
    def from_config(cls, config: Config, device: str | None = None) -> Embedder:
        return cls(
            model_name=config.embedding.model_name,
            batch_size=config.embedding.batch_size,
            device=device,
        )

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode documents. Order is preserved: result[i] is texts[i]."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Encode a search query. BGE models get the retrieval prefix."""
        return self.embed_texts([self._prepare_query(query)])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed each chunk's embedding_text (prefix + body)."""
        return self.embed_texts([chunk.embedding_text for chunk in chunks])

    def _prepare_query(self, query: str) -> str:
        if "bge" in self.model_name.lower():
            return f"{BGE_QUERY_PREFIX}{query}"
        return query
