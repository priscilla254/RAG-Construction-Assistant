"""
Chroma wrapper for the embedded corpus.

Collection metadata stores the embedding model name and dimension. add()
always rebuilds the collection from scratch -- incremental upsert is not
worth the stale-vector risk on a 4,000-chunk corpus. query() refuses to
search if the caller's model or dimension does not match what was stored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rag_assistant.chunker import Chunk

COLLECTION_NAME = "construction_docs"
CHROMA_ADD_BATCH = 500
METADATA_MODEL = "embedding_model"
METADATA_DIMENSION = "embedding_dimension"


def _silence_chroma_telemetry() -> None:
    """
    Stop Chroma's PostHog client from printing a false failure.

    chromadb 0.5.x still calls posthog.capture(user_id, event, props)
    after telemetry is turned off. Current posthog only accepts one
    positional argument, so every client start logs
    "Failed to send telemetry event ... capture() takes 1 positional
    argument but 3 were given". The collection write already succeeded.
    """
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    try:
        import posthog

        posthog.disabled = True
    except ImportError:
        pass


class ModelMismatchError(RuntimeError):
    """Raised when the store was built with a different embedding model."""


class VectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = COLLECTION_NAME,
        client=None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            _silence_chroma_telemetry()
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        model_name: str,
        dimension: int,
    ) -> None:
        """Replace the collection with this corpus. No incremental upsert."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length"
            )
        if embeddings and len(embeddings[0]) != dimension:
            raise ValueError(
                f"embedding length {len(embeddings[0])} does not match "
                f"declared dimension {dimension}"
            )

        self._delete_collection()
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                METADATA_MODEL: model_name,
                METADATA_DIMENSION: dimension,
            },
        )
        for start in range(0, len(chunks), CHROMA_ADD_BATCH):
            end = start + CHROMA_ADD_BATCH
            batch = chunks[start:end]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings[start:end],
                metadatas=[_chunk_metadata(chunk) for chunk in batch],
            )

    def query(
        self,
        embedding: list[float],
        k: int,
        *,
        model_name: str | None = None,
        dimension: int | None = None,
    ) -> list[dict]:
        collection = self.client.get_collection(self.collection_name)
        stored_model, stored_dim = self._stored_identity(collection)
        if model_name is not None and model_name != stored_model:
            raise ModelMismatchError(
                f"store was built with {stored_model!r}, query used {model_name!r}. "
                "Rebuild the collection after changing embedding.model_name."
            )
        if dimension is not None and dimension != stored_dim:
            raise ModelMismatchError(
                f"store dimension is {stored_dim}, query used {dimension}."
            )
        if stored_dim is not None and len(embedding) != stored_dim:
            raise ModelMismatchError(
                f"query vector length {len(embedding)} does not match "
                f"store dimension {stored_dim}."
            )

        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()) or k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            row = dict(metadata or {})
            row["chunk_id"] = chunk_id
            row["text"] = text
            row["distance"] = distance
            hits.append(row)
        return hits

    def _delete_collection(self) -> None:
        names = {collection.name for collection in self.client.list_collections()}
        if self.collection_name in names:
            self.client.delete_collection(self.collection_name)

    def _stored_identity(self, collection) -> tuple[str | None, int | None]:
        metadata = collection.metadata or {}
        model = metadata.get(METADATA_MODEL)
        raw_dim = metadata.get(METADATA_DIMENSION)
        dimension = int(raw_dim) if raw_dim is not None else None
        return model, dimension


def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "source_doc": chunk.source_doc,
        "title": chunk.title,
        "source_url": chunk.source_url,
        "doc_type": chunk.doc_type,
        "chunk_type": chunk.chunk_type,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "part_code": chunk.part_code,
        "clause_number": chunk.clause_number,
        "context_prefix": chunk.context_prefix,
        "topic": chunk.topic,
        "subtopic": chunk.subtopic,
        "paragraph_numbers": chunk.paragraph_numbers,
    }


def index_corpus(config) -> int:
    """Embed chunks.jsonl and rebuild the Chroma collection."""
    from rag_assistant.chunker import load_chunks
    from rag_assistant.embedder import Embedder

    chunks = load_chunks(config.paths.chunks_path)
    embedder = Embedder.from_config(config)
    print(
        f"Embedding {len(chunks)} chunks with {embedder.model_name} "
        f"(dim {embedder.dimension}, batch {embedder.batch_size}, "
        f"device {embedder.device})"
    )
    embeddings = embedder.embed_chunks(chunks)
    store = VectorStore(config.paths.chroma_dir)
    store.add(
        chunks,
        embeddings,
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    print(f"Wrote {len(chunks)} vectors to {config.paths.chroma_dir}")
    return len(chunks)
