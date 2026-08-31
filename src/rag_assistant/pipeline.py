"""End-to-end retrieval + generation pipeline."""

from __future__ import annotations

from rag_assistant.chunker import load_chunks
from rag_assistant.config import Config
from rag_assistant.embedder import Embedder
from rag_assistant.generator import (
    Generator,
    _citation_numbers,
    _cited_chunks,
    _passage_label,
)
from rag_assistant.keyword_search import KeywordSearch
from rag_assistant.retriever import HybridRetriever
from rag_assistant.vector_store import VectorStore

_SOURCES_MARK = "\n\nSources\n"


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        retriever: HybridRetriever,
        generator: Generator,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_config(cls, config: Config) -> RetrievalPipeline:
        embedder = Embedder.from_config(config)
        retriever = HybridRetriever.from_config(
            config,
            vector_store=VectorStore(config.paths.chroma_dir),
            keyword_search=KeywordSearch(load_chunks(config.paths.chunks_path)),
            embedder=embedder,
        )
        return cls(embedder, retriever, Generator(config.generation))

    def answer(
        self,
        question: str,
        show_all_sources: bool | None = None,
    ) -> dict:
        """Retrieve, generate, and return answer, sources, and chunks."""
        retrieved_chunks = self.retriever.retrieve(question)
        include_all = (
            self.generator.config.show_all_sources
            if show_all_sources is None
            else show_all_sources
        )
        combined = self.generator.generate(
            question,
            retrieved_chunks,
            show_all_sources=include_all,
        )
        answer = _answer_body(combined)
        return {
            "answer": answer,
            "sources": _source_records(retrieved_chunks, answer, include_all),
            "retrieved_chunks": retrieved_chunks,
        }


def _answer_body(text: str) -> str:
    if _SOURCES_MARK in text:
        return text.split(_SOURCES_MARK, 1)[0]
    return text


def _source_records(
    chunks: list[dict],
    answer: str,
    show_all_sources: bool,
) -> list[dict]:
    numbered = _cited_chunks(chunks)
    if not show_all_sources:
        cited = _citation_numbers(answer)
        numbered = [(index, chunk) for index, chunk in numbered if index in cited]
    return [
        {
            "n": index,
            "label": _passage_label(chunk),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        for index, chunk in numbered
    ]
