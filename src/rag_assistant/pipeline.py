"""End-to-end retrieval + generation.

`answer()` is the only method the CLI and Streamlit app need: retrieve,
generate, then return the prose, the numbered source list, and the raw
chunks. The source list is built from passage headings so the model
cannot invent document names.
"""

from __future__ import annotations

from rag_assistant.config import Config
from rag_assistant.embedder import Embedder
from rag_assistant.generator import Generator, cited_passages, passage_label
from rag_assistant.retriever import HybridRetriever

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
        retriever = HybridRetriever.build(config)
        return cls(retriever.embedder, retriever, Generator(config.generation))

    def answer(
        self,
        question: str,
        show_all_sources: bool | None = None,
    ) -> dict:
        """Retrieve, generate, and return answer, sources, and chunks.

        `show_all_sources=None` uses `generation.show_all_sources` from
        config. The Streamlit demo passes False (cited [n] only).
        """
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
    """Drop a trailing Sources block if the model emitted one anyway."""
    if _SOURCES_MARK in text:
        return text.split(_SOURCES_MARK, 1)[0]
    return text


def _source_records(
    chunks: list[dict],
    answer: str,
    show_all_sources: bool,
) -> list[dict]:
    return [
        {
            "n": index,
            "label": passage_label(chunk),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        for index, chunk in cited_passages(
            chunks, answer=answer, show_all_sources=show_all_sources
        )
    ]
