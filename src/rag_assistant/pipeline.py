"""End-to-end retrieval + generation pipeline."""

from __future__ import annotations

from rag_assistant.config import Config
from rag_assistant.generator import Generator
from rag_assistant.retriever import HybridRetriever


class RetrievalPipeline:
    def __init__(self, config: Config, retriever: HybridRetriever, generator: Generator) -> None:
        self.config = config
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str) -> dict:
        raise NotImplementedError
