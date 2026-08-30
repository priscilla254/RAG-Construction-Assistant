"""LLM generation (Groq by default)."""

from __future__ import annotations

from rag_assistant.config import GenerationConfig


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, query: str, contexts: list[dict]) -> str:
        raise NotImplementedError
