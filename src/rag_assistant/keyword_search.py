"""
BM25 keyword search over the same strings the embedder indexed.

Documents are tokenised from Chunk.embedding_text so a short clause still
carries its Part/section/clause breadcrumb, matching the dense index.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag_assistant.chunker import Chunk

# Keep numbers and hyphenated measures ("110", "950mm", "1.5") as tokens.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class KeywordSearch:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tokens = [tokenize(chunk.embedding_text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokens) if self.chunks else None

    def search(self, query: str, k: int) -> list[dict]:
        if self._bm25 is None or k < 1 or not query.strip():
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[dict] = []
        for index, score in ranked:
            if score <= 0:
                break
            hits.append(_hit(self.chunks[index], float(score)))
            if len(hits) >= k:
                break
        return hits


def _hit(chunk: Chunk, score: float) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": score,
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
