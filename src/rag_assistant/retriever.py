"""
Hybrid dense + keyword retrieval.

Each query is run through the vector store and BM25. The two ranked
lists are merged with weighted reciprocal rank fusion:

    score(d) = w_vec / (60 + rank_vec) + w_kw / (60 + rank_kw)

A document that appears in only one list still scores from that list.
The constant 60 is the usual RRF damping term; ranks are 1-based.
"""

from __future__ import annotations

from rag_assistant.embedder import Embedder
from rag_assistant.keyword_search import KeywordSearch
from rag_assistant.vector_store import VectorStore

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Combine ranked id lists. `rankings[i][0]` is the top hit of list i.

    Returns (id, fused_score) sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        embedder: Embedder,
        k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.embedder = embedder
        self.k = k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        top_k = self.k if k is None else k
        if top_k < 1 or not query.strip():
            return []

        pool = max(top_k * 4, 20)
        vector_hits = self.vector_store.query(
            self.embedder.embed_query(query),
            pool,
            model_name=self.embedder.model_name,
            dimension=self.embedder.dimension,
        )
        keyword_hits = self.keyword_search.search(query, pool)

        fused = reciprocal_rank_fusion(
            [
                [hit["chunk_id"] for hit in vector_hits],
                [hit["chunk_id"] for hit in keyword_hits],
            ],
            weights=[self.vector_weight, self.keyword_weight],
        )

        by_id: dict[str, dict] = {hit["chunk_id"]: hit for hit in keyword_hits}
        by_id.update({hit["chunk_id"]: hit for hit in vector_hits})

        results: list[dict] = []
        for chunk_id, rrf_score in fused:
            hit = dict(by_id[chunk_id])
            hit["rrf_score"] = rrf_score
            results.append(hit)
            if len(results) >= top_k:
                break
        return results
