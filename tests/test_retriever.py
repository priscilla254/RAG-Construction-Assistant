from types import SimpleNamespace

from rag_assistant.config import RetrievalConfig
from rag_assistant.retriever import HybridRetriever, reciprocal_rank_fusion


class _FakeStore:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.last_k: int | None = None

    def query(self, embedding, k: int, **kwargs) -> list[dict]:
        self.last_k = k
        return self.hits[:k]


class _FakeKeywords:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def search(self, query: str, k: int) -> list[dict]:
        return self.hits[:k]


class _FakeEmbedder:
    model_name = "BAAI/bge-small-en-v1.5"
    dimension = 4

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def test_hybrid_retriever_init():
    retriever = HybridRetriever(
        vector_store=None,  # type: ignore[arg-type]
        keyword_search=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
    )
    assert retriever.k == 5


def test_hybrid_retriever_from_config_reads_k_and_weights():
    config = SimpleNamespace(
        retrieval=RetrievalConfig(k=3, vector_weight=0.9, keyword_weight=0.1)
    )
    retriever = HybridRetriever.from_config(
        config,  # type: ignore[arg-type]
        vector_store=_FakeStore([]),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords([]),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
    )
    assert retriever.k == 3
    assert retriever.vector_weight == 0.9
    assert retriever.keyword_weight == 0.1
    assert retriever.max_per_source == 3


def test_reciprocal_rank_fusion_promotes_items_on_both_lists():
    fused = reciprocal_rank_fusion(
        [
            ["vec-only", "shared", "vec-third"],
            ["kw-only", "shared"],
        ],
        weights=[0.7, 0.3],
    )
    ids = [chunk_id for chunk_id, _score in fused]
    assert ids[0] == "shared"
    assert "vec-only" in ids
    assert "kw-only" in ids


def test_retrieve_fuses_vector_and_keyword_hits():
    vector_hits = [
        {"chunk_id": "v1", "text": "vector first"},
        {"chunk_id": "both", "text": "from vector"},
    ]
    keyword_hits = [
        {"chunk_id": "k1", "text": "keyword first", "score": 4.2},
        {"chunk_id": "both", "text": "from keyword", "score": 3.1},
    ]
    retriever = HybridRetriever(
        vector_store=_FakeStore(vector_hits),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords(keyword_hits),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=3,
        vector_weight=0.7,
        keyword_weight=0.3,
    )
    hits = retriever.retrieve("guard rails")
    ids = [hit["chunk_id"] for hit in hits]
    assert ids[0] == "both"
    assert set(ids) == {"both", "v1", "k1"}
    assert hits[0]["text"] == "from vector"
    assert "rrf_score" in hits[0]


def test_retrieve_asks_each_side_for_a_candidate_pool():
    store = _FakeStore([{"chunk_id": "a", "text": "a"}])
    retriever = HybridRetriever(
        vector_store=store,  # type: ignore[arg-type]
        keyword_search=_FakeKeywords([]),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
    )
    retriever.retrieve("anything")
    assert store.last_k == 20


def test_retrieve_empty_query_returns_nothing():
    retriever = HybridRetriever(
        vector_store=_FakeStore([{"chunk_id": "a", "text": "a"}]),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords([{"chunk_id": "a", "text": "a"}]),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
    )
    assert retriever.retrieve("   ") == []


def _hits(*rows: tuple[str, str]) -> list[dict]:
    return [{"chunk_id": chunk_id, "text": chunk_id, "source_doc": source} for chunk_id, source in rows]


def test_max_per_source_lets_a_later_pdf_into_top_k():
    vector_hits = _hits(
        ("s150_a", "hsg150.pdf"),
        ("s150_b", "hsg150.pdf"),
        ("s150_c", "hsg150.pdf"),
        ("s150_d", "hsg150.pdf"),
        ("s150_e", "hsg150.pdf"),
        ("s141_a", "hsg141.pdf"),
        ("mad_a", "The_Merged_Approved_Documents_Oct24.pdf"),
    )
    retriever = HybridRetriever(
        vector_store=_FakeStore(vector_hits),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords(vector_hits),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        max_per_source=3,
    )
    hits = retriever.retrieve("guard rails")
    ids = [hit["chunk_id"] for hit in hits]
    assert ids == ["s150_a", "s150_b", "s150_c", "s141_a", "mad_a"]
    assert ids.count("s150_d") == 0


def test_max_per_source_backfills_when_only_one_pdf_has_hits():
    vector_hits = _hits(
        ("s150_a", "hsg150.pdf"),
        ("s150_b", "hsg150.pdf"),
        ("s150_c", "hsg150.pdf"),
        ("s150_d", "hsg150.pdf"),
        ("s150_e", "hsg150.pdf"),
    )
    retriever = HybridRetriever(
        vector_store=_FakeStore(vector_hits),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords(vector_hits),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        max_per_source=3,
    )
    hits = retriever.retrieve("guard rails")
    assert [hit["chunk_id"] for hit in hits] == [
        "s150_a",
        "s150_b",
        "s150_c",
        "s150_d",
        "s150_e",
    ]


def test_max_per_source_zero_keeps_the_global_ranking():
    vector_hits = _hits(
        ("s150_a", "hsg150.pdf"),
        ("s150_b", "hsg150.pdf"),
        ("s150_c", "hsg150.pdf"),
        ("s150_d", "hsg150.pdf"),
        ("s150_e", "hsg150.pdf"),
        ("s141_a", "hsg141.pdf"),
    )
    retriever = HybridRetriever(
        vector_store=_FakeStore(vector_hits),  # type: ignore[arg-type]
        keyword_search=_FakeKeywords(vector_hits),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        max_per_source=0,
    )
    hits = retriever.retrieve("guard rails")
    assert [hit["chunk_id"] for hit in hits] == [
        "s150_a",
        "s150_b",
        "s150_c",
        "s150_d",
        "s150_e",
    ]
