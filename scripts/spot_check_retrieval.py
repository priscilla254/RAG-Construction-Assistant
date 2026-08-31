"""Smoke-test retrieval: 5 known questions through vector-only and hybrid."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.chunker import load_chunks  # noqa: E402
from rag_assistant.config import Config  # noqa: E402
from rag_assistant.embedder import Embedder  # noqa: E402
from rag_assistant.keyword_search import KeywordSearch  # noqa: E402
from rag_assistant.retriever import HybridRetriever  # noqa: E402
from rag_assistant.vector_store import VectorStore  # noqa: E402

CHECKS = [
    {
        "question": "How high should scaffold guard rails or barriers be?",
        "expect_ids": {"hsg150_0090", "hsg150_0089"},
        "expect_hint": "950 mm / guard rails and toe boards",
    },
    {
        "question": "When must a construction site be notified to HSE?",
        "expect_ids": {"hsg150_0021"},
        "expect_hint": "30 days or 500 person days",
    },
    {
        "question": "Why use 110 V on a construction site instead of mains voltage?",
        "expect_ids": {"hsg141_0038", "hsg141_0033", "hsg141_0035"},
        "expect_hint": "reduced low voltage / fatal shock",
    },
    {
        "question": "What escape provision do habitable rooms in a dwellinghouse need?",
        "expect_ids": {
            "The_Merged_Approved_Documents_Oct24_0177",
            "The_Merged_Approved_Documents_Oct24_0178",
        },
        "expect_hint": "Part B clause 2.1 / 2.2",
    },
    {
        "question": "Must the rise and going of stair steps be consistent throughout a flight?",
        "expect_ids": {"The_Merged_Approved_Documents_Oct24_2300"},
        "expect_hint": "Part K clauses 1.5-1.6",
    },
]


def _print_hits(label: str, hits: list[dict], expect_ids: set[str]) -> bool:
    ids = [hit["chunk_id"] for hit in hits]
    ok = bool(expect_ids & set(ids))
    print(f"  {label}: {'HIT' if ok else 'MISS'}")
    for rank, hit in enumerate(hits, start=1):
        marker = " <<" if hit["chunk_id"] in expect_ids else ""
        preview = " ".join(hit["text"].split())[:140]
        extra = ""
        if "rrf_score" in hit:
            extra = f"  rrf={hit['rrf_score']:.4f}"
        elif "distance" in hit:
            extra = f"  d={hit['distance']:.3f}"
        print(f"    {rank}. {hit['chunk_id']}{extra}  {hit.get('section', '')}{marker}")
        print(f"       {preview}")
    return ok


def main() -> None:
    config = Config.from_yaml(ROOT / "config.yaml")
    embedder = Embedder.from_config(config)
    store = VectorStore(config.paths.chroma_dir)
    keywords = KeywordSearch(load_chunks(config.paths.chunks_path))
    hybrid = HybridRetriever.from_config(
        config,
        vector_store=store,
        keyword_search=keywords,
        embedder=embedder,
    )
    top_k = config.retrieval.k

    vector_hits_count = 0
    hybrid_hits_count = 0
    for index, check in enumerate(CHECKS, start=1):
        vector = embedder.embed_query(check["question"])
        vector_hits = store.query(
            vector,
            k=top_k,
            model_name=embedder.model_name,
            dimension=embedder.dimension,
        )
        hybrid_hits = hybrid.retrieve(check["question"], k=top_k)

        print("=" * 80)
        print(f"Q{index}: {check['question']}")
        print(f"  looking for {sorted(check['expect_ids'])} ({check['expect_hint']})")
        vector_hits_count += int(_print_hits("vector", vector_hits, check["expect_ids"]))
        hybrid_hits_count += int(_print_hits("hybrid", hybrid_hits, check["expect_ids"]))
        print()

    print(
        f"vector {vector_hits_count}/{len(CHECKS)}  "
        f"hybrid {hybrid_hits_count}/{len(CHECKS)}  "
        f"had an expected chunk in top-{top_k}"
    )


if __name__ == "__main__":
    main()
