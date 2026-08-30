from rag_assistant.chunker import Chunk
from rag_assistant.keyword_search import KeywordSearch, tokenize


def _chunk(chunk_id: str, text: str, prefix: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_doc="hsg150.pdf",
        title="Health and safety in construction",
        source_url="",
        doc_type="broad_reference",
        chunk_type="topic_guidance",
        section="Section 3",
        page_number=1,
        text=text,
        context_prefix=prefix,
    )


def test_tokenize_keeps_measures_and_clause_numbers():
    tokens = tokenize("Barriers must be 950mm high. See clause 1.5.")
    assert "950mm" in tokens
    assert "1.5" in tokens
    assert "barriers" in tokens


def _filler(count: int = 10) -> list[Chunk]:
    # BM25 IDF is ~0 on a 2-3 document corpus; pad so query terms are rare.
    return [_chunk(f"filler-{i}", f"Unrelated weather notes number {i}.") for i in range(count)]


def test_bm25_ranks_the_chunk_that_contains_the_query_words():
    search = KeywordSearch(
        [
            _chunk("a", "Notify HSE if work lasts longer than 30 days or 500 person days."),
            _chunk("b", "Guard rails and toe boards prevent falls from scaffolds."),
            _chunk("c", "Habitable rooms need an escape window or a hall to a final exit."),
            *_filler(),
        ]
    )
    hits = search.search("notify HSE 30 days", k=2)
    assert hits[0]["chunk_id"] == "a"
    assert hits[0]["score"] > 0


def test_bm25_indexes_embedding_text_not_just_the_body():
    search = KeywordSearch(
        [
            _chunk("body-only", "The stair should comply with Table 3.2."),
            _chunk(
                "prefixed",
                "The stair should comply with Table 3.2.",
                prefix="Approved Document K: Protection from falling — clause 1.5",
            ),
            *_filler(),
        ]
    )
    hits = search.search("Approved Document K clause 1.5", k=2)
    assert hits[0]["chunk_id"] == "prefixed"


def test_empty_query_or_corpus_returns_nothing():
    assert KeywordSearch([]).search("anything", k=5) == []
    search = KeywordSearch([_chunk("a", "Guard rails prevent falls.")])
    assert search.search("   ", k=5) == []
    assert search.search("guard rails", k=0) == []
