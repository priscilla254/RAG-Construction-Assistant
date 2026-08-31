from types import SimpleNamespace

from rag_assistant.config import Config
from rag_assistant.pipeline import RetrievalPipeline


def test_config_loads():
    config = Config.from_yaml("config.yaml")
    assert config.retrieval.k > 0
    assert config.chunking.chunk_size > config.chunking.chunk_overlap


class _FakeRetriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.seen: list[str] = []

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        self.seen.append(question)
        return self.chunks


class _FakeGenerator:
    def __init__(self, reply: str, show_all_sources: bool = True) -> None:
        self.config = SimpleNamespace(show_all_sources=show_all_sources)
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, question, retrieved_chunks, show_all_sources=None):
        self.calls.append(
            {
                "question": question,
                "chunks": retrieved_chunks,
                "show_all_sources": show_all_sources,
            }
        )
        return self.reply


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "mad_0177",
            "text": "Habitable rooms should have an escape window.",
            "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
            "context_prefix": "Approved Document B — clause 2.1",
        },
        {
            "chunk_id": "mad_flats",
            "text": "Flats have a similar provision.",
            "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
            "context_prefix": "Approved Document B — clause 3.15",
        },
    ]


def test_pipeline_init_stores_embedder_retriever_and_generator():
    embedder = object()
    retriever = _FakeRetriever([])
    generator = _FakeGenerator("unused")
    pipeline = RetrievalPipeline(embedder, retriever, generator)
    assert pipeline.embedder is embedder
    assert pipeline.retriever is retriever
    assert pipeline.generator is generator


def test_answer_returns_prose_sources_and_chunks():
    chunks = _chunks()
    retriever = _FakeRetriever(chunks)
    generator = _FakeGenerator(
        "Habitable rooms need an escape window. [1]\n\n"
        "Sources\n[1] Approved Document B — clause 2.1\n"
        "[2] Approved Document B — clause 3.15"
    )
    pipeline = RetrievalPipeline(object(), retriever, generator)

    result = pipeline.answer("What escape provision is required?")

    assert retriever.seen == ["What escape provision is required?"]
    assert generator.calls[0]["question"] == "What escape provision is required?"
    assert generator.calls[0]["chunks"] == chunks
    assert result["answer"] == "Habitable rooms need an escape window. [1]"
    assert "Sources" not in result["answer"]
    assert result["retrieved_chunks"] == chunks
    assert result["sources"] == [
        {
            "n": 1,
            "label": "Approved Document B — clause 2.1",
            "chunk_id": "mad_0177",
        },
        {
            "n": 2,
            "label": "Approved Document B — clause 3.15",
            "chunk_id": "mad_flats",
        },
    ]


def test_answer_can_list_only_cited_sources():
    retriever = _FakeRetriever(_chunks())
    generator = _FakeGenerator(
        "Habitable rooms need an escape window. [1]",
        show_all_sources=False,
    )
    pipeline = RetrievalPipeline(object(), retriever, generator)

    result = pipeline.answer("What is required?")

    assert [source["n"] for source in result["sources"]] == [1]
    assert generator.calls[0]["show_all_sources"] is False


def test_answer_override_takes_precedence_over_generator_config():
    retriever = _FakeRetriever(_chunks())
    generator = _FakeGenerator(
        "Habitable rooms need an escape window. [1]",
        show_all_sources=True,
    )
    pipeline = RetrievalPipeline(object(), retriever, generator)

    result = pipeline.answer("What is required?", show_all_sources=False)

    assert [source["n"] for source in result["sources"]] == [1]
    assert generator.calls[0]["show_all_sources"] is False
