import chromadb

from rag_assistant.chunker import Chunk
from rag_assistant.vector_store import (
    METADATA_DIMENSION,
    METADATA_MODEL,
    ModelMismatchError,
    VectorStore,
)


def _chunk(chunk_id: str = "hsg141_0001", text: str = "Electricity can kill.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_doc="hsg141.pdf",
        title="Electrical safety on construction sites",
        source_url="",
        doc_type="procedural_guidance",
        chunk_type="procedural_guidance",
        section="Introduction",
        page_number=5,
        text=text,
        context_prefix="HSG141 — Introduction — paragraph 1",
    )


def _store() -> VectorStore:
    return VectorStore(persist_directory="unused", client=chromadb.Client())


def test_add_rebuilds_the_collection_and_stores_model_identity():
    store = _store()
    first = _chunk("a", "first")
    store.add([first], [[1.0, 0.0, 0.0, 0.0]], model_name="model-a", dimension=4)
    store.add(
        [_chunk("b", "second")],
        [[0.0, 1.0, 0.0, 0.0]],
        model_name="BAAI/bge-small-en-v1.5",
        dimension=4,
    )
    collection = store.client.get_collection(store.collection_name)
    assert collection.count() == 1
    assert collection.metadata[METADATA_MODEL] == "BAAI/bge-small-en-v1.5"
    assert int(collection.metadata[METADATA_DIMENSION]) == 4
    hits = store.query([0.0, 1.0, 0.0, 0.0], k=1)
    assert hits[0]["chunk_id"] == "b"
    assert hits[0]["text"] == "second"


def test_query_rejects_a_different_model():
    store = _store()
    store.add(
        [_chunk()],
        [[1.0, 0.0, 0.0, 0.0]],
        model_name="BAAI/bge-small-en-v1.5",
        dimension=4,
    )
    try:
        store.query(
            [1.0, 0.0, 0.0, 0.0],
            k=1,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    except ModelMismatchError as error:
        assert "BAAI/bge-small-en-v1.5" in str(error)
    else:
        raise AssertionError("expected ModelMismatchError")


def test_add_requires_matching_lengths():
    store = _store()
    try:
        store.add([_chunk()], [], model_name="m", dimension=4)
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected ValueError")
