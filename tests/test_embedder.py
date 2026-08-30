from unittest.mock import MagicMock, patch

import numpy as np

from rag_assistant.chunker import Chunk
from rag_assistant.embedder import BGE_QUERY_PREFIX, Embedder


def _chunk(**overrides) -> Chunk:
    fields = dict(
        chunk_id="hsg150_0001",
        source_doc="hsg150.pdf",
        title="Health and safety in construction",
        source_url="",
        doc_type="broad_reference",
        chunk_type="topic_guidance",
        section="Section 2: Setting up the site",
        page_number=20,
        text="101 Make sure everybody follows the rules relevant to them.",
        context_prefix="HSG150: Health and safety in construction — Site rules",
    )
    fields.update(overrides)
    return Chunk(**fields)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        if isinstance(texts, str):
            texts = [texts]
        # Distinct, non-unit rows so the test can see normalize_embeddings
        # was requested rather than applied here.
        return np.ones((len(texts), 4), dtype=float) * np.arange(1, len(texts) + 1)[:, None]


def _embedder(model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> Embedder:
    embedder = Embedder(model_name=model_name, batch_size=batch_size, device="cpu")
    embedder._model = _FakeModel()
    return embedder


def test_embed_texts_preserves_order_and_returns_python_lists():
    vectors = _embedder().embed_texts(["one", "two"])
    assert len(vectors) == 2
    assert vectors[0] != vectors[1]
    assert all(isinstance(row, list) for row in vectors)
    assert all(isinstance(value, float) for value in vectors[0])


def test_embed_texts_requests_normalisation():
    embedder = _embedder()
    embedder.embed_texts(["one"])
    assert embedder.model.calls[0]["normalize_embeddings"] is True
    assert embedder.model.calls[0]["batch_size"] == 64


def test_embed_texts_empty_list_does_not_touch_the_model():
    embedder = _embedder()
    assert embedder.embed_texts([]) == []
    assert embedder.model.calls == []


def test_bge_query_gets_the_retrieval_prefix():
    embedder = _embedder("BAAI/bge-small-en-v1.5")
    embedder.embed_query("how high should guard rails be?")
    encoded = embedder.model.calls[0]["texts"]
    assert encoded == [
        f"{BGE_QUERY_PREFIX}how high should guard rails be?"
    ]


def test_minilm_query_is_not_prefixed():
    embedder = _embedder("sentence-transformers/all-MiniLM-L6-v2")
    embedder.embed_query("how high should guard rails be?")
    encoded = embedder.model.calls[0]["texts"]
    assert encoded == ["how high should guard rails be?"]


def test_embed_chunks_uses_embedding_text_not_display_text():
    embedder = _embedder()
    chunk = _chunk()
    embedder.embed_chunks([chunk])
    encoded = embedder.model.calls[0]["texts"]
    assert encoded == [chunk.embedding_text]
    assert chunk.text not in encoded
    assert chunk.context_prefix in encoded[0]


def test_dimension_comes_from_the_model():
    assert _embedder().dimension == 4


def test_batch_size_must_be_positive():
    try:
        Embedder(model_name="BAAI/bge-small-en-v1.5", batch_size=0)
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_sentence_transformer_is_constructed_with_resolved_device():
    fake = MagicMock()
    fake.get_sentence_embedding_dimension.return_value = 4
    fake.encode.return_value = np.ones((1, 4))
    with patch(
        "sentence_transformers.SentenceTransformer", return_value=fake
    ) as constructed:
        embedder = Embedder(
            model_name="BAAI/bge-small-en-v1.5", batch_size=32, device="cpu"
        )
        embedder.embed_texts(["hello"])
    constructed.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cpu")
