from types import SimpleNamespace

from rag_assistant.config import GenerationConfig
from rag_assistant.generator import (
    MISSING_CONTEXT,
    SYSTEM_PROMPT,
    Generator,
    _format_passages,
    _format_source_list,
)


def _config() -> GenerationConfig:
    return GenerationConfig(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Barriers must be at least 950 mm high. [1]")
                )
            ]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_generate_sends_question_and_passages_to_groq():
    fake = _FakeClient()
    generator = Generator(_config(), client=fake)
    answer = generator.generate(
        "How high should scaffold barriers be?",
        [
            {
                "text": "Barriers other than guard rails can be used if they are at least 950 mm high.",
                "title": "Health and safety in construction (HSG150)",
                "section": "Section 3: Construction-phase health and safety",
                "paragraph_numbers": "150-151",
                "page_number": 29,
            }
        ],
    )
    assert "950 mm" in answer
    assert "Sources" in answer
    assert "[1] Health and safety in construction (HSG150)" in answer
    sent = fake.chat.completions.last_kwargs
    assert sent["model"] == "llama-3.1-8b-instant"
    assert sent["temperature"] == 0.1
    assert sent["messages"][0]["content"] == SYSTEM_PROMPT
    user = sent["messages"][1]["content"]
    assert "950 mm" in user
    assert "How high should scaffold barriers be?" in user
    assert "[1]" in user
    assert "HSG150" in user
    assert "only the passages above" in user
    assert "end of each sentence" in user
    assert "Do not write document names or a source list" in user


def test_prompt_requires_context_only_answers_and_per_claim_citations():
    assert "Answer ONLY from the numbered passages" in SYSTEM_PROMPT
    assert "Every claim must cite the passage" in SYSTEM_PROMPT
    assert "Do not use prior knowledge" in SYSTEM_PROMPT
    collapsed = " ".join(SYSTEM_PROMPT.split())
    assert "IEEE-style" in collapsed
    assert "Do not collect markers into a list at the end" in collapsed
    assert "Do not write a Sources section" in collapsed


def test_generate_uses_display_text_not_a_context_prefix_field():
    fake = _FakeClient()
    generator = Generator(_config(), client=fake)
    generator.generate(
        "What does clause 1.5 require?",
        [
            {
                "text": "Have level treads. The rise and going must be consistent.",
                "context_prefix": "Approved Document K — clause 1.5",
                "title": "The Merged Approved Documents",
                "part_code": "K",
                "clause_number": "1.5",
            }
        ],
    )
    user = fake.chat.completions.last_kwargs["messages"][1]["content"]
    assert "Approved Document K — clause 1.5" in user
    assert "Have level treads" in user
    assert user.index("Approved Document K") < user.index("Have level treads")


def test_generate_does_not_append_sources_when_the_model_returns_nothing():
    fake = _FakeClient()

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )

    fake.chat.completions.create = create
    generator = Generator(_config(), client=fake)
    answer = generator.generate(
        "Anything?",
        [{"text": "Notify HSE before work lasting more than 30 days.", "title": "HSG150"}],
    )
    assert answer == ""


def test_generate_without_passages_does_not_call_the_model():
    fake = _FakeClient()
    generator = Generator(_config(), client=fake)
    assert generator.generate("Anything?", []) == MISSING_CONTEXT
    assert fake.chat.completions.last_kwargs is None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    generator = Generator(_config())
    try:
        _ = generator.client
    except RuntimeError as error:
        assert "GROQ_API_KEY" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_format_passages_numbers_and_labels():
    text = _format_passages(
        [
            {
                "text": "Notify HSE before work lasting more than 30 days.",
                "title": "HSG150",
                "paragraph_numbers": "34-36",
                "page_number": 9,
            }
        ]
    )
    assert text.startswith("[1] HSG150 — paragraphs 34-36 — p.9")
    assert "Passage text:" in text
    assert "30 days" in text


def test_mad_passage_label_uses_approved_document_prefix_not_merged_title():
    text = _format_passages(
        [
            {
                "text": "All habitable rooms should have an opening onto a hall or an escape window.",
                "title": "The Merged Approved Documents",
                "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
                "part_code": "B",
                "section": "Section 2: Means of escape – dwellinghouses",
                "clause_number": "2.1",
                "context_prefix": (
                    "Approved Document B: Fire safety — "
                    "Section 2: Means of escape – dwellinghouses — clause 2.1"
                ),
                "page_number": 95,
            }
        ]
    )
    heading = text.split("\n", 1)[0]
    assert heading.startswith("[1] Approved Document B: Fire safety")
    assert "clause 2.1" in heading
    assert "The Merged Approved Documents" not in heading
    assert "p.95" not in heading
    assert "Passage text:" in text
    sources = _format_source_list(
        [
            {
                "text": "All habitable rooms should have an opening onto a hall or an escape window.",
                "title": "The Merged Approved Documents",
                "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
                "context_prefix": (
                    "Approved Document B: Fire safety — "
                    "Section 2: Means of escape – dwellinghouses — clause 2.1"
                ),
                "page_number": 95,
            }
        ]
    )
    assert sources == (
        "[1] Approved Document B: Fire safety — "
        "Section 2: Means of escape – dwellinghouses — clause 2.1"
    )


def _two_passages() -> list[dict]:
    return [
        {
            "text": "Have an opening onto a hall or an escape window.",
            "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
            "context_prefix": "Approved Document B — clause 2.1",
            "page_number": 95,
        },
        {
            "text": "Flats have a similar provision.",
            "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
            "context_prefix": "Approved Document B — clause 3.15",
            "page_number": 105,
        },
    ]


def test_source_list_includes_unused_passages_when_show_all_sources():
    sources = _format_source_list(
        _two_passages(),
        answer="Habitable rooms need an escape window. [1]",
        show_all_sources=True,
    )
    assert "[1] Approved Document B — clause 2.1" in sources
    assert "[2] Approved Document B — clause 3.15" in sources
    assert "p.95" not in sources
    assert "p.105" not in sources


def test_source_list_keeps_only_cited_passages_in_demo_mode():
    sources = _format_source_list(
        _two_passages(),
        answer="Habitable rooms need an escape window. [1]",
        show_all_sources=False,
    )
    assert sources == "[1] Approved Document B — clause 2.1"


def test_generate_honours_show_all_sources_override():
    fake = _FakeClient()
    generator = Generator(_config(), client=fake)
    chunks = _two_passages()
    all_sources = generator.generate("What is required?", chunks, show_all_sources=True)
    cited_only = generator.generate("What is required?", chunks, show_all_sources=False)
    assert "[2] Approved Document B — clause 3.15" in all_sources
    assert "[2] Approved Document B — clause 3.15" not in cited_only
    assert "[1] Approved Document B — clause 2.1" in cited_only
    assert "p.95" not in all_sources
