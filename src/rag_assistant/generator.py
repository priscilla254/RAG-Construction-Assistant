"""
Grounded answer generation via Groq.

The model only sees the question and the retrieved passages. Display
text is used (Chunk.text), not embedding_text, so the context prefix
does not leak into the answer. Claims are marked [n] in the sentence;
the source list is appended from passage headings so document names
cannot be invented.
"""

from __future__ import annotations

import os
import re

from rag_assistant.config import GenerationConfig
from rag_assistant.documents import is_merged_approved_document

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You are a UK construction safety and building-regulations assistant.

Answer ONLY from the numbered passages provided in the user message.
Do not use prior knowledge. If the passages do not contain the answer,
say that the retrieved documents do not cover it. Do not invent
requirements, measurements, or clause numbers.

Every claim must cite the passage it comes from with an IEEE-style
marker [n] at the end of the sentence that uses it. Use only the
passage number, not the document name or heading, in the answer body.
Do not collect markers into a list at the end. Do not write a Sources
section — that is added separately. Ungrounded statements are not
allowed.

Rules:
- Quote figures, limits, and clause numbers only if they appear in a passage.
- Prefer the most specific passage when two conflict.
- This is guidance from the source documents, not legal advice.
"""

MISSING_CONTEXT = (
    "I do not have retrieved passages for this question, so I cannot "
    "answer from the construction documents."
)


class Generator:
    def __init__(self, config: GenerationConfig, client=None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if self.config.provider != "groq":
                raise NotImplementedError(
                    f"generation.provider {self.config.provider!r} is not supported"
                )
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            from groq import Groq

            self._client = Groq(api_key=api_key)
        return self._client

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
        show_all_sources: bool | None = None,
    ) -> str:
        """Return an answer grounded in the retrieved chunks.

        `show_all_sources` overrides `config.show_all_sources` when set
        (Streamlit demo passes False; spot-checks leave the default True).
        """
        passages = _format_passages(retrieved_chunks)
        if not passages:
            return MISSING_CONTEXT

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Passages:\n\n{passages}\n\n"
                    f"Question: {question.strip()}\n\n"
                    "Answer using only the passages above. "
                    "Put [n] at the end of each sentence you rely on, "
                    "not as a group after the answer. "
                    "Do not write document names or a source list."
                ),
            },
        ]
        text = self._complete(messages)
        if not text:
            text = self._complete(messages)
        include_all = (
            self.config.show_all_sources
            if show_all_sources is None
            else show_all_sources
        )
        sources = _format_source_list(
            retrieved_chunks,
            answer=text,
            show_all_sources=include_all,
        )
        if not text or not sources:
            return text
        return f"{text}\n\nSources\n{sources}"

    def _complete(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=messages,
        )
        return (completion.choices[0].message.content or "").strip()


def _format_passages(chunks: list[dict]) -> str:
    blocks: list[str] = []
    for index, chunk in cited_passages(chunks):
        body = (chunk.get("text") or "").strip()
        blocks.append(
            f"[{index}] {passage_label(chunk)}\nPassage text:\n{body}"
        )
    return "\n\n".join(blocks)


def citation_numbers(answer: str) -> set[int]:
    """IEEE [n] markers found in the answer body."""
    return {int(number) for number in CITATION_RE.findall(answer)}


def cited_passages(
    chunks: list[dict],
    answer: str = "",
    show_all_sources: bool = True,
) -> list[tuple[int, dict]]:
    """Number non-empty retrieved chunks 1..n, optionally keeping only [n] cites."""
    numbered: list[tuple[int, dict]] = []
    index = 0
    for chunk in chunks:
        if not (chunk.get("text") or "").strip():
            continue
        index += 1
        numbered.append((index, chunk))
    if show_all_sources:
        return numbered
    cited = citation_numbers(answer)
    return [(i, chunk) for i, chunk in numbered if i in cited]


def _format_source_list(
    chunks: list[dict],
    answer: str = "",
    show_all_sources: bool = True,
) -> str:
    return "\n".join(
        f"[{index}] {passage_label(chunk)}"
        for index, chunk in cited_passages(
            chunks, answer=answer, show_all_sources=show_all_sources
        )
    )


def passage_label(chunk: dict) -> str:
    """Human-readable citation line for a retrieved chunk."""
    # The context prefix is the citation trail built at chunk time
    # ("Approved Document B: Fire safety — … clause 2.1"). Prefer it
    # over the merged-file title, which does not name the Part.
    prefix = (chunk.get("context_prefix") or "").strip()
    if prefix:
        extra = []
        if _cite_page_number(chunk):
            extra.append(f"p.{chunk['page_number']}")
        return " — ".join([prefix, *extra])

    parts: list[str] = []
    title = chunk.get("title") or chunk.get("source_doc") or ""
    if title:
        parts.append(title)
    if chunk.get("part_code"):
        parts.append(f"Part {chunk['part_code']}")
    if chunk.get("section"):
        parts.append(chunk["section"])
    if chunk.get("clause_number"):
        parts.append(f"clause {chunk['clause_number']}")
    if chunk.get("paragraph_numbers"):
        numbers = chunk["paragraph_numbers"]
        label = "paragraphs" if "-" in str(numbers) else "paragraph"
        parts.append(f"{label} {numbers}")
    if _cite_page_number(chunk):
        parts.append(f"p.{chunk['page_number']}")
    return " — ".join(parts) if parts else "source"


def _cite_page_number(chunk: dict) -> bool:
    if not chunk.get("page_number"):
        return False
    return not is_merged_approved_document(chunk)
