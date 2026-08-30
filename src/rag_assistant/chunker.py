"""
Chunking: cleaned document JSON -> chunks.jsonl.

Design decisions (see Notion "Corpus & Chunking Design Notes"):
  - Approved Documents        -> split at their own clause numbers ("3.24",
    (regulatory)                 and Part A's older "2D1"), one Part at a
                                 time, since clause numbers restart in every
                                 Part. Short clauses are merged forward,
                                 long ones windowed, and every chunk carries
                                 a Part/section/clause context prefix.
  - HSG141 (procedural)       -> split on numbered paragraphs (1..87);
                                 case study boxes are pulled out into their
                                 own chunk, tagged chunk_type="case_study",
                                 linked back to the parent paragraph number.
  - HSG150 (broad reference)  -> split on numbered paragraphs (like HSG141),
                                 with section/topic forward-filled from the
                                 ToC and subheadings attached as metadata
                                 (not as chunk boundaries). Short paragraphs
                                 merge within the same topic/subtopic; every
                                 chunk carries a context prefix.

A heading sits above the paragraph it introduces, so splitting on numbers
alone leaves it stranded at the tail of the *preceding* chunk. All three
strategies therefore run headings.trim_trailing_headings over each unit,
passing the titles that documents.known_headings_for finds for that source
file. Those catch headings the shape heuristic misses -- notably section
titles that lose the word "Section" in extraction and wrap across two
lines.

Per-document structure (part ranges, contents-page titles, document-
specific regexes) lives in one module per source: hsg141.py, hsg150.py and
approved_documents.py. This module holds only the strategies.

Every chunk carries the metadata needed to cite it: chunk_id, source_doc,
title, source_url, doc_type, chunk_type, section, page_number, and text.

Page numbers are preserved by recording where each page starts in the
flattened document text, then mapping each chunk's offset back to a page.
Structural units such as numbered paragraphs routinely span a page break,
so they cannot simply be chunked one page at a time.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.approved_documents import map_document
from rag_assistant.documents import known_headings_for
from rag_assistant.headings import trim_trailing_headings
from rag_assistant.hsg141 import CASE_STUDY_START_RE, section_for_paragraph
from rag_assistant.hsg150 import label_paragraphs, peel_trailing_subheading
from rag_assistant.watermark import first_substantive_line

# --- tunables -----------------------------------------------------------

# Sizes are measured in WORDS, matching the chunking section of config.yaml.
DEFAULT_CHUNK_SIZE_WORDS = 400
DEFAULT_CHUNK_OVERLAP_WORDS = 60

# Matches the start of a numbered paragraph, e.g. "29 Making electrical
# services safe..." -- a line beginning with 1-3 digits, then whitespace,
# then a capital letter. Anchored so it only matches at a line start.
NUMBERED_PARAGRAPH_RE = re.compile(r"(?m)^(\d{1,3})\s+(?=[A-Z])")

# The regex alone also matches ratings and voltages that begin a line,
# e.g. "110 V AC distribution points" or "13 A socket outlets". Requiring
# the numbering to increase filters those out: a real paragraph number is
# always greater than the previous one, and never jumps far ahead of it.
MAX_PARAGRAPH_NUMBER_GAP = 5
# HSG150 runs to ~681 paragraphs; tolerate slightly larger gaps when a
# single number is missed by the regex so the sequence does not stall.
HSG150_PARAGRAPH_NUMBER_GAP = 20

MIN_PARAGRAPH_WORDS = 30
PARAGRAPH_MERGE_TARGET_WORDS = 80

# --- Approved Documents (doc_type "regulatory") -------------------------

# Modern clause numbering, e.g. "3.24 The flights and landings...".
CLAUSE_MODERN_RE = re.compile(r"(?m)^(\d{1,2}\.\d{1,3})\s+(?=[A-Z0-9\u201c])")
# Older numbering used by Part A, e.g. "2D1 Where a chimney is not...".
CLAUSE_OLD_RE = re.compile(r"(?m)^(\d{1,2}[A-Z]\d{1,3})\s+(?=[A-Z0-9])")

# A clause this short says nothing on its own ("The stair should comply
# with Table 3.2"), so it is merged forward until it reaches the target.
MIN_CLAUSE_WORDS = 30
MERGE_TARGET_WORDS = 50

# Beyond this, a "clause" is a detection failure rather than a real
# provision: numbering stopped and the splitter ran on to the next match.
RUNAWAY_CLAUSE_WORDS = 2500

# Diagram and table captions that name the clauses they illustrate, e.g.
# "Diagram 11 Insulated external walls: see paragraphs 5.10, 5.13 and 5.17".
CAPTION_REFERENCE_RE = re.compile(
    r"see\s+(?:paragraphs?|clauses?)\s+((?:\d{1,2}\.\d{1,3}[,\s and]*)+)", re.I
)
CLAUSE_NUMBER_RE = re.compile(r"\d{1,2}\.\d{1,3}")
MAX_CAPTION_WORDS = 120

# Appendices sit in the middle of several Parts, not only at the end, and
# they carry their own "B1", "B2" numbering plus long tables. They have to
# break the clause run or they are absorbed into the preceding clause.
APPENDIX_PAGE_RE = re.compile(r"^\s*Appendix\s+[A-Z0-9]\b")

# Regulation and Requirement blocks quote the Building Regulations verbatim.
# They use "38. (1)…" numbering, not "16.15", so they must break the clause
# run the same way appendices do or they get glued to the preceding clause.
REGULATION_HEADING_RE = re.compile(r"^\s*Regulation\s+\d+\s*:", re.I)
REQUIREMENT_HEADING_RE = re.compile(r"^\s*Requirement\s+[A-S]\d", re.I)
REGULATION_BLOCK_RE = re.compile(
    r"(?m)^(?:Regulation\s+\d+\s*:|Requirement\s+[A-S]\d)", re.I
)


@dataclass
class Chunk:
    chunk_id: str
    source_doc: str
    title: str
    source_url: str
    doc_type: str
    # "regulatory_clause" | "reference_material" | "procedural_guidance"
    # | "case_study" | "topic_guidance" | "generic"
    chunk_type: str
    section: str
    page_number: int
    text: str  # display text: never includes the context prefix
    part_code: str = ""
    clause_number: str = ""
    context_prefix: str = ""
    topic: str = ""
    subtopic: str = ""
    paragraph_numbers: str = ""

    @property
    def embedding_text(self) -> str:
        """
        What gets embedded and indexed.

        The breadcrumb is prepended here rather than being baked into
        `text`, so a short clause is still interpretable and retrievable
        on its own while the trail never leaks into the displayed answer.
        """
        if not self.context_prefix:
            return self.text
        return f"{self.context_prefix} \u2014 {self.text}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "title": self.title,
            "source_url": self.source_url,
            "doc_type": self.doc_type,
            "chunk_type": self.chunk_type,
            "part_code": self.part_code,
            "section": self.section,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "clause_number": self.clause_number,
            "paragraph_numbers": self.paragraph_numbers,
            "context_prefix": self.context_prefix,
            "page_number": self.page_number,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            source_doc=data["source_doc"],
            title=data.get("title", ""),
            source_url=data.get("source_url", ""),
            doc_type=data.get("doc_type", ""),
            chunk_type=data.get("chunk_type", ""),
            section=data.get("section", ""),
            page_number=int(data.get("page_number", 0)),
            text=data["text"],
            part_code=data.get("part_code", ""),
            clause_number=data.get("clause_number", ""),
            context_prefix=data.get("context_prefix", ""),
            topic=data.get("topic", ""),
            subtopic=data.get("subtopic", ""),
            paragraph_numbers=data.get("paragraph_numbers", ""),
        )


@dataclass
class _Document:
    """A cleaned document flattened for chunking, with page offsets kept."""

    filename: str
    title: str
    source_url: str
    doc_type: str
    text: str
    page_starts: list[int]
    page_numbers: list[int]

    def page_for_offset(self, offset: int) -> int:
        if not self.page_starts:
            return 0
        index = bisect.bisect_right(self.page_starts, offset) - 1
        return self.page_numbers[max(index, 0)]

    def subset(self, pages: list[tuple[int, str]]) -> _Document:
        """A view over some of this document's pages, keeping its metadata."""
        return _flatten_pages(
            pages, self.filename, self.title, self.source_url, self.doc_type
        )


def _flatten_pages(
    pages: list[tuple[int, str]],
    filename: str,
    title: str,
    source_url: str,
    doc_type: str,
) -> _Document:
    parts: list[str] = []
    page_starts: list[int] = []
    page_numbers: list[int] = []
    offset = 0

    for page_number, text in pages:
        if not text:
            continue
        page_starts.append(offset)
        page_numbers.append(page_number)
        parts.append(text)
        offset += len(text) + 2  # the "\n\n" join below

    return _Document(
        filename=filename,
        title=title,
        source_url=source_url,
        doc_type=doc_type,
        text="\n\n".join(parts),
        page_starts=page_starts,
        page_numbers=page_numbers,
    )


def flatten_document(document: dict) -> _Document:
    """Join a cleaned document's pages, recording where each page starts."""
    pages = [
        (page["page_number"], page.get("text") or "")
        for page in document.get("pages", [])
    ]
    return _flatten_pages(
        pages,
        document["filename"],
        document.get("title", ""),
        document.get("source_url", ""),
        document.get("doc_type", "generic"),
    )


def find_numbered_paragraphs(
    text: str, max_gap: int = MAX_PARAGRAPH_NUMBER_GAP
) -> list[tuple[int, int]]:
    """
    Return (paragraph_number, start_offset) for each numbered paragraph.

    Candidates whose number doesn't continue the sequence are rejected, so
    lines like "110 V AC distribution points" aren't mistaken for the start
    of paragraph 110.
    """
    accepted: list[tuple[int, int]] = []
    last_number: int | None = None
    for match in NUMBERED_PARAGRAPH_RE.finditer(text):
        number = int(match.group(1))
        if last_number is not None and (
            number <= last_number or number > last_number + max_gap
        ):
            continue
        # The first match anchors the sequence: a document may be chunked
        # from a page range that doesn't start at paragraph 1.
        accepted.append((number, match.start()))
        last_number = number
    return accepted


class DocumentChunker:
    """
    Turns a cleaned document (the JSON produced by ingest.py) into a list
    of Chunk objects, using a different strategy per doc_type.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._counter = 0  # used to generate readable, ordered chunk_ids

    def chunk(self, document: dict) -> list[Chunk]:
        """Dispatch to the right strategy based on the document's doc_type."""
        doc = flatten_document(document)
        if not doc.text:
            return []

        if doc.doc_type == "procedural_guidance":
            return self._chunk_numbered_paragraphs(doc)
        if doc.doc_type == "regulatory":
            return self._chunk_regulatory(document, doc)
        if doc.doc_type == "broad_reference":
            return self._chunk_broad_reference(doc)

        return self._chunk_generic(doc)

    # -- strategy: HSG150 paragraph-first with topic metadata ------------

    def _chunk_broad_reference(self, doc: _Document) -> list[Chunk]:
        """
        Split HSG150 on numbered paragraphs; label with section/topic/subtopic.

        Subheadings are metadata on the paragraph they precede, not chunk
        boundaries. Short paragraphs merge only while they share the same
        topic and subtopic.
        """
        paragraphs = find_numbered_paragraphs(
            doc.text, max_gap=HSG150_PARAGRAPH_NUMBER_GAP
        )
        if not paragraphs:
            return self._chunk_generic(doc)

        labels = label_paragraphs(doc.text, paragraphs)

        # Build (numbers, text, offset, section, topic, subtopic) units
        # before merging.
        units: list[tuple[list[int], str, int, str, str, str]] = []
        for index, (para_num, start) in enumerate(paragraphs):
            end = paragraphs[index + 1][1] if index + 1 < len(paragraphs) else len(doc.text)
            raw = doc.text[start:end].strip()
            body, _trailing = peel_trailing_subheading(raw)
            body = trim_trailing_headings(body, known_headings_for(doc.filename))
            if not body:
                continue
            label = labels[index]
            units.append(
                (
                    [para_num],
                    body,
                    start,
                    label.section,
                    label.topic,
                    label.subtopic,
                )
            )

        chunks: list[Chunk] = []
        index = 0
        while index < len(units):
            numbers, text, offset, section, topic, subtopic = units[index]
            words = len(text.split())

            while words < PARAGRAPH_MERGE_TARGET_WORDS and index + 1 < len(units):
                next_numbers, next_text, _, next_section, next_topic, next_sub = units[
                    index + 1
                ]
                if (
                    next_topic != topic
                    or next_sub != subtopic
                    or next_section != section
                ):
                    break
                index += 1
                text = f"{text}\n{next_text}"
                words = len(text.split())
                numbers.extend(next_numbers)

            para_label = self._format_paragraph_range(numbers)
            prefix = self._hsg150_context_prefix(section, topic, subtopic, para_label)
            chunks.extend(
                self._windows(
                    doc,
                    text,
                    "topic_guidance",
                    section,
                    offset,
                    force=False,
                    context_prefix=prefix,
                    topic=topic,
                    subtopic=subtopic,
                    paragraph_numbers=para_label,
                )
            )
            index += 1
        return chunks

    def _format_paragraph_range(self, numbers: list[int]) -> str:
        if not numbers:
            return ""
        if len(numbers) == 1:
            return str(numbers[0])
        return f"{numbers[0]}-{numbers[-1]}"

    def _hsg150_context_prefix(
        self, section: str, topic: str, subtopic: str, paragraph_numbers: str
    ) -> str:
        parts = ["HSG150: Health and safety in construction"]
        if section:
            parts.append(section)
        if topic:
            parts.append(topic)
        if subtopic:
            parts.append(subtopic)
        if paragraph_numbers:
            label = "paragraphs" if "-" in paragraph_numbers else "paragraph"
            parts.append(f"{label} {paragraph_numbers}")
        return " \u2014 ".join(parts)

    # -- strategy: Approved Documents clauses ----------------------------

    def _chunk_regulatory(self, document: dict, doc: _Document) -> list[Chunk]:
        """
        Split the Approved Documents at their own clause numbers.

        Work proceeds one Part at a time. Clause numbers restart in each
        Part, so a Part is the widest span in which a clause number means
        one thing, and it is therefore also the boundary that merging and
        context prefixing must never cross.
        """
        pages = [
            (page["page_number"], page.get("text") or "")
            for page in document.get("pages", [])
            if page.get("text")
        ]
        structure = map_document(pages)

        dropped = [n for n, _ in pages if structure[n].is_index]
        for page_number in dropped:
            print(f"  dropped page {page_number}: contents/index, no clause content")
        if dropped:
            print(f"  dropped {len(dropped)} contents/index pages in total")

        kept = [(n, t) for n, t in pages if not structure[n].is_index]

        chunks: list[Chunk] = []
        for part_code, part_pages in self._group_by_part(kept, structure):
            for block in self._split_at_special_blocks(part_pages):
                chunks.extend(self._chunk_block(doc, structure, part_code, block))
        return chunks

    def _starts_special_block(self, text: str) -> bool:
        head = first_substantive_line(text)
        if not head:
            return False
        if APPENDIX_PAGE_RE.match(head):
            return True
        return bool(
            REGULATION_HEADING_RE.match(head) or REQUIREMENT_HEADING_RE.match(head)
        )

    def _split_at_special_blocks(
        self, pages: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        blocks: list[list[tuple[int, str]]] = []
        for page_number, text in pages:
            if not blocks or self._starts_special_block(text):
                blocks.append([])
            blocks[-1].append((page_number, text))
        return blocks

    def _group_by_part(
        self, pages: list[tuple[int, str]], structure: dict
    ) -> list[tuple[str, list[tuple[int, str]]]]:
        grouped: list[tuple[str, list[tuple[int, str]]]] = []
        for page_number, text in pages:
            part_code = structure[page_number].part_code
            if not grouped or grouped[-1][0] != part_code:
                grouped.append((part_code, []))
            grouped[-1][1].append((page_number, text))
        return grouped

    def _chunk_block(
        self,
        doc: _Document,
        structure: dict,
        part_code: str,
        pages: list[tuple[int, str]],
    ) -> list[Chunk]:
        numbered = [
            index
            for index, (_, text) in enumerate(pages)
            if CLAUSE_MODERN_RE.search(text) or CLAUSE_OLD_RE.search(text)
        ]
        if not numbered:
            return self._chunk_reference_pages(doc, structure, part_code, pages)

        first, last = numbered[0], numbered[-1]
        chunks = self._chunk_reference_pages(doc, structure, part_code, pages[:first])
        for segment_pages, is_regulation in self._split_pages_at_regulations(
            pages[first : last + 1]
        ):
            if is_regulation:
                chunks.extend(
                    self._chunk_reference_pages(
                        doc, structure, part_code, segment_pages
                    )
                )
            else:
                chunks.extend(
                    self._chunk_clauses(doc, structure, part_code, segment_pages)
                )
        trailing = self._chunk_reference_pages(
            doc, structure, part_code, pages[last + 1 :], chunks
        )
        return chunks + trailing

    def _split_pages_at_regulations(
        self, pages: list[tuple[int, str]]
    ) -> list[tuple[list[tuple[int, str]], bool]]:
        """Split a page run where Regulation/Requirement blocks begin."""
        segments: list[tuple[list[tuple[int, str]], bool]] = []
        for page_number, text in pages:
            is_regulation = self._starts_special_block(text)
            if not segments or segments[-1][1] != is_regulation:
                segments.append(([], is_regulation))
            segments[-1][0].append((page_number, text))
        return segments

    def _chunk_clauses(
        self,
        doc: _Document,
        structure: dict,
        part_code: str,
        pages: list[tuple[int, str]],
    ) -> list[Chunk]:
        run = doc.subset(pages)
        starts = self._clause_starts(run.text)
        if not starts:
            return self._chunk_generic(run)

        # (clause_number, text, offset) before merging and windowing.
        clauses: list[tuple[str, str, int]] = []
        if starts[0][1] > 0:
            # Text before the first clause number continues a clause that
            # began on an earlier Part page run.
            lead = run.text[: starts[0][1]].strip()
            if lead:
                clauses.append(("", lead, 0))
        for i, (number, start) in enumerate(starts):
            end = starts[i + 1][1] if i + 1 < len(starts) else len(run.text)
            body = trim_trailing_headings(
                run.text[start:end].strip(), known_headings_for(run.filename)
            )
            if body:
                clauses.append((number, body, start))

        return self._emit_clause_chunks(run, structure, part_code, clauses)

    def _clause_starts(self, text: str) -> list[tuple[str, int]]:
        """
        Locate clause numbers, rejecting matches that break the sequence.

        Numbering ascends within a Part and restarts at each section
        (1.1, 1.2, ... 2.1), so a candidate is accepted when it advances
        the sequence. Any ".1" is also accepted wherever it appears,
        because numbering restarts outright partway through some Parts --
        Part B's two volumes both begin at 1.1 -- and refusing to go
        backwards there would swallow the whole second volume into one
        chunk.

        Two kinds of impostor are rejected. A mid-section number that
        repeats or goes backwards is how table values appear. A number
        that leaps several sections ahead is how diagram dimensions
        appear: "7.5m max." on a diagram of flats leaves "7.5" starting a
        line, and accepting it beside clause 3.27 would lock out every
        real clause behind it until the numbering caught up.
        """
        found = [
            (match.group(1), match.start(), True)
            for match in CLAUSE_MODERN_RE.finditer(text)
        ] + [
            (match.group(1), match.start(), False)
            for match in CLAUSE_OLD_RE.finditer(text)
        ]
        found.sort(key=lambda item: item[1])

        accepted: list[tuple[str, int]] = []
        last: tuple[int, int] | None = None
        for number, start, is_modern in found:
            if not is_modern:
                accepted.append((number, start))
                continue
            section_str, item_str = number.split(".")
            key = (int(section_str), int(item_str))
            if last is not None:
                goes_backwards = key <= last and key[1] != 1
                jumps_ahead = key[0] > last[0] + 1 and key[1] != 1
                if goes_backwards or jumps_ahead:
                    continue
            accepted.append((number, start))
            last = key
        return accepted

    def _emit_clause_chunks(
        self,
        run: _Document,
        structure: dict,
        part_code: str,
        clauses: list[tuple[str, str, int]],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        while index < len(clauses):
            number, text, offset = clauses[index]
            words = len(text.split())
            page_number = run.page_for_offset(offset)
            section = structure[page_number].section_title

            # Size floor: absorb following clauses until the chunk says
            # enough to stand alone, but never across a section boundary.
            numbers = [number] if number else []
            while (
                words < MIN_CLAUSE_WORDS
                or (numbers[1:] and words < MERGE_TARGET_WORDS)
            ) and index + 1 < len(clauses):
                next_number, next_text, next_offset = clauses[index + 1]
                next_section = structure[run.page_for_offset(next_offset)].section_title
                if section and next_section and next_section != section:
                    break
                index += 1
                text = f"{text}\n{next_text}"
                words = len(text.split())
                if next_number:
                    numbers.append(next_number)

            clause_number = self._format_clause_range(numbers)
            prefix = self._context_prefix(structure[page_number], clause_number)

            if words > RUNAWAY_CLAUSE_WORDS:
                print(
                    f"  warning: clause {clause_number or '?'} in Part {part_code} "
                    f"is {words} words (page {page_number}); clause numbering "
                    "was probably not detected -- falling back to word windows"
                )
                chunks.extend(
                    self._windows(
                        run, text, "reference_material", section, offset,
                        part_code=part_code, clause_number="", context_prefix="",
                    )
                )
            else:
                window_chunks = self._windows(
                    run, text, "regulatory_clause", section, offset,
                    part_code=part_code,
                    clause_number=clause_number,
                    context_prefix=prefix,
                    force=False,
                )
                if len(window_chunks) > 1:
                    for extra in window_chunks[1:]:
                        extra.chunk_type = "reference_material"
                        extra.clause_number = ""
                        extra.context_prefix = ""
                chunks.extend(window_chunks)
            index += 1
        return chunks

    def _format_clause_range(self, numbers: list[str]) -> str:
        if not numbers:
            return ""
        if len(numbers) == 1:
            return numbers[0]
        return f"{numbers[0]}-{numbers[-1]}"

    def _context_prefix(self, page_structure, clause_number: str) -> str:
        parts = [f"Approved Document {page_structure.part_code}"]
        if page_structure.part_title:
            parts[0] += f": {page_structure.part_title}"
        if page_structure.section_title:
            parts.append(page_structure.section_title)
        if clause_number:
            label = "clauses" if "-" in clause_number else "clause"
            parts.append(f"{label} {clause_number}")
        return " \u2014 ".join(parts)

    def _chunk_reference_pages(
        self,
        doc: _Document,
        structure: dict,
        part_code: str,
        pages: list[tuple[int, str]],
        clause_chunks: list[Chunk] | None = None,
    ) -> list[Chunk]:
        """
        Chunk appendices, glossaries and standards lists.

        Captions that name the clauses they illustrate are appended to
        those clauses instead of becoming orphan chunks, since a caption
        alone ("Diagram 11 Insulated external walls") carries almost no
        retrievable meaning.
        """
        if not pages:
            return []

        remaining: list[tuple[int, str]] = []
        for page_number, text in pages:
            referenced = self._caption_targets(text)
            attached = False
            # Only genuine captions get attached. A long page that happens
            # to cite a clause is content in its own right, and appending
            # it would blow the size of the clause chunk it lands on.
            if len(text.split()) > MAX_CAPTION_WORDS:
                referenced = set()
            if referenced and clause_chunks:
                for chunk in clause_chunks:
                    if chunk.clause_number in referenced:
                        chunk.text = f"{chunk.text}\n{text.strip()}"
                        attached = True
                        break
            if not attached:
                remaining.append((page_number, text))

        if not remaining:
            return []

        run = doc.subset(remaining)
        section = structure[remaining[0][0]].section_title
        prefix = self._context_prefix(structure[remaining[0][0]], "")
        return self._windows(
            run,
            run.text,
            "reference_material",
            section,
            0,
            part_code=part_code,
            context_prefix=prefix,
            force=True,
        )

    def _caption_targets(self, text: str) -> set[str]:
        targets: set[str] = set()
        for match in CAPTION_REFERENCE_RE.finditer(text):
            targets.update(CLAUSE_NUMBER_RE.findall(match.group(1)))
        return targets

    # -- strategy: HSG141-style numbered paragraphs + case studies -------

    def _chunk_numbered_paragraphs(self, doc: _Document) -> list[Chunk]:
        paragraphs = find_numbered_paragraphs(doc.text)
        if not paragraphs:
            # Structure wasn't detected (e.g. cleaning stripped the
            # numbering) -- fall back rather than silently producing zero
            # chunks for a whole document.
            return self._chunk_generic(doc)

        chunks: list[Chunk] = []
        for i, (para_num, start) in enumerate(paragraphs):
            end = paragraphs[i + 1][1] if i + 1 < len(paragraphs) else len(doc.text)
            para_text = doc.text[start:end].strip()
            section = section_for_paragraph(para_num)

            for segment_text, offset, is_case_study in self._split_case_studies(
                para_text, start
            ):
                segment_text = trim_trailing_headings(
                    segment_text, known_headings_for(doc.filename)
                )
                if not segment_text:
                    continue
                if is_case_study:
                    chunks.append(
                        self._make_chunk(
                            doc,
                            segment_text,
                            "case_study",
                            f"{section} (para {para_num}, case study)",
                            offset,
                        )
                    )
                else:
                    chunks.extend(
                        self._split_if_too_long(
                            doc,
                            segment_text,
                            "procedural_guidance",
                            f"{section} (para {para_num})",
                            offset,
                        )
                    )

        return chunks

    def _split_case_studies(
        self, para_text: str, para_offset: int
    ) -> list[tuple[str, int, bool]]:
        """
        Split a paragraph into prose and case study segments.

        Each box is assumed to run to the next box or to the end of the
        paragraph. Where body text resumes after a box it stays attached to
        that box, which is the trade-off for not having a reliable marker
        for where these boxes end.
        """
        starts = [m.start() for m in CASE_STUDY_START_RE.finditer(para_text)]
        if not starts:
            return [(para_text, para_offset, False)]

        segments: list[tuple[str, int, bool]] = []
        prose = para_text[: starts[0]].strip()
        if prose:
            segments.append((prose, para_offset, False))

        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(para_text)
            case_text = para_text[start:end].strip()
            if case_text:
                segments.append((case_text, para_offset + start, True))
        return segments

    # -- strategy: generic word-window with overlap -----------------------

    def _chunk_generic(self, doc: _Document) -> list[Chunk]:
        return self._windows(doc, doc.text, "generic", "N/A", 0, force=True)

    def _split_if_too_long(
        self,
        doc: _Document,
        text: str,
        chunk_type: str,
        section: str,
        offset: int,
        force: bool = False,
    ) -> list[Chunk]:
        return self._windows(doc, text, chunk_type, section, offset, force=force)

    def _windows(
        self,
        doc: _Document,
        text: str,
        chunk_type: str,
        section: str,
        offset: int,
        force: bool = False,
        part_code: str = "",
        clause_number: str = "",
        context_prefix: str = "",
        topic: str = "",
        subtopic: str = "",
        paragraph_numbers: str = "",
    ) -> list[Chunk]:
        """
        Sub-split text into overlapping word windows.

        A structural unit shorter than one window is kept whole unless
        `force` is set, so a clause or paragraph stays intact wherever it
        fits. The context prefix is carried onto every window, so a
        fragment of a long clause keeps its citation trail.
        """
        words = text.split()
        if not words:
            return []
        if not force and len(words) <= self.chunk_size:
            return [
                self._make_chunk(
                    doc, text, chunk_type, section, offset,
                    part_code=part_code,
                    clause_number=clause_number,
                    context_prefix=context_prefix,
                    topic=topic,
                    subtopic=subtopic,
                    paragraph_numbers=paragraph_numbers,
                )
            ]

        chunks: list[Chunk] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            window_offset = offset + len(" ".join(words[:start]))
            chunks.append(
                self._make_chunk(
                    doc, " ".join(window), chunk_type, section, window_offset,
                    part_code=part_code,
                    clause_number=clause_number,
                    context_prefix=context_prefix,
                    topic=topic,
                    subtopic=subtopic,
                    paragraph_numbers=paragraph_numbers,
                )
            )
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _make_chunk(
        self,
        doc: _Document,
        text: str,
        chunk_type: str,
        section: str,
        offset: int,
        part_code: str = "",
        clause_number: str = "",
        context_prefix: str = "",
        topic: str = "",
        subtopic: str = "",
        paragraph_numbers: str = "",
    ) -> Chunk:
        self._counter += 1
        return Chunk(
            chunk_id=f"{Path(doc.filename).stem}_{self._counter:04d}",
            source_doc=doc.filename,
            title=doc.title,
            source_url=doc.source_url,
            doc_type=doc.doc_type,
            chunk_type=chunk_type,
            section=section,
            page_number=doc.page_for_offset(offset),
            text=text,
            part_code=part_code,
            clause_number=clause_number,
            context_prefix=context_prefix,
            topic=topic,
            subtopic=subtopic,
            paragraph_numbers=paragraph_numbers,
        )


def chunk_corpus(config) -> Path:
    """Chunk every cleaned document and write the result to chunks.jsonl."""
    cleaned_dir = Path(config.paths.cleaned_dir)
    out_path = Path(config.paths.chunks_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for json_path in sorted(cleaned_dir.glob("*.json")):
            document = json.loads(json_path.read_text(encoding="utf-8"))

            # A fresh chunker per document keeps chunk_id numbering
            # restarting at 0001 within each source file.
            chunker = DocumentChunker(
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap,
            )
            chunks = chunker.chunk(document)
            for chunk in chunks:
                out_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"  {document['filename']}: {len(chunks)} chunks")

    print(f"Wrote {total} chunks total to {out_path}")
    return out_path


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks.jsonl back into Chunk objects."""
    chunks: list[Chunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
