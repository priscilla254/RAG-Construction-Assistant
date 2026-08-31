"""
Structure detection for the Merged Approved Documents.

The merged file concatenates the individual Approved Documents (Regulation
7, then Parts A to S) into one PDF. Clause numbers restart in every one of
them -- clause 3.24 exists in Part B and again in Part L -- so a clause is
only citable once you know which Part its page belongs to.

Nothing in the extracted text states the Part outright on most pages. The
signals available are the running headers, and each is individually
unreliable:

  - Requirement codes ("B1", "L1(a)") are strong but appear on under half
    the pages, and cross-references to other Parts appear in headers too.
  - Bare letters ("J", "JJ") head some Parts, but stray capitals survive
    cleaning and the "ONLINE VERSION" watermark decomposes into single
    letters (O, N, L, I, E) that imitate them.
  - Document titles ("SITE PREPARATION AND RESISTANCE TO CONTAMINANTS")
    are reliable where they appear, and cover the Parts whose headers
    carry no codes at all.
  - Appendix headings ("Appendix C: Fire doorsets") look exactly like
    requirement codes but belong to whichever Part contains them, so they
    must be excluded rather than counted.

So each page casts weighted votes, and the Part assignment is then chosen
by dynamic programming under the constraint that Parts appear in
publication order and never go backwards. That constraint is what makes a
sparse, noisy signal usable: a handful of correct votes anywhere inside a
document pins down the pages around it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.publisher_matter import looks_like_publisher_matter
from rag_assistant.watermark import strip_watermark_line

# Publication order within the merged file. "7" is the Regulation 7
# materials and workmanship document that opens the file; there is no
# Part I or N.
PART_ORDER: list[str] = list("7ABCDEFGHJKLMOPQRS")

PART_TITLES: dict[str, str] = {
    "7": "Materials and workmanship",
    "A": "Structure",
    "B": "Fire safety",
    "C": "Site preparation and resistance to contaminants",
    "D": "Toxic substances",
    "E": "Resistance to the passage of sound",
    "F": "Ventilation",
    "G": "Sanitation, hot water safety and water efficiency",
    "H": "Drainage and waste disposal",
    "J": "Combustion appliances and fuel storage systems",
    "K": "Protection from falling, collision and impact",
    "L": "Conservation of fuel and power",
    "M": "Access to and use of buildings",
    "O": "Overheating",
    "P": "Electrical safety",
    "Q": "Security in dwellings",
    "R": "Physical infrastructure for electronic communications networks",
    "S": "Infrastructure for charging electric vehicles",
}

# Distinctive phrases from each document's running title. Matched against
# the top of the page only, so body text that happens to mention
# "ventilation" doesn't vote.
PART_TITLE_PHRASES: dict[str, tuple[str, ...]] = {
    "7": ("MATERIALS AND WORKMANSHIP",),
    "A": ("STRUCTURE",),
    "B": ("FIRE SAFETY",),
    "C": ("SITE PREPARATION", "RESISTANCE TO CONTAMINANTS"),
    "D": ("TOXIC SUBSTANCES", "CAVITY INSULATION"),
    "E": ("RESISTANCE TO THE PASSAGE OF SOUND",),
    "F": ("VENTILATION",),
    "G": ("SANITATION", "HOT WATER SAFETY", "WATER EFFICIENCY"),
    "H": ("DRAINAGE AND WASTE DISPOSAL", "FOUL WATER DRAINAGE", "SURFACE WATER DRAINAGE"),
    "J": ("COMBUSTION APPLIANCES", "FUEL STORAGE"),
    "K": ("PROTECTION FROM FALLING", "COLLISION AND IMPACT"),
    "L": ("CONSERVATION OF FUEL AND POWER",),
    "M": ("ACCESS TO AND USE OF BUILDINGS",),
    "O": ("OVERHEATING",),
    "P": ("ELECTRICAL SAFETY",),
    "Q": ("SECURITY IN DWELLINGS",),
    "R": ("PHYSICAL INFRASTRUCTURE", "ELECTRONIC COMMUNICATIONS"),
    "S": ("INFRASTRUCTURE FOR CHARGING ELECTRIC VEHICLES", "ELECTRIC VEHICLES"),
}

# A Part letter with a single digit: B1, A1/2, L1(a), M4(2). The single
# digit matters -- two-digit codes are regulation references (R39, R44),
# not Part identifiers.
REQUIREMENT_CODE_RE = re.compile(r"\b([A-S])\d(?:/\d)?(?:\([0-9a-z]+\))?\b")

# "Appendix C: Fire doorsets" belongs to the Part that contains it, so
# these lines are skipped before looking for requirement codes.
APPENDIX_HEADING_RE = re.compile(r"^\s*(Appendix|Table|Diagram)\s+[A-Z0-9]", re.I)

BARE_LETTER_RE = re.compile(r"^([A-S])\1?$")

SECTION_HEADING_RE = re.compile(r"^\s*(Section\s+\d+[A-Z]?\s*:\s*.+?)\s*$")

# Contents and back-of-book index lines end in a page number or a page
# range: "Hose laying distances 15.7", "Provision 15.1-15.7".
INDEX_LINE_RE = re.compile(r"(?:\b[ivxlc]+|\d+(?:\.\d+)?(?:[\u2013-]\d+(?:\.\d+)?)?)\s*$")

# Index entries point somewhere: a clause number, a range, or a named
# table, diagram or appendix. Entries also wrap ("External walls adjacent
# to 3.63-3.64, 11.10," / "Diagram 3.10"), so the pointer is not always at
# the end of the line.
INDEX_REFERENCE_RE = re.compile(
    r"\d+\.\d+|\b(?:Table|Diagram|Appendix)\s+[A-Z0-9]", re.I
)
# Index entries are short. Body prose runs to roughly 90 characters a
# line, which is what stops a clause page full of cross-references from
# being mistaken for an index and discarded.
MAX_INDEX_LINE_CHARS = 60

HEADER_LINES = 3
MAX_HEADER_WORDS = 8
TITLE_VOTE = 1.0
CODE_VOTE = 1.0
BARE_LETTER_VOTE = 0.4


@dataclass
class PageStructure:
    part_code: str
    part_title: str
    section_title: str
    is_index: bool


def page_part_votes(text: str) -> dict[str, float]:
    """Weighted evidence for which Part a page belongs to."""
    votes: dict[str, float] = {}

    def add(letter: str, weight: float) -> None:
        votes[letter] = votes.get(letter, 0.0) + weight

    header_lines = [line for line in text.split("\n") if line.strip()][:HEADER_LINES]
    header_upper = " ".join(header_lines).upper()

    for letter, phrases in PART_TITLE_PHRASES.items():
        if any(phrase in header_upper for phrase in phrases):
            add(letter, TITLE_VOTE)

    for line in header_lines:
        if APPENDIX_HEADING_RE.match(line):
            continue
        line = strip_watermark_line(line)
        # Codes only count from a genuine running header. Body text is
        # full of look-alikes -- the fire classification "A2-s3" would
        # otherwise vote for Part A on every page of Part B.
        if len(line.split()) > MAX_HEADER_WORDS:
            continue
        for match in REQUIREMENT_CODE_RE.finditer(line):
            add(match.group(1), CODE_VOTE)
        for token in line.split():
            match = BARE_LETTER_RE.match(token)
            if match:
                add(match.group(1), BARE_LETTER_VOTE)

    return votes


def assign_parts(votes: list[dict[str, float]]) -> list[str]:
    """
    Choose a Part for every page, constrained to publication order.

    Each Approved Document occupies one contiguous run of pages, so the
    assignment can never move backwards through PART_ORDER. Maximising
    total vote agreement under that constraint is a shortest-path problem,
    solved here with a linear scan carrying the best score so far.
    """
    if not votes:
        return []

    n = len(PART_ORDER)
    best = [0.0] * n
    back: list[list[int]] = []

    for index, page in enumerate(votes):
        new_best = [0.0] * n
        row = [0] * n
        running_best, running_arg = float("-inf"), 0
        for j, letter in enumerate(PART_ORDER):
            if index == 0:
                new_best[j] = page.get(letter, 0.0)
                continue
            if best[j] > running_best:
                running_best, running_arg = best[j], j
            new_best[j] = running_best + page.get(letter, 0.0)
            row[j] = running_arg
        best = new_best
        back.append(row)

    j = max(range(n), key=lambda k: best[k])
    result: list[str] = []
    for index in range(len(votes) - 1, -1, -1):
        result.append(PART_ORDER[j])
        j = back[index][j]
    return list(reversed(result))


def looks_like_index(text: str) -> bool:
    """
    True for contents pages and back-of-book index pages.

    These are lists of pointers to content rather than content, so they
    match keyword queries without ever answering them.

    An index entry is words followed by a page reference, so a line only
    counts when it has words to begin with. Without that, diagrams whose
    labels are bare numbers -- the wind speed map in Part A, for instance
    -- look exactly like an index and real content gets dropped.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 8:
        return False
    if any(word in " ".join(lines[:2]).upper() for word in ("DIAGRAM", "TABLE", "MAP")):
        return False
    if lines[0].upper() in {"CONTENTS", "INDEX"}:
        return True

    lengths = sorted(len(line) for line in lines)
    if lengths[len(lengths) // 2] > MAX_INDEX_LINE_CHARS:
        return False

    hits = 0
    for line in lines:
        if not (INDEX_LINE_RE.search(line) or INDEX_REFERENCE_RE.search(line)):
            continue
        head = INDEX_LINE_RE.sub("", line).strip()
        if len(head.split()) >= 2 and sum(c.isalpha() for c in head) >= 6:
            hits += 1
    return hits >= len(lines) * 0.6


def _join_wrapped_heading(heading: str, following: list[str], max_lines: int = 2) -> str:
    """
    Reattach a section heading that was set over more than one line.

    Headings wrap in the source layout ("Section 3: Horizontal" /
    "circulation in buildings other than dwellings"), and a truncated
    heading makes a misleading citation. A following line continues the
    heading when it starts lower case or the heading ends mid-clause.
    """
    for line in following[:max_lines]:
        line = line.strip()
        if not line:
            continue
        continues = heading.endswith(",") or (line[:1].islower() and len(line) < 80)
        if not continues:
            break
        heading = f"{heading.rstrip(',')} {line}".strip()
    return heading


def map_document(pages: list[tuple[int, str]]) -> dict[int, PageStructure]:
    """
    Build the page_number -> structure lookup used by the chunker.

    Section titles are forward-filled: a heading appears once, on the page
    where the section opens, and the pages that follow belong to it until
    the next heading or the next Part.
    """
    parts = assign_parts([page_part_votes(text) for _, text in pages])

    structure: dict[int, PageStructure] = {}
    current_section = ""
    previous_part: str | None = None

    for (page_number, text), part_code in zip(pages, parts):
        if part_code != previous_part:
            current_section = ""
            previous_part = part_code

        lines = text.split("\n")
        for index, line in enumerate(lines):
            match = SECTION_HEADING_RE.match(line)
            if match:
                # Contents lines repeat the heading with a trailing page
                # number; drop that so the stored title is clean.
                heading = re.sub(r"\s+\d+$", "", match.group(1)).strip()
                current_section = _join_wrapped_heading(heading, lines[index + 1 :])
                break

        structure[page_number] = PageStructure(
            part_code=part_code,
            part_title=PART_TITLES.get(part_code, ""),
            section_title=current_section,
            is_index=looks_like_index(text) or looks_like_publisher_matter(text),
        )

    return structure
