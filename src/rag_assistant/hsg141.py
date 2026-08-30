"""
Structure detection for HSG141 (Electrical safety on construction sites).

Paragraph numbering is continuous across the whole document (1-87), so a
numeric range lookup is enough to label which Part a paragraph belongs to
-- unlike HSG150, which needs offsets to forward-fill its section/topic.

Case study boxes sit inside a numbered paragraph but are narratively
distinct, so they are pulled out into their own chunk by the chunker.
"""

from __future__ import annotations

import re

from rag_assistant.headings import normalise_heading

# From HSG141's own table of contents.
HSG141_PART_RANGES: list[tuple[int, int, str]] = [
    (1, 12, "Introduction"),
    (13, 37, "Part 1: Planning (pre-construction phase)"),
    (38, 74, "Part 2: Managing electrical risks (construction phase)"),
    (75, 87, "Part 3: Systems and equipment"),
]

# Headings from HSG141's contents page. A heading sits above the paragraph
# it introduces, so it lands at the tail of the preceding paragraph's text
# and has to be stripped there.
HSG141_KNOWN_HEADINGS: frozenset[str] = frozenset(
    normalise_heading(title)
    for title in [
        "INTRODUCTION",
        "Who should read this guidance?",
        "Competence",
        "PART 1 PLANNING: THE PRE-CONSTRUCTION PHASE",
        "PART 2 MANAGING ELECTRICAL RISKS: THE CONSTRUCTION PHASE",
        "PART 3 SYSTEMS AND EQUIPMENT",
        "Overhead power lines",
        "Underground services",
        "Underground cables",
        "Distribution network operators",
        "Existing electrical services in buildings",
        "Temporary distribution systems",
        "Temporary electrical distribution systems",
        "Electrical distribution systems",
        "Isolation procedures",
        "Earthing the site’s supply",
        "Maintenance and inspection",
        "Commissioning permanent installations",
        "Portable tools and equipment",
        "Equipment for use on construction sites",
        "Cables",
        "Generators",
        "REFERENCES",
        "FURTHER READING",
        "HSE publications",
        "Industry guidance",
        "British Standards",
        "FURTHER INFORMATION",
    ]
)

# Marks the start of an embedded case study box. Anchored to a line start:
# ingest.py rebuilds visual lines, so the box heading sits on its own line,
# and anchoring avoids matching a passing mention of the phrase in body text.
CASE_STUDY_START_RE = re.compile(r"(?m)^Case study\b")


def section_for_paragraph(para_num: int) -> str:
    """Return the Part label covering a paragraph number."""
    for start, end, label in HSG141_PART_RANGES:
        if start <= para_num <= end:
            return label
    return "Unknown section"
