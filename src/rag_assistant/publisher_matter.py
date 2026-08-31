"""
Detect title, copyright, and publications-order pages.

Those pages sit in the source PDFs (NBS/RIBA order lines, HSE Books
covers) and match keyword queries without answering them. The 0845 300
9924 fax that appeared in generation is this class of text.
"""

from __future__ import annotations

import re

# Real guidance has numbered clauses (MAD) or numbered paragraphs (HSG).
CLAUSE_START_RE = re.compile(r"(?m)^\d+\.\d+\s+\S")
PARAGRAPH_START_RE = re.compile(r"(?m)^\d{1,3}\s+[A-Z]")

# Unique to publisher / order / copyright sheets.
STRONG_MARKERS = (
    re.compile(r"0845\s*300\s*9924"),
    re.compile(r"published by nbs", re.I),
    re.compile(r"riba enterprises", re.I),
    re.compile(r"www\.thebuildingregs\.com", re.I),
    re.compile(r"\bhse books\b", re.I),
    re.compile(r"isbn[\s:\-]*\d", re.I),
)

WEAK_MARKERS = (
    re.compile(r"(?im)^fax\s*:"),
    re.compile(r"(?im)^tel(?:ephone)?\s*:"),
    re.compile(r"crown copyright", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"po box\s+\d+", re.I),
    re.compile(r"\b(?:to order|mail order)\b", re.I),
    re.compile(r"further (?:information|copies)", re.I),
)

TITLE_PAGE_RE = re.compile(
    r"building regulations 2010|\bapproved document\b",
    re.I,
)


def _guidance_starts(text: str) -> int:
    return len(CLAUSE_START_RE.findall(text)) + len(PARAGRAPH_START_RE.findall(text))


def looks_like_publisher_matter(text: str) -> bool:
    """True for title, copyright, and order pages, not for clause pages."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _guidance_starts(stripped) >= 2:
        return False

    if any(pattern.search(stripped) for pattern in STRONG_MARKERS):
        return True
    weak_hits = sum(1 for pattern in WEAK_MARKERS if pattern.search(stripped))
    if weak_hits >= 2:
        return True

    words = stripped.split()
    if len(words) <= 80 and TITLE_PAGE_RE.search(stripped) and _guidance_starts(stripped) == 0:
        if weak_hits or "copyright" in stripped.lower():
            return True
    return False
