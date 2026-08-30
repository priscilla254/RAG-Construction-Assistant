"""
Detecting and stripping headings that bleed into a neighbouring chunk.

Every chunking strategy splits on numbering (clause numbers, paragraph
numbers), but a heading sits *above* the unit it introduces. Splitting on
numbers alone therefore strands each heading at the tail of the preceding
unit, where it is noise for retrieval.

Detection is deliberately two-tier: an exact match against titles taken
from a document's own contents page, then a shape heuristic for headings
that never appear in the contents. The known-title tier exists because the
heuristic cannot safely be loosened far enough to catch everything --
body text that merely wraps without a full stop is indistinguishable from
a heading by shape alone.
"""

from __future__ import annotations

import re

SECTION_HEADING_RE = re.compile(r"^\s*Section\s+\d+", re.I)

# Section titles lose the word "Section" in extraction on some pages, so
# they arrive as "3: Construction-phase health and safety". The numbering
# is stripped before the title is case-checked.
HEADING_NUMBER_PREFIX_RE = re.compile(r"^\s*(?:Section\s+)?\d+\s*:\s*", re.I)

# A heading set in a narrow column wraps, so the final line of a chunk can
# be a bare continuation ("safety") that reads like nothing at all. Trailing
# lines are rejoined before testing.
MAX_WRAPPED_HEADING_LINES = 3

# Above this, a candidate is prose rather than a title.
MAX_HEADING_WORDS = 12


def normalise_heading(value: str) -> str:
    """Key for comparing a candidate line against a known heading title."""
    without_number = HEADING_NUMBER_PREFIX_RE.sub("", value.strip())
    return re.sub(r"\s+", " ", without_number).strip().rstrip(".").lower()


def looks_like_heading(
    candidate: str, known_headings: frozenset[str] | set[str] = frozenset()
) -> bool:
    """
    Decide whether one line (or a rejoined wrapped heading) is a heading.

    `known_headings` holds titles taken from a document's own contents
    page. Matching those exactly catches headings the shape heuristics
    below miss, and costs nothing in false positives.
    """
    line = candidate.strip()
    if not line:
        return False
    if known_headings and normalise_heading(line) in known_headings:
        return True
    # Length-guarded: body text citing "Section 59 (Drainage of building) of
    # the Building Act 1984..." is not a heading.
    if SECTION_HEADING_RE.match(line) and len(line.split()) <= MAX_HEADING_WORDS:
        return True
    if re.match(r"^[A-Z]{1,2}$", line):
        return True

    title = HEADING_NUMBER_PREFIX_RE.sub("", line)
    if not title:
        return False
    # A heading is a short noun phrase. Body text that merely wraps without
    # a full stop looks identical unless these are excluded: an internal
    # comma ("If isolations are undertaken by others, you should get") and a
    # trailing digit (a displaced footnote marker, "(PUWER 1998). 8 8").
    # Real headings with either shape are still caught by known_headings.
    if "," in title or title[-1].isdigit():
        return False
    words = title.split()
    return (
        len(words) <= MAX_HEADING_WORDS
        and title[0].isupper()
        and not re.match(r"^\d+\.\d+", title)
        and title[-1] not in ".?!"
        and not title.lower().startswith(("note:", "diagram", "table", "figure"))
    )


def trim_trailing_headings(
    text: str, known_headings: frozenset[str] | set[str] = frozenset()
) -> str:
    """
    Remove section/subsection headings accidentally included at a unit's end.

    The splitter runs until the next clause or paragraph number, so
    unnumbered lines at the tail — the next section title, a diagram
    heading — ride along unless they are stripped here.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    while lines:
        for span in range(1, min(MAX_WRAPPED_HEADING_LINES, len(lines)) + 1):
            candidate = " ".join(line.strip() for line in lines[-span:])
            if looks_like_heading(candidate, known_headings):
                del lines[-span:]
                break
        else:
            break
    return "\n".join(lines).strip()
