"""
Watermark and margin-code removal for the Approved Documents PDF.

That PDF carries a diagonal "ONLINE VERSION" watermark which pdfplumber
reads as spaced single letters ("O N L I N E V E R S I O N"), often mixed
with the requirement codes printed in the margin ("R38", "B1", "A2/3").

Nothing here is specific to the chunking strategy, so ingest.py applies it
to every document; it is a no-op on the HSE publications, which carry no
watermark and no margin codes.
"""

from __future__ import annotations

import re

MARGIN_CODE_RE = re.compile(r"^(?:R\d+|[A-S]\d(?:/\d)?(?:\([0-9a-z]+\))?[A-Z]?)$", re.I)

# Below this, a run of single letters is more likely to be real text
# ("a", "I") than watermark fragments.
MIN_WATERMARK_RUN = 3


def strip_watermark_line(line: str) -> str:
    """
    Remove spaced-out watermark letters from one line.

    Runs of three or more single-letter tokens are dropped; shorter runs
    are kept so real words are not damaged.
    """
    tokens = line.split()
    kept: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) < MIN_WATERMARK_RUN:
            kept.extend(run)
        run = []
        kept.append(token)
    if len(run) < MIN_WATERMARK_RUN:
        kept.extend(run)
    return " ".join(kept)


def strip_watermark_text(text: str) -> str:
    """Apply watermark stripping line-by-line and drop margin-only lines."""
    lines: list[str] = []
    for line in text.split("\n"):
        cleaned = strip_watermark_line(line).strip()
        if not cleaned or MARGIN_CODE_RE.match(cleaned):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def first_substantive_line(text: str) -> str:
    """First line that is neither watermark residue nor a margin code."""
    for line in text.split("\n"):
        line = strip_watermark_line(line).strip()
        if line and not MARGIN_CODE_RE.match(line):
            return line
    return ""
