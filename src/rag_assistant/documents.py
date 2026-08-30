"""
Registry tying each source PDF to its own structure module.

Chunking dispatches on doc_type, but heading stripping and the reporting
scripts need a lookup by filename. Keeping that mapping here rather than
in either consumer stops the two drifting apart, and avoids a cycle: the
per-document modules import `headings`, so `headings` cannot import them.
"""

from __future__ import annotations

from rag_assistant.hsg141 import HSG141_KNOWN_HEADINGS
from rag_assistant.hsg150 import HSG150_KNOWN_HEADINGS

# The Approved Documents run to 1,541 pages with no usable contents page,
# so they have no known-title list and fall back to the shape heuristic.
MAD_SOURCE_DOC = "The_Merged_Approved_Documents_Oct24.pdf"

KNOWN_HEADINGS_BY_SOURCE: dict[str, frozenset[str]] = {
    "hsg141.pdf": HSG141_KNOWN_HEADINGS,
    "hsg150.pdf": HSG150_KNOWN_HEADINGS,
    MAD_SOURCE_DOC: frozenset(),
}


def is_merged_approved_document(chunk: dict) -> bool:
    """True for the merged Approved Documents PDF.

    Its page numbers are the bound volume, not the original Parts, so
    citations should use clause/section instead of p.N.
    """
    source = chunk.get("source_doc") or ""
    if source == MAD_SOURCE_DOC:
        return True
    if chunk.get("doc_type") == "regulatory":
        return True
    title = chunk.get("title") or ""
    return "Merged Approved Documents" in title


def known_headings_for(source_doc: str) -> frozenset[str]:
    """Heading titles from a document's contents page, empty if unknown."""
    return KNOWN_HEADINGS_BY_SOURCE.get(source_doc, frozenset())
