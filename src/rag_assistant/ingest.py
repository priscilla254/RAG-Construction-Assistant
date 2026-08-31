"""
PDF ingestion: raw PDFs -> cleaned per-page JSON.

Fixes applied here vs. the naive first pass:
  1. Column-aware extraction (word-bucketing by x-midpoint, not a hard crop)
     -> fixes column interleaving without truncating words like "advic"/"operatio".
     A page is only split when a real vertical gutter exists, so full-width
     single-column pages are not scrambled.
  2. Footnote/superscript filtering by font size, not by regex on digits
     -> avoids clipping real numbers like "1998" down to "199".
  3. Exact-line header stripping (whole line must match, not a prefix match)
     -> fixes the back-cover page being wiped to empty.
  4. Empty-page-after-cleaning warning
     -> catches bugs like #3 automatically instead of relying on manual spot checks.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from rag_assistant.config import Config
from rag_assistant.publisher_matter import looks_like_publisher_matter
from rag_assistant.watermark import strip_watermark_text

# Minimum fraction of the dominant body font size a character must have
# to be kept. Anything smaller is treated as a footnote/superscript marker.
FOOTNOTE_SIZE_THRESHOLD = 0.85

# If a page's raw extraction has real content but the cleaned text ends up
# shorter than this many characters, treat it as a likely bug and warn.
MIN_EXPECTED_CLEANED_CHARS = 20

# Words whose baselines differ by less than this many points belong to the
# same visual line.
LINE_TOLERANCE = 2.5

# Known running headers, per filename, to strip. These must match a
# COMPLETE line exactly (after whitespace normalisation) -- not a prefix --
# so a page like the back cover (which reuses similar wording in a longer
# blurb) doesn't get wiped out by a partial match.
KNOWN_HEADERS: dict[str, set[str]] = {
    "hsg141.pdf": {
        "Electrical Safety on Construction Sites",
        "Electrical safety on",
        "construction sites",
        "PART 1 PLANNING: THE PRE-CONSTRUCTION PHASE",
        "PART 1 PLANNING: THE PRE-CONSTRUCTION",
        "PART 2 MANAGING ELECTRICAL RISKS:",
        "PART 2 MANAGING ELECTRICAL RISKS: THE CONSTRUCTION PHASE",
        "THE CONSTRUCTION PHASE",
        "PART 3 SYSTEMS AND EQUIPMENT",
    },
    "hsg150.pdf": {
        "Health and Safety",
        "Executive",
        "HSE Books",
        "Health and safety in construction",
    },
    "The_Merged_Approved_Documents_Oct24.pdf": {
        "ONLINE VERSION",
        "Building Regulations 2010",
    },
}

PAGE_NUMBER_RE = re.compile(
    r"""
    ^(?:
        \d{1,4}
        | page\s+\d+(?:\s+of\s+\d+)?
        | \d+\s*/\s*\d+
        | [\-–—]\s*\d+\s*[\-–—]
        | [ivxlcdm]{1,7}
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)
PAGE_OF_LINE_RE = re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE)
APPROVED_DOCUMENT_FOOTER_RE = re.compile(
    r"^Building Regulations 2010\b.*\bApproved Document\b|^Approved Document\b.*\bedition\b",
    re.IGNORECASE,
)
ONLINE_VERSION_COMPACT_RE = re.compile(r"^[a-z]{0,2}onlineversion[a-z]{0,2}$")

HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


@dataclass
class DocumentRecord:
    filename: str
    title: str
    source_url: str
    date: str
    publisher: str
    doc_type: str
    pages: int  # expected page count from manifest.csv, used as a sanity check


def load_manifest(manifest_path: Path) -> list[DocumentRecord]:
    """Read manifest.csv into a list of DocumentRecord."""
    records: list[DocumentRecord] = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                DocumentRecord(
                    filename=row["filename"],
                    title=row["title"],
                    source_url=row["source_url"],
                    date=row["date"],
                    publisher=row["publisher"],
                    doc_type=row["doc_type"],
                    pages=int(row["pages"]),
                )
            )
    return records


def dominant_font_size(page) -> float:
    """
    Most common character size on this page.

    Measured per page rather than document-wide: small-print pages
    (copyright notices, reference lists, dense tables) are set entirely
    below the document's body size, and a document-wide threshold would
    delete every character on them.
    """
    size_counts: dict[float, int] = {}
    for char in page.chars:
        size = round(char["size"], 1)
        size_counts[size] = size_counts.get(size, 0) + 1

    if not size_counts:
        return 0.0  # no characters at all (e.g. a pure-image page)
    return max(size_counts, key=size_counts.__getitem__)


def _lines_from_words(words: list[dict]) -> list[str]:
    """Rebuild visual lines from words, so reading order isn't scrambled."""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])) for line in lines
    ]


def _both_sides_are_prose(words: list[dict], gutter: float, min_words_per_line: float = 3.0) -> bool:
    """
    Reject splits where one side is not running text.

    Contents pages and tables put right-aligned page numbers in their own
    vertical strip; splitting on that would divorce every heading from its
    number, so require both sides to read like sentences.
    """
    for side in (
        [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter],
        [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter],
    ):
        lines = _lines_from_words(side)
        if not lines:
            return False
        if len(side) / len(lines) < min_words_per_line:
            return False
    return True


def find_column_gutter(words: list[dict], width: float) -> float | None:
    """
    Locate a real vertical gutter: an x-range in the middle of the page that
    no word crosses.

    Detecting the gap itself (rather than assuming a split at the page
    centre) matters because the gutter is rarely exactly centred, and
    full-width single-column text would otherwise be scrambled into halves.
    """
    if len(words) < 40:
        return None

    # Words crossing each x position. A handful of full-width headings or
    # figures span the gutter on an otherwise two-column page, so look for
    # the minimum rather than insisting on a completely empty band.
    crossings = [0] * (int(width) + 1)
    for word in words:
        start = max(0, int(word["x0"]))
        end = min(int(width), int(word["x1"]) + 1)
        for x in range(start, end):
            crossings[x] += 1

    search_lo, search_hi = int(width * 0.35), int(width * 0.65)
    if search_hi <= search_lo:
        return None
    band = crossings[search_lo:search_hi]
    # Tolerate a few full-width headings or figures spanning the gutter.
    allowed = max(3.0, len(words) * 0.02)

    # A real gutter is a wide strip, not a one-off gap where word spacing
    # happened to line up between adjacent lines.
    best_start = best_len = 0
    run_start: int | None = None
    for i, count in enumerate(band + [allowed + 1]):
        if count <= allowed:
            if run_start is None:
                run_start = i
            if i - run_start + 1 > best_len:
                best_start, best_len = run_start, i - run_start + 1
        else:
            run_start = None
    if best_len < width * 0.015:
        return None

    gutter = search_lo + best_start + best_len / 2
    if not _both_sides_are_prose(words, gutter):
        return None
    left = sum(1 for w in words if w["x1"] <= gutter)
    right = sum(1 for w in words if w["x0"] >= gutter)
    if left < len(words) * 0.2 or right < len(words) * 0.2:
        return None
    return gutter


def extract_page_text(page) -> str:
    """
    Extract text from a single page, handling two-column layout and
    filtering out footnote/superscript-sized characters.
    """
    # --- Fix 2: drop characters rendered smaller than this page's body text ---
    # Operates on the actual rendered font size rather than guessing from
    # digit shape in the flattened text, so real numbers aren't clipped.
    words = page.extract_words()
    body_size = dominant_font_size(page)
    if body_size > 0:
        filtered = page.filter(
            lambda obj: obj.get("object_type") != "char"
            or obj["size"] >= body_size * FOOTNOTE_SIZE_THRESHOLD
        )
        filtered_words = filtered.extract_words()
        # Safety net: if the filter ate most of the page, it misjudged the
        # body size, so keep the unfiltered text instead.
        if len(filtered_words) >= len(words) * 0.5:
            page, words = filtered, filtered_words

    if not words:
        return ""

    # --- Fix 1: bucket words into columns around the detected gutter ---
    # A word is assigned wholly to one column based on its own centre point,
    # so no word gets sliced the way a hard x-coordinate crop would.
    gutter = find_column_gutter(words, float(page.width))
    if gutter is None:
        return "\n".join(_lines_from_words(words))

    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter]
    return "\n".join(_lines_from_words(left) + _lines_from_words(right))


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _compact(line: str) -> str:
    return re.sub(r"\s+", "", line).casefold()


def is_known_header(line: str, headers: set[str]) -> bool:
    """Whole-line match against this document's known headers and page furniture."""
    stripped = _norm_line(line)
    if not stripped:
        return False
    if PAGE_NUMBER_RE.match(stripped) or PAGE_OF_LINE_RE.fullmatch(stripped):
        return True
    if ONLINE_VERSION_COMPACT_RE.match(_compact(stripped)):
        return True
    if APPROVED_DOCUMENT_FOOTER_RE.match(stripped):
        return True
    folded = {_norm_line(h).casefold() for h in headers}
    return stripped.casefold() in folded


def strip_headers_footers(text: str, filename: str = "") -> str:
    """
    Remove lines that exactly match a known running header.

    Requires a full-line match -- NOT a prefix/substring match -- so pages
    whose real content merely starts similarly to a header (e.g. the back
    cover blurb) survive intact.
    """
    headers = KNOWN_HEADERS.get(filename, set())
    kept = [
        line for line in text.split("\n") if line.strip() and not is_known_header(line, headers)
    ]
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Normalise hyphenation, whitespace, and watermark artefacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [MULTI_SPACE_RE.sub(" ", line).rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return strip_watermark_text(text.strip())


def ingest_document(pdf_path: Path, record: DocumentRecord, verbose: bool = True) -> dict:
    """Ingest a single PDF into the cleaned JSON structure."""
    pages_out: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        if total != record.pages:
            print(
                f"  WARNING: {record.filename} has {total} pages but manifest.csv "
                f"says {record.pages}."
            )

        for i, page in enumerate(pdf.pages, start=1):
            if verbose and total > 80 and i % 100 == 0:
                print(f"    extracted {i}/{total} pages")

            raw_text = page.extract_text() or ""
            extracted = extract_page_text(page)
            cleaned = clean_text(strip_headers_footers(extracted, record.filename))

            # --- Fix 4: warn instead of silently accepting empty pages ---
            if raw_text.strip() and len(cleaned) < MIN_EXPECTED_CLEANED_CHARS:
                print(
                    f"  WARNING: {record.filename} page {i} had "
                    f"{len(raw_text.strip())} raw chars but only "
                    f"{len(cleaned)} after cleaning -- "
                    f"check header-stripping didn't over-match this page."
                )

            if looks_like_publisher_matter(cleaned):
                print(
                    f"  dropped page {i}: title/copyright/order page, no guidance"
                )
                continue

            if cleaned:
                pages_out.append({"page_number": i, "text": cleaned})

    return {
        "filename": record.filename,
        "title": record.title,
        "source_url": record.source_url,
        "date": record.date,
        "publisher": record.publisher,
        "doc_type": record.doc_type,
        "page_count": total,
        "cleaned_page_count": len(pages_out),
        "pages": pages_out,
    }


def ingest_corpus(config: Config) -> list[Path]:
    """
    Run ingestion for every document listed in manifest.csv.
    Returns the list of paths written.
    """
    raw_dir = Path(config.paths.raw_dir)
    cleaned_dir = Path(config.paths.cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(config.paths.manifest_path)
    written: list[Path] = []

    for record in records:
        pdf_path = raw_dir / record.filename
        if not pdf_path.exists():
            print(f"  SKIPPING {record.filename}: not found in {raw_dir}")
            continue

        print(f"Ingesting {record.filename} ...")
        result = ingest_document(pdf_path, record)

        out_path = cleaned_dir / f"{pdf_path.stem}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(out_path)
        print(
            f"  wrote {out_path.name} "
            f"({result['cleaned_page_count']}/{result['page_count']} pages with text)"
        )

    return written
