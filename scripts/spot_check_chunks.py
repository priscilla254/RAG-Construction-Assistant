"""
Print random chunks from data/chunks.jsonl and flag suspicious cuts.

Use this after run_chunking.py to answer:
  - Does each chunk make sense on its own?
  - Is anything cut mid-sentence in a bad way?

Examples:
  python scripts/spot_check_chunks.py
  python scripts/spot_check_chunks.py --source hsg150.pdf
  python scripts/spot_check_chunks.py --chunk-type regulatory_clause --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.documents import known_headings_for  # noqa: E402
from rag_assistant.headings import (  # noqa: E402
    MAX_WRAPPED_HEADING_LINES,
    looks_like_heading,
)

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# Structural chunks are whole units of the source document and should not
# start or end mid-thought. The remaining types ("reference_material",
# "generic") are word windows, which are expected to cut mid-sentence.
STRUCTURAL_TYPES = frozenset(
    {
        "regulatory_clause",
        "procedural_guidance",
        "case_study",
        "topic_guidance",
    }
)

SENTENCE_END_RE = re.compile(r"[.?!:\"\')]$")


def load_chunks(
    path: Path,
    source: str | None = None,
    chunk_type: str | None = None,
) -> list[dict]:
    if not path.is_file():
        print(f"Chunks file not found: {path}")
        print("Run: python scripts/run_chunking.py")
        sys.exit(1)

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if source and row.get("source_doc") != source:
            continue
        if chunk_type and row.get("chunk_type") != chunk_type:
            continue
        rows.append(row)
    return rows


def flag_suspicious_cut(chunk: dict) -> list[str]:
    """
    Heuristic flags for bad mid-sentence cuts and trailing heading bleed.

    Not perfect — list items and abbreviations can false-positive — but
    useful for spotting structural chunks that were split incorrectly.
    """
    text = (chunk.get("text") or "").strip()
    if not text:
        return ["empty text"]

    flags: list[str] = []
    chunk_type = chunk.get("chunk_type", "")

    if text[0].islower() and not text.startswith(("e.g.", "i.e.", "eg ", "ie ")):
        flags.append("starts lowercase (possible mid-sentence cut)")

    # Scoped per document: HSG150's topic "Fire" would otherwise match a
    # line ending "...fire." anywhere in the corpus.
    known = known_headings_for(chunk.get("source_doc", ""))
    bleed = detect_trailing_heading_bleed(text, known)
    if bleed:
        flags.append(f"trailing heading bleed: {bleed!r}")

    tail = text[-1]
    if not SENTENCE_END_RE.search(text) and not bleed:
        if not re.search(r"\d$", text[-20:]):
            flags.append("no sentence-ending punctuation")

    if chunk_type in STRUCTURAL_TYPES and flags:
        return [f"STRUCTURAL: {f}" for f in flags]
    return flags


def detect_trailing_heading_bleed(
    text: str, known_headings: frozenset[str] = frozenset()
) -> str | None:
    """
    Return the suspected heading at the end of a chunk, if any.

    Shares `looks_like_heading` with the stripper in headings.py, so what
    this flags and what chunking removes cannot drift apart. Headings that
    wrapped across lines are rejoined before testing — a bare continuation
    like "safety" is invisible on its own.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None
    for span in range(1, min(MAX_WRAPPED_HEADING_LINES, len(lines)) + 1):
        candidate = " ".join(lines[-span:])
        if looks_like_heading(candidate, known_headings):
            return candidate
    return None


def print_chunk(index: int, chunk: dict, max_chars: int) -> bool:
    flags = flag_suspicious_cut(chunk)
    print("=" * 80)
    print(f"Spot check {index}: {chunk['chunk_id']}")
    print(f"  source : {chunk['source_doc']}  p.{chunk['page_number']}")
    print(f"  type   : {chunk.get('chunk_type', '')}")
    if chunk.get("section"):
        print(f"  section: {chunk['section']}")
    if chunk.get("part_code"):
        print(f"  part   : {chunk['part_code']}")
    if chunk.get("clause_number"):
        print(f"  clause : {chunk['clause_number']}")
    if chunk.get("context_prefix"):
        print(f"  prefix : {chunk['context_prefix']}")
    word_count = len(chunk["text"].split())
    print(f"  words  : {word_count}")

    if flags:
        print(f"  FLAGS  : {'; '.join(flags)}")
    else:
        print("  FLAGS  : none")

    print("-" * 80)
    body = chunk["text"]
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[... truncated ...]"
    encoding = sys.stdout.encoding or "utf-8"
    print(body.encode(encoding, errors="replace").decode(encoding))
    print()
    return bool(flags)


def corpus_summary(chunks: list[dict]) -> None:
    by_type: dict[str, int] = {}
    flagged_by_type: dict[str, int] = {}
    reasons: dict[str, int] = {}
    suspicious_structural = 0
    suspicious_any = 0

    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
        flags = flag_suspicious_cut(chunk)
        if not flags:
            continue
        suspicious_any += 1
        flagged_by_type[chunk_type] = flagged_by_type.get(chunk_type, 0) + 1
        for flag in flags:
            reason = flag.removeprefix("STRUCTURAL: ").split(":")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
        if any(f.startswith("STRUCTURAL:") for f in flags):
            suspicious_structural += 1

    print("=" * 80)
    print(f"Corpus: {len(chunks)} chunks")
    print("By chunk_type:", dict(sorted(by_type.items())))
    print(
        f"Suspicious cuts (heuristic): {suspicious_any} total, "
        f"{suspicious_structural} on structural chunks "
        f"({', '.join(sorted(STRUCTURAL_TYPES))})"
    )
    for chunk_type, total in sorted(by_type.items()):
        flagged = flagged_by_type.get(chunk_type, 0)
        marker = " [structural]" if chunk_type in STRUCTURAL_TYPES else ""
        print(
            f"  {chunk_type}{marker}: {flagged}/{total} flagged "
            f"({flagged / total:.0%})"
        )
    if reasons:
        print("By reason:", dict(sorted(reasons.items(), key=lambda i: -i[1])))
    if suspicious_structural:
        print(
            "Inspect structural chunks with FLAGS — clauses and numbered "
            "paragraphs should usually be whole units."
        )
    else:
        print("No suspicious cuts on structural chunks.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", "-n", type=int, default=5, help="number of chunks to print"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducible samples"
    )
    parser.add_argument(
        "--source",
        help="filter by source_doc filename, e.g. hsg150.pdf",
    )
    parser.add_argument(
        "--chunk-type",
        help="filter by chunk_type, e.g. regulatory_clause or generic",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="truncate displayed text at this many characters",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=CHUNKS_PATH,
        help="path to chunks.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.path, args.source, args.chunk_type)
    if not chunks:
        print("No chunks matched the filters.")
        sys.exit(1)
    if len(chunks) < args.count:
        print(f"Only {len(chunks)} chunks match; showing all of them.")
        sample = chunks
    else:
        sample = random.Random(args.seed).sample(chunks, args.count)

    any_flags = False
    for index, chunk in enumerate(sample, start=1):
        any_flags = print_chunk(index, chunk, args.max_chars) or any_flags

    corpus_summary(chunks)
    if any_flags:
        print("Sample contained flagged chunks — read the text above.")
    else:
        print("Sample looked clean under the heuristic flags.")


if __name__ == "__main__":
    main()
