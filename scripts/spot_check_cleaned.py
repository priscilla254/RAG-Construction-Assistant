"""Print 5 random cleaned-page excerpts and flag glued-word artefacts."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "cleaned"

GLUED_CAMEL = re.compile(r"[a-z]{5,}[A-Z][a-z]{3,}")
KNOWN_GLUED = (
    "distributionsystem",
    "constructionphase",
    "electricalsystem",
    "temporarydistribution",
    "electricalsafety",
    "powerlines",
)


def load_windows(min_chars: int = 400) -> list[tuple[str, int, str]]:
    windows: list[tuple[str, int, str]] = []
    for path in sorted(CLEANED.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        filename = record.get("filename", path.name)
        for page in record.get("pages", []):
            text = page.get("text") or ""
            if len(text) < min_chars:
                continue
            windows.append((filename, page["page_number"], text))
    return windows


def flag_glued(text: str) -> list[str]:
    hits = [term for term in KNOWN_GLUED if term in text.casefold().replace(" ", "")]
    hits.extend(GLUED_CAMEL.findall(text))
    return hits


def main() -> None:
    rng = random.Random(42)
    windows = load_windows()
    if len(windows) < 5:
        print(f"Not enough cleaned pages in {CLEANED}")
        sys.exit(1)

    sample = rng.sample(windows, 5)
    any_glued = False
    for i, (filename, page_number, text) in enumerate(sample, start=1):
        hits = flag_glued(text)
        any_glued = any_glued or bool(hits)
        print("=" * 80)
        print(f"Spot check {i}: {filename} p.{page_number}")
        if hits:
            print(f"GLUED CANDIDATES: {hits}")
        else:
            print("No glued-word candidates in this excerpt.")
        print("-" * 80)
        print(text[:1200].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        print()

    corpus_hits = 0
    for filename, page_number, text in windows:
        hits = flag_glued(text)
        if hits:
            corpus_hits += 1
    print("=" * 80)
    print(f"Pages scanned: {len(windows)}; pages with glued-word candidates: {corpus_hits}")
    if any_glued or corpus_hits:
        print("Extraction still has some glued-word artefacts — inspect before Day 2.")
    else:
        print("No glued-word artefacts found in the sampled/scanned pages.")


if __name__ == "__main__":
    main()
