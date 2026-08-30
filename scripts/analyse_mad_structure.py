"""Throwaway analysis: check Part/section detection on the merged file.

Not part of the pipeline.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.approved_documents import map_document  # noqa: E402

CLEANED = ROOT / "data" / "cleaned" / "The_Merged_Approved_Documents_Oct24.json"


def main() -> None:
    doc = json.loads(CLEANED.read_text(encoding="utf-8"))
    pages = [(p["page_number"], p["text"] or "") for p in doc["pages"]]
    structure = map_document(pages)

    print("=== part ranges ===")
    items = [(n, structure[n].part_code) for n, _ in pages]
    for key, group in itertools.groupby(items, key=lambda x: x[1]):
        group = list(group)
        print(f"{key}  {group[0][0]:>5} - {group[-1][0]:>5}   ({len(group)} pages)")

    print()
    print("=== index/contents pages ===")
    index_pages = [n for n, _ in pages if structure[n].is_index]
    print(len(index_pages), "pages:", index_pages[:25], "...")

    print()
    print("=== section coverage ===")
    with_section = sum(1 for n, _ in pages if structure[n].section_title)
    print(with_section, "of", len(pages), "pages have a section title")

    print()
    print("=== spot check ===")
    for n in (60, 300, 470, 700, 860, 1050, 1300, 1480):
        s = structure.get(n)
        if not s:
            continue
        head = " ".join((dict(pages)[n]).split()[:12])
        print(f"p{n}: {s.part_code} ({s.part_title}) | {s.section_title!r} | {head[:70]}")


if __name__ == "__main__":
    main()

