"""Validation checks for the chunked corpus."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.approved_documents import map_document  # noqa: E402

CHUNKS = ROOT / "data" / "chunks.jsonl"
CLEANED = ROOT / "data" / "cleaned" / "The_Merged_Approved_Documents_Oct24.json"
MAD = "The_Merged_Approved_Documents_Oct24.pdf"


def main() -> None:
    rows = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    mad = [r for r in rows if r["source_doc"] == MAD]
    clauses = [r for r in mad if r["chunk_type"] == "regulatory_clause"]
    pages = {p["page_number"]: (p["text"] or "") for p in json.loads(
        CLEANED.read_text(encoding="utf-8"))["pages"]}

    print("total chunks:", len(rows), "| MAD:", len(mad))
    counts: dict[str, int] = {}
    for r in mad:
        counts[r["chunk_type"]] = counts.get(r["chunk_type"], 0) + 1
    print("MAD chunk types:", counts)

    print()
    print("=== 1. spot check 5 random clause chunks against the page text ===")
    random.seed(11)
    for r in random.sample(clauses, 5):
        page = pages.get(r["page_number"], "")
        snippet = " ".join(r["text"].split()[:12])
        print(f"p{r['page_number']} Part {r['part_code']} clause {r['clause_number']}")
        print(f"   prefix : {r['context_prefix']}")
        print(f"   text   : {snippet}")
        print(f"   clause number appears on that page: "
              f"{r['clause_number'].split('-')[0] in page}")

    print()
    print("=== 2. same clause number in different Parts ===")
    by_number: dict[str, set[str]] = {}
    for r in clauses:
        by_number.setdefault(r["clause_number"], set()).add(r["part_code"])
    collisions = {k: v for k, v in by_number.items() if len(v) > 1}
    print(f"{len(collisions)} clause numbers appear in more than one Part")
    example = by_number.get("3.24", set())
    print("clause 3.24 appears in Parts:", sorted(example))
    for r in clauses:
        if r["clause_number"] == "3.24":
            print(f"   {r['chunk_id']} Part {r['part_code']} p{r['page_number']}: "
                  f"{' '.join(r['text'].split()[:10])}")

    print()
    print("=== 3. chunk sizes ===")
    sizes = sorted(len(r["text"].split()) for r in rows)
    print("max words:", sizes[-1], "| median:", sizes[len(sizes) // 2])
    print("chunks over 3000 words:", sum(1 for s in sizes if s > 3000))
    print("chunks over the configured window (400):", sum(1 for s in sizes if s > 400))

    print()
    print("=== 4. the old 16,489-word runaway ===")
    print("largest MAD chunk:", max(len(r["text"].split()) for r in mad), "words")

    print()
    print("=== 5. dropped pages ===")
    # A page that has no chunk starting on it is not necessarily dropped:
    # its text may continue a chunk that began on the previous page. Only
    # the index/contents pages are actually discarded.
    structure = map_document([(n, t) for n, t in sorted(pages.items())])
    dropped = sorted(n for n in pages if structure[n].is_index)
    print(len(dropped), "pages dropped as contents/index")
    for n in dropped[:3]:
        print(f"--- page {n} ---")
        print(pages[n][:180])
    no_chunk_start = sorted(set(pages) - {r["page_number"] for r in mad} - set(dropped))
    print(f"{len(no_chunk_start)} further pages start no chunk (continuation pages)")

    print()
    print("=== 6. prefix hygiene ===")
    missing = [r for r in clauses if not r["context_prefix"]]
    leaked = [r for r in clauses if r["context_prefix"] and r["context_prefix"] in r["text"]]
    print("clause chunks missing a prefix:", len(missing))
    print("clause chunks whose display text contains the prefix:", len(leaked))
    no_clause_number = [r for r in clauses if not r["clause_number"]]
    print("clause chunks with no clause number:", len(no_clause_number))


if __name__ == "__main__":
    main()
