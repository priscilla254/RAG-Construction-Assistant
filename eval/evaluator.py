"""Evaluator for labeled questions (recall@k)."""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_K = 5

_PARA_RE = re.compile(
    r"(?:paragraphs?|paras?)\s+(\d+)(?:\s*[-–]\s*(\d+))?",
    re.I,
)
_CLAUSE_RE = re.compile(
    r"clauses?\s+(\d+\.\d+)(?:\s*[-–]\s*(\d+\.\d+))?",
    re.I,
)
_AD_RE = re.compile(r"Approved Document\s+([A-Z])\b", re.I)
_CHUNK_PARA_FIELD_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
_CHUNK_CLAUSE_FIELD_RE = re.compile(r"^(\d+\.\d+)(?:-(\d+\.\d+))?$")
_SECTION_PARA_RE = re.compile(
    r"para(?:graph)?s?\s+(\d+)(?:\s*[-–]\s*(\d+))?",
    re.I,
)


class Evaluator:
    def __init__(self, retriever, k: int | None = None) -> None:
        self.retriever = retriever
        self.k = k

    def recall_at_k(self, retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top = set(retrieved_ids[:k])
        hits = sum(1 for doc_id in relevant_ids if doc_id in top)
        return hits / len(relevant_ids)

    def run(self, eval_set: list[dict] | Path | str) -> dict:
        items = _load_eval_set(eval_set)
        k = self._k()
        rows: list[dict] = []
        scores: list[float] = []

        for item in items:
            question = item.get("question") or ""
            gold = _gold_pairs(item)
            retrieved = list(self.retriever.retrieve(question, k=k) or [])[:k]
            retrieved_ids = [str(chunk.get("chunk_id") or "") for chunk in retrieved]

            if not gold:
                rows.append(
                    _row(item, retrieved_ids, recall=None, hit=None)
                )
                continue

            gold_keys = [_gold_key(source, section) for source, section in gold]
            matched_keys = [
                key
                for key, (source, section) in zip(gold_keys, gold)
                if any(_chunk_matches(chunk, source, section) for chunk in retrieved)
            ]
            recall = self.recall_at_k(matched_keys, gold_keys, k)
            scores.append(recall)
            rows.append(_row(item, retrieved_ids, recall=recall, hit=recall == 1.0))

        return {
            "recall_at_k": (sum(scores) / len(scores)) if scores else 0.0,
            "results_table": rows,
        }

    def _k(self) -> int:
        if self.k is not None:
            return self.k
        retriever_k = getattr(self.retriever, "k", None)
        if isinstance(retriever_k, int) and retriever_k > 0:
            return retriever_k
        return DEFAULT_K


def _load_eval_set(eval_set: list[dict] | Path | str) -> list[dict]:
    if isinstance(eval_set, (str, Path)):
        path = Path(eval_set)
        return json.loads(path.read_text(encoding="utf-8"))
    return list(eval_set)


def _row(
    item: dict,
    retrieved: list[str],
    recall: float | None,
    hit: bool | None,
) -> dict:
    return {
        "question": item.get("question") or "",
        "expected_source": item.get("expected_source"),
        "expected_section": item.get("expected_section"),
        "retrieved": retrieved,
        "recall": recall,
        "hit": hit,
    }


def _gold_pairs(item: dict) -> list[tuple[str, str]]:
    source = item.get("expected_source")
    section = item.get("expected_section")
    if source is None:
        return []
    sources = source if isinstance(source, list) else [source]
    if section is None:
        sections: list[str] = [""] * len(sources)
    elif isinstance(section, list):
        sections = [item if item is not None else "" for item in section]
    else:
        sections = [section] * len(sources)
    pairs: list[tuple[str, str]] = []
    for src, sec in zip(sources, sections):
        if src is None:
            continue
        pairs.append((str(src), str(sec)))
    return pairs


def _gold_key(source: str, section: str) -> str:
    return f"{source}::{section}"


def _chunk_matches(chunk: dict, source: str, section: str) -> bool:
    if not _source_matches(chunk.get("source_doc") or "", source):
        return False

    expected_paras, expected_clauses, expected_part = _parse_expected_section(section)
    if expected_part and not _part_matches(chunk, expected_part):
        return False

    chunk_paras, chunk_clauses = _ids_from_chunk(chunk)
    if expected_paras:
        if chunk_paras:
            return bool(expected_paras & chunk_paras)
        return _text_matches(chunk, section)
    if expected_clauses:
        if chunk_clauses:
            return bool(expected_clauses & chunk_clauses)
        return _text_matches(chunk, section)
    return _text_matches(chunk, section)


def _source_matches(source_doc: str, expected: str) -> bool:
    left = Path(str(source_doc)).name.lower()
    right = Path(str(expected)).name.lower()
    return bool(left) and left == right


def _part_matches(chunk: dict, expected_part: str) -> bool:
    part = (chunk.get("part_code") or "").strip().upper()
    if part == expected_part:
        return True
    blob = " ".join(
        [
            chunk.get("context_prefix") or "",
            chunk.get("section") or "",
            chunk.get("title") or "",
        ]
    ).upper()
    return f"APPROVED DOCUMENT {expected_part}" in blob


def _parse_expected_section(section: str) -> tuple[set[int], set[str], str | None]:
    paras: set[int] = set()
    clauses: set[str] = set()
    for match in _PARA_RE.finditer(section or ""):
        paras |= _int_range(match.group(1), match.group(2))
    for match in _CLAUSE_RE.finditer(section or ""):
        clauses |= _clause_range(match.group(1), match.group(2))
    ad = _AD_RE.search(section or "")
    part = ad.group(1).upper() if ad else None
    return paras, clauses, part


def _ids_from_chunk(chunk: dict) -> tuple[set[int], set[str]]:
    paras: set[int] = set()
    clauses: set[str] = set()
    paragraph_numbers = (chunk.get("paragraph_numbers") or "").strip()
    match = _CHUNK_PARA_FIELD_RE.match(paragraph_numbers)
    if match:
        paras |= _int_range(match.group(1), match.group(2))
    clause_number = (chunk.get("clause_number") or "").strip()
    match = _CHUNK_CLAUSE_FIELD_RE.match(clause_number)
    if match:
        clauses |= _clause_range(match.group(1), match.group(2))
    for field in (chunk.get("section") or "", chunk.get("context_prefix") or ""):
        for para_match in _SECTION_PARA_RE.finditer(field):
            paras |= _int_range(para_match.group(1), para_match.group(2))
    return paras, clauses


def _int_range(start: str, end: str | None) -> set[int]:
    low = int(start)
    high = int(end) if end is not None else low
    if high < low:
        low, high = high, low
    return set(range(low, high + 1))


def _clause_range(start: str, end: str | None) -> set[str]:
    if end is None:
        return {start}
    start_major, start_minor = start.split(".", 1)
    end_major, end_minor = end.split(".", 1)
    if start_major == end_major:
        low, high = int(start_minor), int(end_minor)
        if high < low:
            low, high = high, low
        return {f"{start_major}.{index}" for index in range(low, high + 1)}
    return {start, end}


def _text_matches(chunk: dict, section: str) -> bool:
    blob = _normalise(
        " ".join(
            [
                chunk.get("context_prefix") or "",
                chunk.get("section") or "",
                chunk.get("topic") or "",
                chunk.get("subtopic") or "",
                chunk.get("title") or "",
            ]
        )
    )
    if not blob:
        return False
    for needle in _section_needles(section):
        if needle and needle in blob:
            return True
    return False


def _section_needles(section: str) -> list[str]:
    cleaned = _AD_RE.sub(" ", section or "")
    cleaned = re.sub(r"HSG1[45][01]\s*", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(
        r"clauses?\s+\d+\.\d+(?:\s*[-–]\s*\d+\.\d+)?",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"paragraphs?\s+\d+(?:\s*[-–]\s*\d+)?",
        " ",
        cleaned,
        flags=re.I,
    )
    parts = re.split(r"\s+and\s+(?=section\b)", cleaned, flags=re.I)
    needles = [_normalise(part) for part in parts]
    return [needle for needle in needles if len(needle) >= 12]


def _normalise(value: str) -> str:
    collapsed = re.sub(r"[\s—–\-]+", " ", value).strip().lower()
    return collapsed
