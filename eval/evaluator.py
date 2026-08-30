"""Evaluator for labeled questions (recall@k)."""

from __future__ import annotations

import json
from pathlib import Path


class Evaluator:
    def __init__(self, eval_set_path: Path) -> None:
        self.eval_set_path = eval_set_path
        self.items = json.loads(eval_set_path.read_text(encoding="utf-8"))

    def recall_at_k(self, retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top = set(retrieved_ids[:k])
        hits = sum(1 for doc_id in relevant_ids if doc_id in top)
        return hits / len(relevant_ids)

    def run(self) -> dict:
        raise NotImplementedError
