import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from evaluator import Evaluator  # noqa: E402


class _FakeRetriever:
    def __init__(self, by_question: dict[str, list[dict]], k: int = 5) -> None:
        self.by_question = by_question
        self.k = k
        self.seen: list[tuple[str, int | None]] = []

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        self.seen.append((question, k))
        hits = self.by_question.get(question, [])
        limit = self.k if k is None else k
        return hits[:limit]


def test_recall_at_k_is_fraction_of_relevant_ids_in_top_k():
    evaluator = Evaluator(_FakeRetriever({}))
    assert evaluator.recall_at_k(["a", "b", "c"], ["a", "d"], k=2) == 0.5
    assert evaluator.recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert evaluator.recall_at_k(["x"], ["a"], k=1) == 0.0
    assert evaluator.recall_at_k(["a"], [], k=1) == 0.0


def test_run_scores_single_source_hit_and_miss():
    question = "How high should scaffold guard rails be?"
    gold = {
        "question": question,
        "expected_source": "hsg150.pdf",
        "expected_section": (
            "Section 3: Construction-phase health and safety — "
            "Working at height — Guard rails (paragraphs 148-151)"
        ),
    }
    hit_chunk = {
        "chunk_id": "hsg150_0090",
        "source_doc": "hsg150.pdf",
        "paragraph_numbers": "150-151",
        "section": "Section 3: Construction-phase health and safety",
        "topic": "Working at height",
    }
    miss_chunk = {
        "chunk_id": "hsg141_0001",
        "source_doc": "hsg141.pdf",
        "section": "Part 1: Planning (pre-construction phase) (para 36)",
    }
    retriever = _FakeRetriever({question: [hit_chunk, miss_chunk]})
    result = Evaluator(retriever, k=5).run([gold])

    assert result["recall_at_k"] == 1.0
    row = result["results_table"][0]
    assert row["question"] == question
    assert row["expected_source"] == "hsg150.pdf"
    assert row["retrieved"] == ["hsg150_0090", "hsg141_0001"]
    assert row["recall"] == 1.0
    assert row["hit"] is True
    assert "category" not in row

    retriever = _FakeRetriever({question: [miss_chunk]})
    missed = Evaluator(retriever, k=5).run([gold])
    assert missed["recall_at_k"] == 0.0
    assert missed["results_table"][0]["hit"] is False


def test_run_partial_recall_when_one_of_two_sources_is_retrieved():
    question = "How should buried cables be located before excavation?"
    item = {
        "question": question,
        "expected_source": ["hsg141.pdf", "hsg150.pdf"],
        "expected_section": [
            "Part 1: Planning (pre-construction phase) (paragraphs 21-22)",
            "Section 3: Groundwork — Underground services (paragraphs 352-353)",
        ],
    }
    retriever = _FakeRetriever(
        {
            question: [
                {
                    "chunk_id": "hsg141_0021",
                    "source_doc": "hsg141.pdf",
                    "section": "Part 1: Planning (pre-construction phase) (para 21)",
                }
            ]
        }
    )
    result = Evaluator(retriever, k=5).run([item])
    assert result["recall_at_k"] == 0.5
    assert result["results_table"][0]["hit"] is False
    assert result["results_table"][0]["recall"] == 0.5


def test_run_excludes_unlabeled_items_from_mean_recall():
    labeled = {
        "question": "When must a site be notified to HSE?",
        "expected_source": "hsg150.pdf",
        "expected_section": "Section 1: Preparing for work (paragraphs 34-36)",
    }
    unlabeled = {
        "question": "What is the minimum ceiling height for a habitable room?",
        "expected_source": None,
        "expected_section": None,
    }
    retriever = _FakeRetriever(
        {
            labeled["question"]: [
                {
                    "chunk_id": "hsg150_0021",
                    "source_doc": "hsg150.pdf",
                    "paragraph_numbers": "34-36",
                    "section": "Section 1: Preparing for work",
                }
            ],
            unlabeled["question"]: [
                {
                    "chunk_id": "mad_headroom",
                    "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
                    "part_code": "K",
                    "clause_number": "1.13-1.15",
                    "section": "Section 1: Stairs and ladders",
                }
            ],
        }
    )
    result = Evaluator(retriever, k=5).run([labeled, unlabeled])
    assert result["recall_at_k"] == 1.0
    assert result["results_table"][1]["recall"] is None
    assert result["results_table"][1]["hit"] is None
    assert result["results_table"][1]["retrieved"] == ["mad_headroom"]
    assert "category" not in result["results_table"][1]


def test_run_empty_eval_set():
    result = Evaluator(_FakeRetriever({})).run([])
    assert result == {"recall_at_k": 0.0, "results_table": []}


def test_run_accepts_json_path(tmp_path: Path):
    question = "Must the rise and going be consistent?"
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": question,
                    "expected_source": "The_Merged_Approved_Documents_Oct24.pdf",
                    "expected_section": (
                        "Approved Document K — Section 1: Stairs and ladders "
                        "(clauses 1.5-1.6)"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    retriever = _FakeRetriever(
        {
            question: [
                {
                    "chunk_id": "mad_2280",
                    "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
                    "part_code": "K",
                    "clause_number": "1.5-1.6",
                    "section": "Section 1: Stairs and ladders",
                }
            ]
        }
    )
    result = Evaluator(retriever).run(path)
    assert result["recall_at_k"] == 1.0
    assert retriever.seen == [(question, 5)]


def test_run_does_not_match_same_clause_in_a_different_approved_document():
    question = "What escape provision do habitable rooms need?"
    item = {
        "question": question,
        "expected_source": "The_Merged_Approved_Documents_Oct24.pdf",
        "expected_section": (
            "Approved Document B — Section 2: Means of escape – dwellinghouses "
            "(clauses 2.1-2.2)"
        ),
    }
    retriever = _FakeRetriever(
        {
            question: [
                {
                    "chunk_id": "mad_k_2_1",
                    "source_doc": "The_Merged_Approved_Documents_Oct24.pdf",
                    "part_code": "K",
                    "clause_number": "2.1",
                    "section": "Section 2: Ramps",
                }
            ]
        }
    )
    result = Evaluator(retriever, k=5).run([item])
    assert result["recall_at_k"] == 0.0


def test_parse_expected_section_keeps_both_paragraph_ranges():
    from evaluator import _parse_expected_section

    paras, clauses, part = _parse_expected_section(
        "HSG150 Section 3 — Underground services "
        "(paragraphs 352-353 and 366-369)"
    )
    assert paras == set(range(352, 354)) | set(range(366, 370))
    assert clauses == set()
    assert part is None
