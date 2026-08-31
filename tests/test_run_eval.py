import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402


class _FakeRetriever:
    k = 5

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        return [
            {
                "chunk_id": "hsg150_0090",
                "source_doc": "hsg150.pdf",
                "paragraph_numbers": "150-151",
                "section": "Section 3: Construction-phase health and safety",
            }
        ]


def test_run_eval_prints_recall_and_rows(monkeypatch, tmp_path, capsys):
    eval_set = tmp_path / "eval_set.json"
    eval_set.write_text(
        """
        [
          {
            "question": "How high should scaffold guard rails be?",
            "expected_source": "hsg150.pdf",
            "expected_section": "Guard rails (paragraphs 148-151)"
          },
          {
            "question": "What is the minimum ceiling height for a habitable room?",
            "expected_source": null,
            "expected_section": null
          }
        ]
        """,
        encoding="utf-8",
    )
    output = tmp_path / "results.json"
    monkeypatch.setattr(run_eval, "_retriever", lambda config: _FakeRetriever())

    assert run_eval.main([str(eval_set), "--output", str(output)]) == 0

    printed = capsys.readouterr().out
    assert "recall@5 = 1.000" in printed
    assert "1/1 full hit, 1 unlabeled" in printed
    assert "HIT" in printed
    assert "hsg150_0090" in printed
    assert "--" in printed
    assert "expected  (none)" in printed
    assert "category" not in printed
    assert str(output.resolve()) in printed or "Wrote " in printed

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["k"] == 5
    assert saved["recall_at_k"] == 1.0
    assert saved["scored"] == 1
    assert saved["full_hits"] == 1
    assert saved["unlabeled"] == 1
    assert saved["results_table"][0]["retrieved"] == ["hsg150_0090"]
    assert saved["results_table"][1]["recall"] is None
    assert "category" not in saved["results_table"][0]


def test_run_eval_missing_file_does_not_load_the_retriever(monkeypatch, capsys):
    monkeypatch.setattr(
        run_eval,
        "_retriever",
        lambda config: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert run_eval.main(["does-not-exist.json"]) == 1
    assert "not found" in capsys.readouterr().err
