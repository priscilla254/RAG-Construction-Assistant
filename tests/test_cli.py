import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cli  # noqa: E402


class _FakePipeline:
    def __init__(self) -> None:
        self.questions: list[tuple[str, bool | None]] = []

    @classmethod
    def from_config(cls, config):
        return cls()

    def answer(self, question, show_all_sources=None):
        self.questions.append((question, show_all_sources))
        return {
            "answer": "Barriers must be at least 950 mm high. [1]",
            "sources": [
                {
                    "n": 1,
                    "label": "HSG150 — paragraph 151",
                    "chunk_id": "hsg150_0090",
                }
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": "hsg150_0090",
                    "text": "Barriers other than guard rails can be used if they are at least 950 mm high.",
                    "context_prefix": "HSG150 — paragraphs 150-151",
                    "rrf_score": 0.0123,
                }
            ],
        }


def test_cli_prints_answer_and_sources(monkeypatch, capsys):
    pipeline = _FakePipeline()
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: pipeline)

    assert cli.main(["How high should scaffold barriers be?"]) == 0

    output = capsys.readouterr().out
    assert "Barriers must be at least 950 mm high. [1]" in output
    assert "[1] HSG150 — paragraph 151" in output
    assert pipeline.questions == [("How high should scaffold barriers be?", None)]


def test_cli_show_all_sources_flag(monkeypatch, capsys):
    pipeline = _FakePipeline()
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: pipeline)

    cli.main(["--show-all-sources", "When must a site be notified?"])

    assert pipeline.questions == [("When must a site be notified?", True)]
    assert "Sources" in capsys.readouterr().out


def test_cli_debug_prints_retrieved_chunks_and_scores(monkeypatch, capsys):
    pipeline = _FakePipeline()
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: pipeline)

    cli.main(["--debug", "How high should scaffold barriers be?"])

    output = capsys.readouterr().out
    assert pipeline.questions == [("How high should scaffold barriers be?", True)]
    assert "Retrieved (k=1)" in output
    assert "hsg150_0090" in output
    assert "rrf=0.0123" in output
    assert "950 mm" in output
    assert "HSG150 — paragraphs 150-151" in output


def test_cli_empty_question_exits_without_calling_the_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: _FakePipeline())
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert cli.main([]) == 0
    assert "950 mm" not in capsys.readouterr().out


def test_cli_repeatable_q_runs_each_question(monkeypatch, capsys):
    pipeline = _FakePipeline()
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: pipeline)

    assert (
        cli.main(
            [
                "-q",
                "How high should scaffold barriers be?",
                "-q",
                "When must a site be notified?",
            ]
        )
        == 0
    )

    assert pipeline.questions == [
        ("How high should scaffold barriers be?", None),
        ("When must a site be notified?", None),
    ]
    output = capsys.readouterr().out
    assert "=== 1/2 ===" in output
    assert "=== 2/2 ===" in output
    assert "When must a site be notified?" in output


def test_cli_file_skips_blank_lines_and_comments(monkeypatch, tmp_path, capsys):
    pipeline = _FakePipeline()
    monkeypatch.setattr(cli.RetrievalPipeline, "from_config", lambda config: pipeline)
    path = tmp_path / "questions.txt"
    path.write_text(
        "# HSG150\n"
        "How high should scaffold barriers be?\n"
        "\n"
        "When must a site be notified?\n",
        encoding="utf-8",
    )

    assert cli.main(["--file", str(path)]) == 0
    assert [question for question, _ in pipeline.questions] == [
        "How high should scaffold barriers be?",
        "When must a site be notified?",
    ]
    assert "=== 1/2 ===" in capsys.readouterr().out


def test_cli_missing_question_file_does_not_load_the_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.RetrievalPipeline,
        "from_config",
        lambda config: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert cli.main(["--file", "does-not-exist.txt"]) == 1
    assert "not found" in capsys.readouterr().err
