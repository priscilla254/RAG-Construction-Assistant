"""Ask a construction question from the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.generator import _passage_label  # noqa: E402
from rag_assistant.pipeline import RetrievalPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the construction RAG assistant.")
    parser.add_argument(
        "question",
        nargs="*",
        help="One question (words are joined). Omit to type interactively.",
    )
    parser.add_argument(
        "-q",
        "--question",
        dest="questions",
        action="append",
        default=[],
        metavar="TEXT",
        help="A question. Repeat for a batch: -q '...' -q '...'",
    )
    parser.add_argument(
        "--file",
        dest="question_file",
        metavar="PATH",
        help="Text file with one question per line. Blank lines and # comments are skipped.",
    )
    parser.add_argument(
        "--show-all-sources",
        action="store_true",
        help="List every retrieved passage, including ones the answer did not cite.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks and RRF scores, and list every retrieved source.",
    )
    args = parser.parse_args(argv)

    try:
        batch = _collect_questions(args)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    config = Config.from_yaml(ROOT / "config.yaml")
    print(
        f"Loading retriever and {config.generation.model_name} "
        f"(k={config.retrieval.k})...",
        flush=True,
    )
    pipeline = RetrievalPipeline.from_config(config)
    show_all = True if (args.show_all_sources or args.debug) else None

    if batch:
        total = len(batch)
        for index, question in enumerate(batch, start=1):
            print("=" * 80)
            print(f"=== {index}/{total} ===")
            print(question)
            _print_turn(pipeline, question, show_all, debug=args.debug)
        return 0

    print("Ask a construction question (empty line to exit).")
    while True:
        try:
            question = input("Question: ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            return 0
        _print_turn(pipeline, question, show_all, debug=args.debug)


def _collect_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.question_file:
        questions.extend(_questions_from_file(Path(args.question_file)))
    questions.extend(item.strip() for item in args.questions if item.strip())
    positional = " ".join(args.question).strip()
    if positional:
        questions.append(positional)
    return questions


def _questions_from_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Question file not found: {path}")
    questions: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        questions.append(line)
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def _print_turn(
    pipeline: RetrievalPipeline,
    question: str,
    show_all_sources: bool | None,
    debug: bool = False,
) -> None:
    result = pipeline.answer(question, show_all_sources=show_all_sources)
    print()
    print(result["answer"] or "(no answer)")
    print()
    print("Sources")
    if not result["sources"]:
        print("  (none)")
    else:
        for source in result["sources"]:
            print(f"  [{source['n']}] {source['label']}")
    if debug:
        _print_retrieved(result.get("retrieved_chunks") or [])
    print()


def _print_retrieved(chunks: list[dict]) -> None:
    print()
    print(f"Retrieved (k={len(chunks)})")
    if not chunks:
        print("  (none)")
        return
    for rank, chunk in enumerate(chunks, start=1):
        score = chunk.get("rrf_score")
        score_bit = f"  rrf={score:.4f}" if isinstance(score, (int, float)) else ""
        chunk_id = chunk.get("chunk_id") or "?"
        print(f"  [{rank}] {chunk_id}{score_bit}")
        heading = _passage_label(chunk)
        if heading:
            print(f"      {heading}")
        preview = " ".join((chunk.get("text") or "").split())[:160]
        if preview:
            print(f"      {preview}")


if __name__ == "__main__":
    raise SystemExit(main())
