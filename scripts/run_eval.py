"""Run recall@k on eval/eval_set.json (retrieval only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from evaluator import Evaluator  # noqa: E402
from rag_assistant.chunker import load_chunks  # noqa: E402
from rag_assistant.config import Config  # noqa: E402
from rag_assistant.embedder import Embedder  # noqa: E402
from rag_assistant.keyword_search import KeywordSearch  # noqa: E402
from rag_assistant.retriever import HybridRetriever  # noqa: E402
from rag_assistant.vector_store import VectorStore  # noqa: E402

DEFAULT_EVAL_SET = ROOT / "eval" / "eval_set.json"
DEFAULT_OUTPUT = ROOT / "eval" / "results.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score hybrid retrieval against the labeled eval set (recall@k)."
    )
    parser.add_argument(
        "eval_set",
        nargs="?",
        default=str(DEFAULT_EVAL_SET),
        help="JSON eval set (default: eval/eval_set.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="N",
        help="Top-k to retrieve (default: retrieval.k in config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="PATH",
        help="JSON report path (default: eval/results.json)",
    )
    args = parser.parse_args(argv)

    eval_path = Path(args.eval_set)
    if not eval_path.is_file():
        print(f"Eval set not found: {eval_path}", file=sys.stderr)
        return 1

    config = Config.from_yaml(ROOT / "config.yaml")
    k = args.k if args.k is not None else config.retrieval.k
    print(f"Loading retriever (k={k})...", flush=True)
    evaluator = Evaluator(_retriever(config), k=k)
    result = evaluator.run(eval_path)
    report = _report_payload(result, k=k, eval_set=eval_path)
    output_path = _write_report(report, Path(args.output))
    _print_report(report, k=k)
    print(f"Wrote {output_path}")
    return 0


def _retriever(config: Config) -> HybridRetriever:
    embedder = Embedder.from_config(config)
    return HybridRetriever.from_config(
        config,
        vector_store=VectorStore(config.paths.chroma_dir),
        keyword_search=KeywordSearch(load_chunks(config.paths.chunks_path)),
        embedder=embedder,
    )


def _report_payload(result: dict, k: int, eval_set: Path) -> dict:
    rows = result.get("results_table") or []
    scored = [row for row in rows if row.get("recall") is not None]
    return {
        "k": k,
        "eval_set": str(eval_set),
        "recall_at_k": result.get("recall_at_k", 0.0),
        "scored": len(scored),
        "full_hits": sum(1 for row in scored if row.get("hit")),
        "unlabeled": len(rows) - len(scored),
        "results_table": rows,
    }


def _write_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _print_report(result: dict, k: int) -> None:
    rows = result.get("results_table") or []
    scored = [row for row in rows if row.get("recall") is not None]
    full_hits = sum(1 for row in scored if row.get("hit"))
    unlabeled = len(rows) - len(scored)
    print()
    print(
        f"recall@{k} = {result['recall_at_k']:.3f}  "
        f"({full_hits}/{len(scored)} full hit"
        + (f", {unlabeled} unlabeled" if unlabeled else "")
        + ")"
    )
    print()
    for index, row in enumerate(rows, start=1):
        _print_row(index, row)


def _print_row(index: int, row: dict) -> None:
    recall = row.get("recall")
    if recall is None:
        mark = "  -- "
        score = "n/a "
    elif row.get("hit"):
        mark = " HIT "
        score = f"{recall:.2f}"
    else:
        mark = "MISS "
        score = f"{recall:.2f}"
    question = _one_line(row.get("question") or "", 88)
    print(f"{mark} {score}  {index}. {question}")
    print(f"         expected  {_format_expected(row.get('expected_source'))}")
    retrieved = row.get("retrieved") or []
    print(f"         retrieved {', '.join(retrieved) if retrieved else '(none)'}")


def _format_expected(source) -> str:
    if source is None:
        return "(none)"
    if isinstance(source, list):
        return ", ".join(str(item) for item in source)
    return str(source)


def _one_line(text: str, width: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
