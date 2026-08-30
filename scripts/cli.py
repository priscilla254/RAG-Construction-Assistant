"""Command-line Q&A test harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the construction RAG assistant.")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    args = parser.parse_args()

    config = Config.from_yaml(ROOT / "config.yaml")
    question = args.question or input("Question: ").strip()
    if not question:
        print("No question provided.")
        return

    print(f"Loaded config (k={config.retrieval.k}, model={config.generation.model_name}).")
    print("Pipeline not implemented yet.")
    print(f"Q: {question}")


if __name__ == "__main__":
    main()
