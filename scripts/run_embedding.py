"""One-off: data/chunks.jsonl -> data/chroma_db (full rebuild)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.vector_store import index_corpus  # noqa: E402


def main() -> None:
    config = Config.from_yaml(ROOT / "config.yaml")
    index_corpus(config)


if __name__ == "__main__":
    main()
