"""One-off: raw PDFs -> cleaned per-page JSON in data/cleaned/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.ingest import ingest_corpus  # noqa: E402


def main() -> None:
    config = Config.from_yaml(ROOT / "config.yaml")
    written = ingest_corpus(config)
    print(f"Cleaned {len(written)} document(s) into {config.paths.cleaned_dir}")


if __name__ == "__main__":
    main()
