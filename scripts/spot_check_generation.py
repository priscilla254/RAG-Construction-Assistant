"""
Ask the 5 smoke-test questions and check answers stay inside retrieved text.

Uses HybridRetriever + Generator. Flags measurements and clause-like
tokens in the answer that do not appear in any retrieved passage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.pipeline import RetrievalPipeline  # noqa: E402

from spot_check_retrieval import CHECKS  # noqa: E402

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:mm|m|V|kV|days?))?\b",
    re.I,
)
CITATION_RE = re.compile(r"\[(\d+)\]")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _grounding_flags(answer: str, passages: list[str]) -> list[str]:
    blob = _normalise(" ".join(passages))
    flags: list[str] = []
    for match in NUMBER_RE.finditer(answer):
        token = _normalise(match.group(0))
        if token not in blob and token.rstrip("s") not in blob:
            flags.append(f"ungrounded number {match.group(0)!r}")
    cited = {int(n) for n in CITATION_RE.findall(answer)}
    if not cited:
        flags.append("no [n] citations")
    for number in cited:
        if number < 1 or number > len(passages):
            flags.append(f"citation [{number}] is outside 1-{len(passages)}")
    return flags


def main() -> None:
    config = Config.from_yaml(ROOT / "config.yaml")
    pipeline = RetrievalPipeline.from_config(config)

    grounded = 0
    for index, check in enumerate(CHECKS, start=1):
        result = pipeline.answer(check["question"])
        chunks = result["retrieved_chunks"]
        answer = result["answer"]
        passages = [chunk.get("text") or "" for chunk in chunks]
        flags = _grounding_flags(answer, passages)

        print("=" * 80)
        print(f"Q{index}: {check['question']}")
        print("Retrieved:")
        for rank, chunk in enumerate(chunks, start=1):
            preview = " ".join((chunk.get("text") or "").split())[:140]
            print(f"  [{rank}] {chunk['chunk_id']}  {chunk.get('section', '')}")
            print(f"      {preview}")
        print("Answer:")
        print(answer)
        if flags:
            print(f"FLAGS: {'; '.join(flags)}")
        else:
            print("FLAGS: none — numbers and citations sit in the retrieved passages")
            grounded += 1
        print()

    print(f"{grounded}/{len(CHECKS)} answers had no automatic grounding flags")


if __name__ == "__main__":
    main()
