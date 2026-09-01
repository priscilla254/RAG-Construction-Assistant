"""Streamlit demo: ask a question, show the grounded answer and sources.

The pipeline (Chroma + BM25 + Groq) is cached so a page rerun does not
reload the index. Gallery clicks write into the same `question` widget
state as typing, then collapse the expander so the answer is visible.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config
from rag_assistant.pipeline import RetrievalPipeline

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_SQLITE = ROOT / "data" / "chroma_db" / "chroma.sqlite3"


def _apply_cloud_secrets() -> None:
    """Copy Streamlit Cloud secrets into the env vars the generator reads."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return
    if key:
        os.environ["GROQ_API_KEY"] = str(key).strip()


def _index_ready() -> bool:
    return (
        CHUNKS_PATH.is_file()
        and CHUNKS_PATH.stat().st_size > 0
        and CHROMA_SQLITE.is_file()
    )

# Labels for eval_set.json `category` values (human-only; not scored).
GALLERY_LABELS = {
    1: "HSG150 — site health and safety",
    2: "HSG141 — electrical safety",
    3: "Approved Documents — dwellings",
    4: "More than one document",
    5: "Not in these documents",
}

st.set_page_config(page_title="RAG Construction Assistant", layout="wide")
st.title("RAG Construction Assistant")
st.caption("Ask questions over UK building regulations and HSE construction guidance.")

_apply_cloud_secrets()

if not _index_ready():
    st.error(
        "The retrieval index is not in this clone (`data/chunks.jsonl` and "
        "`data/chroma_db/`). Commit those two paths (not the original PDFs) "
        "and reboot the app. See the README deploy section."
    )
    st.stop()

if not os.environ.get("GROQ_API_KEY", "").strip():
    st.error(
        "GROQ_API_KEY is not set. Locally use a `.env` file. On Streamlit "
        "Cloud add it under App settings → Secrets."
    )
    st.stop()


@st.cache_resource
def _pipeline() -> RetrievalPipeline:
    return RetrievalPipeline.from_config(Config.from_yaml(ROOT / "config.yaml"))


@st.cache_data
def _gallery() -> list[tuple[str, list[str]]]:
    """Sample questions grouped in eval-set order."""
    path = ROOT / "eval" / "eval_set.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, list[str]] = {}
    order: list[int] = []
    for item in items:
        category = int(item.get("category") or 0)
        question = (item.get("question") or "").strip()
        if not question:
            continue
        if category not in grouped:
            order.append(category)
            grouped[category] = []
        grouped[category].append(question)
    return [
        (GALLERY_LABELS.get(category, f"Category {category}"), grouped[category])
        for category in order
    ]


def _queue_question(question: str) -> None:
    """Fill the question box, request a run, and remount the gallery closed.

    Streamlit 1.37 expanders have no `key`, so bumping gallery_version
    changes the label (zero-width spaces) and forces a collapsed remount.
    """
    st.session_state.question = question
    st.session_state.pending_ask = True
    st.session_state.gallery_version = st.session_state.get("gallery_version", 0) + 1


if "question" not in st.session_state:
    st.session_state.question = ""
if "gallery_version" not in st.session_state:
    st.session_state.gallery_version = 0

st.text_input("Question", key="question")
ask_clicked = st.button("Ask")

if ask_clicked or st.session_state.get("pending_ask"):
    st.session_state.pending_ask = False
    question = st.session_state.question.strip()
    if not question:
        st.warning("Enter a question.")
        st.session_state.result = None
    else:
        with st.spinner("Retrieving and generating..."):
            # Cited [n] only — unused retrieved passages stay off the demo.
            st.session_state.result = _pipeline().answer(
                question,
                show_all_sources=False,
            )

result = st.session_state.get("result")
if result:
    st.write(result["answer"] or "(no answer)")
    st.subheader("Sources")
    if not result["sources"]:
        st.caption("(none)")
    else:
        for source in result["sources"]:
            st.markdown(f"**[{source['n']}]** {source['label']}")

gallery_label = "Test gallery" + ("\u200b" * st.session_state.gallery_version)
with st.expander(gallery_label, expanded=False):
    st.caption("Click a question to put it in the box and run it.")
    for group_index, (group, questions) in enumerate(_gallery()):
        st.markdown(f"**{group}**")
        for item_index, sample in enumerate(questions):
            st.button(
                sample,
                key=f"gallery-{group_index}-{item_index}",
                on_click=_queue_question,
                args=(sample,),
                use_container_width=True,
            )
