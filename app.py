"""Streamlit demo entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import Config

st.set_page_config(page_title="RAG Construction Assistant", layout="wide")
st.title("RAG Construction Assistant")
st.caption("Ask questions over UK building regulations and HSE construction guidance.")

config = Config.from_yaml(ROOT / "config.yaml")
question = st.text_input("Question")

if st.button("Ask") and question:
    st.info("Pipeline not implemented yet.")
    st.write(f"Would retrieve top {config.retrieval.k} chunks, then call {config.generation.model_name}.")
