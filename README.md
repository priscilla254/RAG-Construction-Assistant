# RAG Construction Assistant

A retrieval-augmented generation assistant for UK construction safety and building-regulation documents:

- The Merged Approved Documents (Oct 2024)
- HSG141 — Electrical safety on construction sites
- HSG150 — Health and safety in construction



## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Add your Groq key to `.env` when you start generation. Put the original PDFs in `data/raw/` (gitignored).

## Pipeline

```
data/raw/*.pdf
    → scripts/run_ingestion.py     cleaned JSON in data/cleaned/
    → scripts/run_chunking.py      data/chunks.jsonl
    → scripts/run_embedding.py     data/chroma_db (BGE-small, 384-d)
    → HybridRetriever              vector + BM25, reciprocal rank fusion
```

Re-run embedding after any chunking change. The vector store is always rebuilt from scratch and tagged with the model name; querying with a different model raises `ModelMismatchError`.

```powershell
python scripts/run_ingestion.py
python scripts/run_chunking.py
python scripts/run_embedding.py
python scripts/spot_check_retrieval.py
python scripts/cli.py "How high should scaffold guard rails be?"
python scripts/cli.py --debug "How high should scaffold guard rails be?"
python scripts/cli.py --file scripts/batch_questions.txt
python -m pytest tests
```

## Chunking

Each document has its own strategy in `src/rag_assistant/`:

| Document | `doc_type` | Split | Notes |
|---|---|---|---|
| Approved Documents | `regulatory` | clause numbers (`3.24`, Part A's `2D1`) | Short clauses merge; long ones window. Index pages dropped. |
| HSG141 | `procedural_guidance` | numbered paragraphs 1–87 | Case-study boxes become their own chunks. |
| HSG150 | `broad_reference` | numbered paragraphs | Section/topic from the ToC; subheadings are metadata, not boundaries. |

Every structural unit is passed through `headings.trim_trailing_headings` so the next heading does not bleed onto the previous chunk. HSG141 and HSG150 also match titles from their contents pages (`documents.known_headings_for`).

What gets **embedded** is `Chunk.embedding_text` (context prefix + body). What gets **shown** is `Chunk.text` (body only).

## Retrieval

`HybridRetriever` runs both sides and fuses ranks:

```
score(d) = 0.7 / (60 + rank_vector) + 0.3 / (60 + rank_bm25)
```

Weights and `k` live in `config.yaml`. BM25 indexes the same `embedding_text` as the vectors. On the five smoke-test questions, vector-only was 4/5; hybrid is 5/5 (the miss was a Part K “rise and going” clause that BM25 recovered).

```powershell
python scripts/spot_check_retrieval.py
```

## Layout

```
src/rag_assistant/
  ingest.py              PDF extraction and cleaning
  chunker.py             strategies only
  hsg141.py / hsg150.py / approved_documents.py
  headings.py            trailing-heading detection
  watermark.py           Approved Documents ONLINE VERSION strip
  documents.py           filename → known headings
  embedder.py            BAAI/bge-small-en-v1.5
  vector_store.py        Chroma wrapper
  keyword_search.py      BM25Okapi
  retriever.py           HybridRetriever (RRF)
  generator.py           Groq, IEEE citations
  pipeline.py            RetrievalPipeline.answer
  config.py              reads config.yaml

scripts/
  run_ingestion.py / run_chunking.py / run_embedding.py
  spot_check_cleaned.py / spot_check_chunks.py / spot_check_retrieval.py
  cli.py                 ask one or many questions (`--file`, `-q`, `--debug`)

data/                    raw PDFs, cleaned JSON, chunks.jsonl, chroma_db
config.yaml              paths, chunk size, embedding model, retrieval weights
```

## Config

```yaml
embedding:
  model_name: "BAAI/bge-small-en-v1.5"
  batch_size: 64
retrieval:
  k: 5
  vector_weight: 0.7
  keyword_weight: 0.3
generation:
  show_all_sources: true   # debug: every retrieved passage. Streamlit defaults to cited-only.
```

Changing `embedding.model_name` means rebuilding `data/chroma_db`.
