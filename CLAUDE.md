# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

No test suite or linter is configured.

## Architecture

Single-file Streamlit app (`app.py`) implementing a RAG pipeline for Q&A over text-based PDFs.

**Pipeline flow:**

```
PDF Upload → Text Extraction (PyMuPDF) → Scanned PDF Guard → Chunking (LangChain RecursiveCharacterTextSplitter)
→ Dual Indexing: FAISS (semantic) + BM25 (lexical)
→ User Query → Query Rewriting (LLM) → Hybrid Search (60% semantic + 40% BM25)
→ CrossEncoder Reranking → Answer Generation (Groq/Llama 3.1 8B) → Display with page citations
```

**Key configuration constants** (top of `app.py`):
- `CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 200`
- `EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"` (downloads ~1.3 GB on first run)
- `RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `GROQ_MODEL = "llama-3.1-8b-instant"`

**External dependencies:**
- Groq API key required — set via `.env` (`GROQ_API_KEY=...`) or the sidebar input in the UI
- All models run on CPU (`device="cpu"`)
- FAISS and BM25 indexes are built in-memory per session (no persistence)
