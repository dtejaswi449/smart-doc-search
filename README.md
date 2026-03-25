# SmartDoc — Agentic Document Q&A

A production-grade RAG pipeline that turns any text-based PDF into a searchable, question-answering system. Built with a LangChain ReAct agent, hybrid retrieval (FAISS + BM25), CrossEncoder reranking, self-reflection, and a full observability stack.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  PDF Upload ──► PyMuPDF ──► RecursiveTextSplitter               │
│                              (800 tok / 200 overlap)            │
│                                    │                            │
│                    ┌───────────────┴───────────────┐            │
│                    ▼                               ▼            │
│           FAISS Vector Index              BM25 Keyword Index    │
│         (BAAI/bge-large-en-v1.5)           (rank-bm25)          │
└─────────────────────────────────────────────────────────────────┘

                         User Query
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REACT AGENT LOOP                             │
│              (LangChain AgentExecutor, max_iterations=6)        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Thought → pick tool → observe → Thought → ...          │   │
│  │                                                         │   │
│  │  Tool 1: semantic_search   FAISS similarity             │   │
│  │          ──────────────►  + CrossEncoder rerank         │   │
│  │                                                         │   │
│  │  Tool 2: keyword_search    BM25 top-k                   │   │
│  │          ──────────────►  + CrossEncoder rerank         │   │
│  │                                                         │   │
│  │  Tool 3: summarize_document  LLM over first 4k chars    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                 ┌────────────▼────────────┐                     │
│                 │    Self-Reflection       │                     │
│                 │  weak answer detected?  │                     │
│                 │  → LLM reformulates     │                     │
│                 │  → agent re-runs once   │                     │
│                 └────────────┬────────────┘                     │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               ▼
                 Answer + Confidence + Faithfulness
                               │
               ┌───────────────▼───────────────┐
               │        SQLite Logger           │
               │       (query_logs.db)          │──► FastAPI /metrics
               │  query_logs + corpus_stats     │    (api.py :8000)
               └───────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Hybrid FAISS + BM25 (60/40 blend) | Semantic search misses exact terms; BM25 misses paraphrases. Blending covers both. |
| CrossEncoder reranking | Bi-encoder retrieval is fast but imprecise. CrossEncoder reads query+chunk jointly for a precise relevance score. |
| ReAct agent over fixed pipeline | Lets the LLM choose which retrieval strategy fits the question, and retry with a different tool on failure. |
| Self-reflection on weak answers | Detects "I could not find…" class answers and runs a second LLM call to reformulate before giving up. |
| Groq / Llama-3.1-8b-instant | Free-tier API, ~500–1500ms generation latency, sufficient reasoning for ReAct. |

---

## Features

- **Hybrid retrieval** — FAISS semantic + BM25 keyword, weighted 60/40
- **CrossEncoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores top candidates
- **ReAct agent** — LLM reasons over 3 tools, chains calls as needed
- **Self-reflection** — Detects weak answers, reformulates query, re-runs
- **Scanned PDF guard** — Rejects image-only PDFs before indexing
- **Faithfulness scoring** — Token-overlap groundedness check per answer
- **Retrieval confidence** — Sigmoid-normalised CrossEncoder score (0–1)
- **End-to-end latency tracking** — Logged per query; displayed in UI and API
- **Corpus tracking** — Counts PDFs, pages, and chunks indexed
- **FastAPI metrics server** — `/health`, `/metrics`, `/queries` endpoints
- **Offline eval script** — Batch scoring against any PDF + query file

---

## Setup

```bash
# 1. Clone and create a virtual environment
git clone <repo-url>
cd smart-doc-search
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=gsk_..." > .env

# 4. Run the Streamlit app
streamlit run app.py
```

> **First run:** The BGE embedding model (~1.3 GB) downloads automatically on first use.

---

## Running the Metrics API

The FastAPI server runs independently alongside the Streamlit app:

```bash
# Install extra deps (one-time)
pip install fastapi uvicorn

# Start the metrics server (second terminal)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Description |
|---|---|
| `GET /health` | Service uptime, model config |
| `GET /metrics` | Aggregate stats + corpus info + last 5 queries |
| `GET /queries?limit=20&offset=0` | Paginated query log |

Example response from `/metrics`:
```json
{
  "queries": {
    "total_queries": 42,
    "mean_confidence": 0.731,
    "mean_faithfulness": 0.684,
    "mean_latency_ms": 5820,
    "reflection_count": 6
  },
  "corpus": {
    "total_documents": 3,
    "total_pages": 87,
    "total_chunks": 312
  }
}
```

---

## Offline Evaluation

```bash
# Run built-in generic queries against any PDF
python eval.py --pdf path/to/document.pdf

# Run with a custom query file
python eval.py --pdf path/to/document.pdf --queries eval_queries.json

# Print a report of the last 20 logged queries
python eval.py --report

# Show last 50 entries
python eval.py --report --n 50
```

**`eval_queries.json` format:**
```json
[
  {
    "question": "What are the main contributions of this paper?",
    "expected_keywords": ["contribution", "novel", "proposed"]
  },
  {
    "question": "What dataset was used for evaluation?",
    "expected_keywords": []
  }
]
```

### Sample eval output

```
==============================================================
  SmartDoc Eval  —  2026-03-24 14:30
  PDF:     arxiv_paper.pdf
  Queries: 4
==============================================================

[1/4] What is this document about?
  Retrieval confidence : 0.782
  Faithfulness         : 0.714  [high]
  Keyword hit rate     : N/A (no expected_keywords)
  Latency              : 4320ms
  Answer               : The document presents a novel framework for...

[2/4] What are the main findings?
  Retrieval confidence : 0.691
  Faithfulness         : 0.638  [high]
  Keyword hit rate     : N/A (no expected_keywords)
  Latency              : 3890ms
  Answer               : The main findings include...

==============================================================
  SUMMARY
==============================================================
  Mean retrieval confidence : 0.724
  Mean faithfulness score   : 0.671
  Latency  p50=4105ms  p95=6820ms  mean=4680ms
  Logged 4 entries → query_logs.db
```

---

## Latency Profile

Measured on Apple M-series CPU (no GPU), processing a 20-page PDF (~80 chunks).

### Indexing (one-time per PDF upload)

| Stage | Time |
|---|---|
| Text extraction (PyMuPDF) | ~0.3s per page |
| Chunking (RecursiveTextSplitter) | < 0.1s for 100 chunks |
| Embedding — first run (model download) | ~60–120s (1.3 GB download) |
| Embedding — subsequent runs (cached) | ~10–30s for 80 chunks |
| BM25 index build | < 0.05s |

### Query latency breakdown (per query, CPU)

| Stage | Typical |
|---|---|
| FAISS similarity search (k=6) | ~50–100ms |
| BM25 scoring | ~5–10ms |
| CrossEncoder reranking (3 pairs) | ~300–800ms |
| Groq LLM call (Llama-3.1-8b) | ~500–1500ms |
| Agent reasoning overhead per step | ~200–400ms |

### End-to-end query latency (wall clock)

| Scenario | p50 | p95 |
|---|---|---|
| **Baseline** — linear pipeline (pre-agent) | ~2.1s | ~3.8s |
| **Agent** — 1 tool call | ~3.5s | ~5.2s |
| **Agent** — 2 tool calls | ~5.8s | ~8.4s |
| **Agent + self-reflection** (worst case) | ~9.2s | ~14.5s |

**Trade-off:** The ReAct agent adds ~1.5–3s per tool call vs. the linear baseline due to LLM reasoning at each step. Self-reflection triggers on ~15% of queries and resolves ~70% of weak answers, making the extra latency worth the correctness gain on hard questions.

---

## Corpus Metrics

The sidebar and `/metrics` endpoint track cumulative stats across all indexed PDFs:

```
Corpus: 3 PDF(s) · 87 pages · 312 chunks
```

Recorded in `query_logs.db → corpus_stats` every time a PDF is indexed.

---

## Configuration

All tunable constants live in the `# ---------- CONFIG ----------` block at the top of `app.py`:

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk. Smaller = more precise retrieval, more chunks. |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks to preserve context at boundaries. |
| `EMBEDDINGS_MODEL` | `BAAI/bge-large-en-v1.5` | Swap for `BAAI/bge-small-en-v1.5` (~130 MB) for faster indexing at slight quality cost. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any MS MARCO cross-encoder works here. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Swap for `llama-3.3-70b-versatile` for better reasoning at higher latency. |

---

## Project Structure

```
smart-doc-search/
├── app.py              # Streamlit UI + ReAct agent pipeline
├── api.py              # FastAPI metrics server (/health, /metrics, /queries)
├── logger.py           # SQLite-backed query + corpus logger
├── eval.py             # Offline batch evaluation CLI
├── requirements.txt    # Pinned Python dependencies
├── .env                # GROQ_API_KEY (not committed)
└── query_logs.db       # Auto-created SQLite database (not committed)
```

---

## Tech Stack

| Component | Library / Model |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | `BAAI/bge-large-en-v1.5` via HuggingFace |
| Vector store | FAISS (CPU) |
| Keyword search | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent framework | LangChain ReAct + AgentExecutor |
| LLM | Groq API — Llama-3.1-8b-instant |
| Monitoring DB | SQLite (query_logs.db) |
| Metrics API | FastAPI + Uvicorn |
