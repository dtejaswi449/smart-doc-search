"""
api.py — FastAPI metrics and health server for SmartDoc.

Exposes real-time stats from query_logs.db alongside the Streamlit UI.

Install extra deps (not in the main requirements.txt):
    pip install fastapi uvicorn

Run (in a second terminal while Streamlit is running):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET /health   → service status, uptime, model info
GET /metrics  → aggregate query stats, corpus stats, last 5 queries
GET /queries  → paginated query log  (?limit=20&offset=0)
"""

import sys
import time

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import JSONResponse
except ImportError:
    print(
        "FastAPI not installed. Run:  pip install fastapi uvicorn",
        file=sys.stderr,
    )
    sys.exit(1)

from logger import QueryLogger

# ---------- App ----------
app = FastAPI(
    title="SmartDoc Metrics API",
    description="Real-time observability for the SmartDoc RAG pipeline.",
    version="1.0.0",
)

_start_time = time.time()
_logger = QueryLogger()

MODEL_INFO = {
    "embeddings": "BAAI/bge-large-en-v1.5",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm": "llama-3.1-8b-instant (Groq)",
    "retrieval": "Hybrid FAISS (60%) + BM25 (40%) → CrossEncoder rerank",
}


# ---------- Routes ----------
@app.get("/health", summary="Service health check")
def health():
    """Returns uptime, model configuration, and database path."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "db_path": _logger.db_path,
        "models": MODEL_INFO,
    }


@app.get("/metrics", summary="Aggregate pipeline metrics")
def metrics():
    """
    Returns:
    - **queries**: aggregate stats over all logged queries
      (count, mean confidence, mean faithfulness, mean latency, reflection count)
    - **corpus**: total PDFs indexed, pages, chunks
    - **recent_queries**: last 5 queries with key metrics
    """
    q_stats = _logger.stats()
    c_stats = _logger.corpus_stats()
    recent = _logger.recent(n=5)

    # Round floats for cleaner JSON
    def _fmt(v):
        return round(v, 3) if isinstance(v, float) else v

    return {
        "queries": {k: _fmt(v) for k, v in q_stats.items()},
        "corpus": c_stats,
        "recent_queries": [
            {
                "id": r["id"],
                "ts": r["ts"],
                "question": r["question"],
                "faithfulness_label": r["faithfulness_label"],
                "avg_confidence": _fmt(r["avg_confidence"]),
                "latency_ms": _fmt(r["latency_ms"]),
                "was_reflected": bool(r["was_reflected"]),
            }
            for r in recent
        ],
    }


@app.get("/queries", summary="Paginated query log")
def queries(
    limit: int = Query(default=20, ge=1, le=100, description="Rows to return"),
    offset: int = Query(default=0, ge=0, description="Skip first N rows"),
):
    """Returns recent query log rows, newest first."""
    import sqlite3

    with sqlite3.connect(_logger.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM query_logs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    total = _logger.stats().get("total_queries") or 0
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": [dict(r) for r in rows],
    }
