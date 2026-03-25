"""
logger.py — SQLite-backed query logger for SmartDoc.

Tables
------
query_logs   : one row per user query (answer, metrics, latency)
corpus_stats : one row per indexed PDF (filename, pages, chunks)
"""
import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "query_logs.db"


class QueryLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                  TEXT    NOT NULL,
                    question            TEXT    NOT NULL,
                    final_answer        TEXT,
                    was_reflected       INTEGER DEFAULT 0,
                    reflected_query     TEXT,
                    tool_calls          TEXT,
                    avg_confidence      REAL,
                    faithfulness_score  REAL,
                    faithfulness_label  TEXT,
                    latency_ms          REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corpus_stats (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    filename    TEXT    NOT NULL,
                    page_count  INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
            """)
            # Migrate existing query_logs tables that pre-date the latency_ms column
            try:
                conn.execute("ALTER TABLE query_logs ADD COLUMN latency_ms REAL")
            except sqlite3.OperationalError:
                pass  # column already exists

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def log(
        self,
        *,
        question: str,
        final_answer: str,
        was_reflected: bool = False,
        reflected_query: str | None = None,
        tool_calls: list | None = None,
        avg_confidence: float | None = None,
        faithfulness_score: float | None = None,
        faithfulness_label: str | None = None,
        latency_ms: float | None = None,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_logs
                    (ts, question, final_answer, was_reflected, reflected_query,
                     tool_calls, avg_confidence, faithfulness_score,
                     faithfulness_label, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    final_answer,
                    int(bool(was_reflected)),
                    reflected_query,
                    json.dumps(tool_calls or []),
                    avg_confidence,
                    faithfulness_score,
                    faithfulness_label,
                    latency_ms,
                ),
            )

    def log_corpus(self, filename: str, page_count: int, chunk_count: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO corpus_stats (ts, filename, page_count, chunk_count) VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), filename, page_count, chunk_count),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def recent(self, n: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                AS total_queries,
                    AVG(avg_confidence)     AS mean_confidence,
                    AVG(faithfulness_score) AS mean_faithfulness,
                    AVG(latency_ms)         AS mean_latency_ms,
                    SUM(was_reflected)      AS reflection_count
                FROM query_logs
            """).fetchone()
        keys = [
            "total_queries", "mean_confidence", "mean_faithfulness",
            "mean_latency_ms", "reflection_count",
        ]
        return dict(zip(keys, row)) if row else dict.fromkeys(keys, None)

    def corpus_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(DISTINCT filename) AS total_documents,
                    SUM(page_count)          AS total_pages,
                    SUM(chunk_count)         AS total_chunks
                FROM corpus_stats
            """).fetchone()
        keys = ["total_documents", "total_pages", "total_chunks"]
        return dict(zip(keys, row)) if row else dict.fromkeys(keys, 0)
