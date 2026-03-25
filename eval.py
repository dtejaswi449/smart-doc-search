#!/usr/bin/env python3
"""
eval.py — Offline RAG evaluation script for SmartDoc.

Runs test queries against a PDF, scores retrieval quality and answer
faithfulness, and writes results to query_logs.db.

Usage:
    python eval.py --pdf path/to/doc.pdf
    python eval.py --pdf path/to/doc.pdf --queries eval_queries.json
    python eval.py --pdf path/to/doc.pdf --queries eval_queries.json --groq-key gsk_...
    python eval.py --report          # print recent log report and exit
    python eval.py --report --n 50   # show last 50 entries

eval_queries.json format:
    [
      {
        "question": "What is this document about?",
        "expected_keywords": ["machine learning", "neural networks"]
      },
      ...
    ]
    "expected_keywords" is optional — omit or leave empty to skip keyword-hit scoring.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import fitz
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from logger import QueryLogger

load_dotenv()

# ---------- CONFIG (mirrors app.py) ----------
EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHARS_PER_PAGE = 100

# Used when no --queries file is provided
DEFAULT_TEST_QUERIES = [
    {"question": "What is this document about?", "expected_keywords": []},
    {"question": "What are the main topics covered?", "expected_keywords": []},
    {"question": "Summarize the key findings or conclusions.", "expected_keywords": []},
    {"question": "Who are the intended audience or stakeholders?", "expected_keywords": []},
]

# ---------- TEXT + INDEX HELPERS ----------
def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    lines = [l for l in text.split("\n") if len(l.strip()) > 3 or l.strip() == ""]
    return "\n".join(lines).strip()


def _load_pdf(pdf_path: str) -> tuple[list[dict], str]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = _clean_text(page.get_text("text"))
        pages.append({"text": txt, "page": i + 1})
    full_text = "\n\n".join(p["text"] for p in pages)
    return pages, full_text


def _make_chunks(pages: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )
    docs, idx = [], 0
    for p in pages:
        if not p["text"].strip():
            continue
        for chunk in splitter.split_text(p["text"]):
            docs.append(Document(
                page_content=chunk,
                metadata={"chunk": idx, "page": p["page"]},
            ))
            idx += 1
    return docs


def _build_indexes(docs: list[Document], embeddings):
    vectorstore = FAISS.from_documents(docs, embeddings)
    bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])
    return vectorstore, bm25


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _hybrid_search(query, vectorstore, bm25, all_docs, k=8) -> list[Document]:
    semantic_hits = vectorstore.similarity_search_with_score(query, k=k)
    max_s = max((s for _, s in semantic_hits), default=1e-9) + 1e-9
    combined = {
        d.metadata["chunk"]: {"doc": d, "score": (1 - s / max_s) * 0.6}
        for d, s in semantic_hits
    }
    bm25_scores = bm25.get_scores(query.lower().split())
    max_b = bm25_scores.max() + 1e-9
    for idx in np.argsort(bm25_scores)[::-1][:k]:
        if bm25_scores[idx] <= 0:
            continue
        doc = all_docs[idx]
        cid = doc.metadata["chunk"]
        contrib = (bm25_scores[idx] / max_b) * 0.4
        if cid in combined:
            combined[cid]["score"] += contrib
        else:
            combined[cid] = {"doc": doc, "score": contrib}
    return [x["doc"] for x in sorted(combined.values(), key=lambda v: v["score"], reverse=True)[:k]]


def _rerank_with_scores(query, docs, reranker, top_k=3) -> tuple[list[Document], list[float]]:
    if not docs:
        return [], []
    pairs = [(query, d.page_content) for d in docs]
    raw = reranker.predict(pairs)
    ranked = sorted(zip(raw, docs), key=lambda x: x[0], reverse=True)[:top_k]
    scores = [_sigmoid(float(s)) for s, _ in ranked]
    top_docs = [d for _, d in ranked]
    return top_docs, scores


# ---------- METRICS ----------
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "i", "you", "he", "she", "they", "we",
    "from", "by", "as", "about", "also", "which", "who", "what", "how", "when",
}


def faithfulness_score(answer: str, context: str) -> tuple[float, str]:
    """
    Token-overlap faithfulness: fraction of content words in the answer
    that appear in the retrieved context. Higher = more grounded.

    Returns (score 0-1, label: "high" / "medium" / "low")
    """
    a_tokens = set(re.findall(r"\b\w+\b", answer.lower())) - _STOP_WORDS
    c_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    if not a_tokens:
        return 0.0, "low"
    score = round(len(a_tokens & c_tokens) / len(a_tokens), 3)
    label = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"
    return score, label


def keyword_hit_rate(context: str, expected_keywords: list[str]) -> float | None:
    """Fraction of expected_keywords found anywhere in the retrieved context."""
    if not expected_keywords:
        return None
    lower_ctx = context.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ctx)
    return round(hits / len(expected_keywords), 3)


# ---------- ANSWER GENERATION ----------
def _generate_answer(question: str, context: str, llm) -> str:
    messages = [
        SystemMessage(content=(
            "You are a precise document assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say: "
            "'I could not find the answer in this document.' "
            "Cite page numbers when possible."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content.strip()


# ---------- MAIN EVAL LOOP ----------
def run_eval(pdf_path: str, queries: list[dict], groq_api_key: str, logger: QueryLogger):
    print(f"\n{'='*62}")
    print(f"  SmartDoc Eval  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  PDF:     {pdf_path}")
    print(f"  Queries: {len(queries)}")
    print(f"{'='*62}\n")

    print("Loading models (embedding + reranker)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    reranker = CrossEncoder(RERANKER_MODEL)
    llm = ChatGroq(model=GROQ_MODEL, groq_api_key=groq_api_key, temperature=0, max_tokens=512)

    print("Extracting text and building indexes...")
    pages, _ = _load_pdf(pdf_path)
    docs = _make_chunks(pages)
    vectorstore, bm25 = _build_indexes(docs, embeddings)
    print(f"Indexed {len(docs)} chunks.\n")

    results = []
    for i, case in enumerate(queries, 1):
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        print(f"[{i}/{len(queries)}] {question}")

        t0 = time.perf_counter()
        candidates = _hybrid_search(question, vectorstore, bm25, docs, k=8)
        top_docs, conf_scores = _rerank_with_scores(question, candidates, reranker, top_k=3)
        avg_conf = round(float(np.mean(conf_scores)), 3) if conf_scores else None

        retrieved_ctx = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in top_docs
        )
        answer = _generate_answer(question, retrieved_ctx, llm)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        faith, faith_label = faithfulness_score(answer, retrieved_ctx)
        kw_hit = keyword_hit_rate(retrieved_ctx, expected_keywords)

        # Print per-query results
        conf_str = f"{avg_conf:.3f}" if avg_conf is not None else "N/A"
        kw_str = f"{kw_hit:.3f}" if kw_hit is not None else "N/A (no expected_keywords)"
        print(f"  Retrieval confidence : {conf_str}")
        print(f"  Faithfulness         : {faith:.3f}  [{faith_label}]")
        print(f"  Keyword hit rate     : {kw_str}")
        print(f"  Latency              : {latency_ms}ms")
        print(f"  Answer               : {answer[:120]}{'...' if len(answer) > 120 else ''}\n")

        logger.log(
            question=question,
            final_answer=answer,
            was_reflected=False,
            tool_calls=[{
                "tool": "hybrid_search+rerank",
                "input": question,
                "observation": retrieved_ctx[:500],
            }],
            avg_confidence=avg_conf,
            faithfulness_score=faith,
            faithfulness_label=faith_label,
            latency_ms=latency_ms,
        )
        results.append({
            "question": question,
            "avg_confidence": avg_conf,
            "faithfulness_score": faith,
            "faithfulness_label": faith_label,
            "keyword_hit_rate": kw_hit,
            "latency_ms": latency_ms,
        })

    # Aggregate summary
    print(f"\n{'='*62}")
    print("  SUMMARY")
    print(f"{'='*62}")
    valid_conf = [r["avg_confidence"] for r in results if r["avg_confidence"] is not None]
    valid_faith = [r["faithfulness_score"] for r in results]
    valid_kw = [r["keyword_hit_rate"] for r in results if r["keyword_hit_rate"] is not None]
    valid_lat = [r["latency_ms"] for r in results]

    if valid_conf:
        print(f"  Mean retrieval confidence : {np.mean(valid_conf):.3f}")
    if valid_faith:
        print(f"  Mean faithfulness score   : {np.mean(valid_faith):.3f}")
    if valid_kw:
        print(f"  Mean keyword hit rate     : {np.mean(valid_kw):.3f}")
    if valid_lat:
        lats = sorted(valid_lat)
        p50 = lats[len(lats) // 2]
        p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)]
        print(f"  Latency  p50={p50}ms  p95={p95}ms  mean={int(np.mean(lats))}ms")
    print(f"  Logged {len(results)} entries → query_logs.db\n")

    return results


# ---------- REPORT ----------
def print_report(logger: QueryLogger, n: int = 20):
    rows = logger.recent(n)
    stats = logger.stats()
    print(f"\n{'='*62}")
    print(f"  SmartDoc Query Log — Last {n} Entries")
    print(f"{'='*62}")
    total = stats.get("total_queries") or 0
    mean_c = stats.get("mean_confidence")
    mean_f = stats.get("mean_faithfulness")
    refl = stats.get("reflection_count") or 0
    print(f"  Total queries       : {total}")
    print(f"  Mean confidence     : {mean_c:.3f}" if mean_c else "  Mean confidence     : N/A")
    print(f"  Mean faithfulness   : {mean_f:.3f}" if mean_f else "  Mean faithfulness   : N/A")
    print(f"  Reflected queries   : {refl}")
    print()
    for r in rows:
        conf = f"{r['avg_confidence']:.3f}" if r["avg_confidence"] is not None else "N/A "
        faith = f"{r['faithfulness_score']:.3f}" if r["faithfulness_score"] is not None else "N/A "
        reflected = "yes" if r["was_reflected"] else "no "
        print(f"  [{r['id']:>4}] {r['ts'][:19]}  conf={conf}  faith={faith}  reflected={reflected}")
        print(f"         Q: {r['question'][:75]}")
        answer_preview = (r["final_answer"] or "")[:75]
        print(f"         A: {answer_preview}{'...' if len(r['final_answer'] or '') > 75 else ''}")
        print()


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartDoc offline RAG evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf", help="Path to the PDF file to evaluate against")
    parser.add_argument("--queries", help="Path to JSON file with test queries")
    parser.add_argument(
        "--groq-key",
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API key (defaults to GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print recent log report from query_logs.db and exit",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of recent entries to show with --report (default: 20)",
    )
    args = parser.parse_args()

    _logger = QueryLogger()

    if args.report:
        print_report(_logger, n=args.n)
        sys.exit(0)

    if not args.pdf:
        parser.error("--pdf is required unless --report is used")
    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    if not args.groq_key:
        print(
            "Error: Groq API key required. Pass --groq-key or set GROQ_API_KEY env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    _queries = DEFAULT_TEST_QUERIES
    if args.queries:
        with open(args.queries) as f:
            _queries = json.load(f)

    run_eval(args.pdf, _queries, args.groq_key, _logger)
