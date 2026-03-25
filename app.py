import math
import os
import re
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import QueryLogger

load_dotenv()  # loads GROQ_API_KEY from .env if present
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain.agents import create_react_agent, AgentExecutor
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ---------- CONFIG ----------
APP_TITLE = "Smart Doc Scan"
APP_ICON = "📄"
SEED_QUERY = "What is this document about?"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHARS_PER_PAGE = 100

EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# Patterns that indicate the agent couldn't find a good answer
WEAK_ANSWER_PATTERNS = [
    "could not find",
    "not in the document",
    "no information",
    "unable to find",
    "don't have information",
    "cannot find",
    "not mentioned",
    "no relevant",
]

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
st.title(APP_TITLE)

# ---------- SIDEBAR: Groq API Key ----------
_env_key = os.getenv("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=_env_key,
    type="password",
    placeholder="gsk_...",
    help="Free key at console.groq.com — takes 1 minute to get",
)
if not groq_api_key:
    st.sidebar.info("Get a free key at [console.groq.com](https://console.groq.com)")

# ---------- SIDEBAR: Monitoring ----------
st.sidebar.divider()
st.sidebar.subheader("Monitoring")
_stats = get_logger().stats() if True else {}  # always fresh
_corpus = get_logger().corpus_stats()
if _corpus.get("total_documents"):
    st.sidebar.markdown(
        f"**Corpus:** {_corpus['total_documents']} PDF(s) · "
        f"{_corpus['total_pages'] or 0} pages · "
        f"{_corpus['total_chunks'] or 0} chunks"
    )
if _stats.get("total_queries"):
    st.sidebar.markdown(
        f"**Queries:** {_stats['total_queries']} total · "
        f"{int(_stats.get('reflection_count') or 0)} reflected"
    )
    if _stats.get("mean_confidence") is not None:
        st.sidebar.markdown(f"**Avg confidence:** {_stats['mean_confidence']:.2f}")
    if _stats.get("mean_faithfulness") is not None:
        st.sidebar.markdown(f"**Avg faithfulness:** {_stats['mean_faithfulness']:.2f}")
    if _stats.get("mean_latency_ms") is not None:
        st.sidebar.markdown(f"**Avg latency:** {_stats['mean_latency_ms'] / 1000:.1f}s")
else:
    st.sidebar.caption("No queries logged yet.")

# ---------- CACHED RESOURCES ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

@st.cache_resource(show_spinner=False)
def get_logger():
    return QueryLogger()

def get_llm(api_key: str) -> ChatGroq:
    # tool_choice="auto" lets Groq decide whether to call a tool
    return ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0, max_tokens=1024)


# ---------- TEXT HELPERS ----------
def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    lines = [l for l in text.split("\n") if len(l.strip()) > 3 or l.strip() == ""]
    return "\n".join(lines).strip()


def extract_text_pdf(file_bytes):
    """Return list of {text, page} dicts and per-page char counts."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages, counts = [], []
    for i, page in enumerate(doc):
        txt = clean_text(page.get_text("text"))
        pages.append({"text": txt, "page": i + 1})
        counts.append(len(txt.strip()))
    return pages, counts

def is_likely_scanned(counts, min_chars=MIN_CHARS_PER_PAGE):
    if not counts:
        return True
    low = sum(c < min_chars for c in counts)
    return low > max(1, len(counts) // 2)

def make_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    docs = []
    chunk_idx = 0
    for page_data in pages:
        if not page_data["text"].strip():
            continue
        for chunk_text in splitter.split_text(page_data["text"]):
            docs.append(Document(
                page_content=chunk_text,
                metadata={"chunk": chunk_idx, "page": page_data["page"]},
            ))
            chunk_idx += 1
    return docs


# ---------- INDEX HELPERS ----------
def build_bm25(docs):
    corpus = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(corpus)

def rerank(query, docs, reranker, top_k=3):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

def rerank_with_scores(query, docs, reranker, top_k=3) -> tuple[list, list[float]]:
    """Like rerank(), but also returns sigmoid-normalized confidence scores (0–1)."""
    if not docs:
        return [], []
    pairs = [(query, d.page_content) for d in docs]
    raw = reranker.predict(pairs)
    ranked = sorted(zip(raw, docs), key=lambda x: x[0], reverse=True)[:top_k]
    scores = [1.0 / (1.0 + math.exp(-float(s))) for s, _ in ranked]
    top_docs = [d for _, d in ranked]
    return top_docs, scores

def faithfulness_score(answer: str, context: str) -> tuple[float, str]:
    """
    Token-overlap faithfulness: fraction of content words in the answer
    that appear in the retrieved context. Proxy for answer groundedness.
    Returns (score 0-1, label: 'high' / 'medium' / 'low').
    """
    stop = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","could","should","may","might","in",
        "on","at","to","for","of","and","or","but","not","with","this","that",
        "it","its","i","you","he","she","they","we","from","by","as","about",
    }
    a_tokens = set(re.findall(r"\b\w+\b", answer.lower())) - stop
    c_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    if not a_tokens:
        return 0.0, "low"
    score = round(len(a_tokens & c_tokens) / len(a_tokens), 3)
    label = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"
    return score, label

def format_docs(docs) -> str:
    """Format retrieved docs into a string with page citations."""
    if not docs:
        return "No relevant content found."
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


# ---------- AGENT TOOLS ----------
def make_tools(vectorstore, bm25, all_docs, reranker, llm, full_text):
    """Create the three LangChain tools for the ReAct agent."""

    def _store_conf(scores: list[float]):
        if "_conf_scores" not in st.session_state:
            st.session_state["_conf_scores"] = []
        st.session_state["_conf_scores"].extend(scores)

    def semantic_search(query: str) -> str:
        """Search the document using vector embeddings (semantic similarity).
        Best for conceptual questions and meaning-based retrieval."""
        hits = vectorstore.similarity_search(query, k=6)
        top_docs, scores = rerank_with_scores(query, hits, reranker, top_k=3)
        _store_conf(scores)
        return format_docs(top_docs)

    def keyword_search(query: str) -> str:
        """Search the document by exact keywords using BM25.
        Best for specific terms, names, numbers, dates, or exact phrases."""
        bm25_scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(bm25_scores)[::-1][:6]
        candidate_docs = [all_docs[i] for i in top_idx if bm25_scores[i] > 0]
        if not candidate_docs:
            return "No keyword matches found."
        top_docs, scores = rerank_with_scores(query, candidate_docs, reranker, top_k=3)
        _store_conf(scores)
        return format_docs(top_docs)

    def summarize_document(_: str) -> str:
        """Get a high-level summary of the entire document.
        Use for questions about overall topics, main points, or document structure."""
        excerpt = full_text[:4000]
        messages = [
            SystemMessage(content=(
                "Summarize the following document excerpt. "
                "Focus on the main topics, key points, and overall structure. "
                "Be concise but comprehensive."
            )),
            HumanMessage(content=excerpt),
        ]
        return llm.invoke(messages).content.strip()

    return [
        StructuredTool.from_function(
            func=semantic_search,
            name="semantic_search",
            description=(
                "Search the document semantically using FAISS vector embeddings. "
                "Best for conceptual questions or when you need passages relevant by meaning. "
                "Input: a natural language query string."
            ),
        ),
        StructuredTool.from_function(
            func=keyword_search,
            name="keyword_search",
            description=(
                "Search the document by exact keywords using BM25. "
                "Best for specific terms, names, numbers, dates, or exact phrases. "
                "Input: keywords or a short phrase to look up."
            ),
        ),
        StructuredTool.from_function(
            func=summarize_document,
            name="summarize_document",
            description=(
                "Get a high-level summary of the entire document. "
                "Use when asked about the overall topic, main points, or document structure. "
                "Input: any string (ignored — always summarizes the full document)."
            ),
        ),
    ]


# ---------- AGENT ----------
REACT_PROMPT = PromptTemplate.from_template(
    "You are a precise document Q&A assistant.\n"
    "Strategy: use semantic_search for conceptual questions, keyword_search for specific "
    "terms/names/numbers, and summarize_document for overview questions. "
    "If one tool gives weak results, try another or rephrase. "
    "Always cite page numbers. "
    "If no tool finds relevant content, say: 'I could not find the answer in this document.'\n\n"
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

def build_agent(llm, tools):
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


# ---------- SELF-REFLECTION ----------
def is_weak_answer(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in WEAK_ANSWER_PATTERNS)

def run_with_reflection(question: str, agent, llm):
    """
    Run the agent. If it returns a weak answer, have the LLM reflect on why
    retrieval failed, generate a better query, and re-run once.

    Returns: (final_answer, intermediate_steps, reflected: bool, reflected_query: str | None)
    AgentExecutor result keys: "output" (str), "intermediate_steps" (list of (AgentAction, str))
    """
    result = agent.invoke({"input": question})
    answer = result["output"]
    steps = result.get("intermediate_steps", [])

    if not is_weak_answer(answer):
        return answer, steps, False, None

    # Self-reflection: ask LLM to reformulate the query
    reflection_msgs = [
        SystemMessage(content=(
            "You are a search strategist. A document Q&A agent returned a weak answer. "
            "Suggest a better, more specific reformulation of the question that might "
            "retrieve the relevant content. Return ONLY the improved query — no explanation."
        )),
        HumanMessage(content=(
            f"Original question: {question}\n"
            f"Weak answer: {answer}\n"
            "Improved search query:"
        )),
    ]
    better_query = llm.invoke(reflection_msgs).content.strip()

    result2 = agent.invoke({"input": better_query})
    return result2["output"], result2.get("intermediate_steps", []), True, better_query


# ---------- TRACE DISPLAY ----------
def render_agent_trace(steps):
    """Render AgentExecutor intermediate steps: list of (AgentAction, observation)."""
    if not steps:
        st.write("No intermediate steps recorded.")
        return
    for i, (action, observation) in enumerate(steps, 1):
        st.markdown(f"**Step {i} — Tool:** `{action.tool}`")
        st.code(action.tool_input, language="text")
        preview = observation[:400] + ("..." if len(observation) > 400 else "")
        st.markdown("**Observation:**")
        st.write(preview)


# ---------- UI FLOW ----------
uploaded = st.file_uploader("Choose a **text-based** PDF", type="pdf")

if uploaded:
    st.success("Document uploaded successfully.")
    st.write("**File name:**", uploaded.name)
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.write("**File size:**", f"{size_mb:.2f} MB")

    # Extract & clean
    with st.spinner("Extracting and cleaning text..."):
        pages, counts = extract_text_pdf(uploaded.getvalue())
        full_text = "\n\n".join(p["text"] for p in pages)

    if is_likely_scanned(counts):
        st.error("This looks like a scanned (image-only) PDF. Please upload a text-based PDF.")
        st.stop()
    if not full_text.strip():
        st.error("No selectable text found in this PDF.")
        st.stop()

    st.subheader("Preview")
    st.code(full_text[:600] + ("..." if len(full_text) > 600 else ""), language="markdown")

    # Chunk
    with st.spinner("Splitting into chunks..."):
        docs = make_chunks(pages)
    st.write(f"**Total chunks:** {len(docs)}")
    with st.expander("Show chunk previews"):
        for i in range(min(5, len(docs))):
            st.markdown(f"**Chunk {i} — Page {docs[i].metadata.get('page', '?')}**")
            st.write(docs[i].page_content)

    # Index
    with st.spinner("Building vector index (first run downloads ~1.3 GB model)..."):
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        bm25 = build_bm25(docs)
    # Log corpus stats (deduplicated by filename in the UI but all runs recorded)
    get_logger().log_corpus(uploaded.name, len(pages), len(docs))

    # Sanity peek
    st.divider()
    st.subheader("Document summary peek")
    hits = vectorstore.similarity_search(SEED_QUERY, k=3)
    for i, res in enumerate(hits, 1):
        st.markdown(f"**Hit {i} — Page {res.metadata.get('page', '?')}**")
        st.write(res.page_content)

    # Q&A
    st.divider()
    st.subheader("Ask a question about this document")
    st.caption("Powered by a ReAct agent with semantic search, keyword search, and summarization tools.")
    query = st.text_input("Your question")

    if query:
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to get answers.")
            st.stop()

        llm = get_llm(groq_api_key)
        reranker = get_reranker()

        # Build tools and agent
        tools = make_tools(vectorstore, bm25, docs, reranker, llm, full_text)
        agent = build_agent(llm, tools)

        # Clear any leftover scores from a previous run
        st.session_state.pop("_conf_scores", None)

        # Run agent with self-reflection (timed)
        with st.spinner("Agent is reasoning and searching..."):
            _t0 = time.perf_counter()
            answer, steps, was_reflected, reflected_query = run_with_reflection(
                query, agent, llm
            )
            latency_ms = round((time.perf_counter() - _t0) * 1000)

        # Collect confidence scores accumulated during tool calls
        raw_scores = st.session_state.pop("_conf_scores", [])
        avg_conf = float(np.mean(raw_scores)) if raw_scores else None

        # Faithfulness: does the answer come from the retrieved context?
        retrieved_context = "\n".join(obs for _, obs in steps)
        faith, faith_label = faithfulness_score(answer, retrieved_context)

        # Log to SQLite
        tool_calls_log = [
            {"tool": action.tool, "input": action.tool_input, "observation": obs[:500]}
            for action, obs in steps
        ]
        get_logger().log(
            question=query,
            final_answer=answer,
            was_reflected=was_reflected,
            reflected_query=reflected_query,
            tool_calls=tool_calls_log,
            avg_confidence=avg_conf,
            faithfulness_score=faith,
            faithfulness_label=faith_label,
            latency_ms=latency_ms,
        )

        # Show reflection notice if triggered
        if was_reflected:
            st.info(
                f"Initial retrieval was weak. Agent self-reflected and re-queried with: "
                f"_{reflected_query}_"
            )

        st.markdown("### Answer")
        st.write(answer)

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            if avg_conf is not None:
                st.metric(
                    "Retrieval confidence",
                    f"{avg_conf:.2f}",
                    help="Avg sigmoid-normalised CrossEncoder score across all tool calls (0–1)",
                )
        with col2:
            st.metric(
                "Faithfulness",
                f"{faith:.2f} ({faith_label})",
                help="Token overlap between answer and retrieved chunks — proxy for groundedness",
            )
        with col3:
            st.metric(
                "Latency",
                f"{latency_ms / 1000:.1f}s",
                help="End-to-end wall-clock time from query submission to answer (includes all LLM + tool calls)",
            )

        with st.expander("Agent reasoning trace"):
            render_agent_trace(steps)
