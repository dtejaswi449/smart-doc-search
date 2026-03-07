import os
import re
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # loads GROQ_API_KEY from .env if present
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ---------- CONFIG ----------
APP_TITLE = "Smart Doc Scan"
APP_ICON = "📄"
SEED_QUERY = "What is this document about?"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHARS_PER_PAGE = 100

#embedding model
EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"

#cross-encoder reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#Groq LLM (free, fast)
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
st.title(APP_TITLE)

# ---------- SIDEBAR: Groq API Key ----------
# Auto-loads from .env; sidebar field is a fallback for others running this app
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

# ---------- CACHED RESOURCES ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # normalize_embeddings=True is recommended for bge models
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

def get_llm(api_key):
    return ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0, max_tokens=512)


def clean_text(text: str) -> str:
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove standalone page numbers (lines with only digits)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Drop lines that are too short to be real content (headers/footers)
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
    # Sentence-aware separators: tries to split on paragraph → sentence → word boundaries
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


def build_bm25(docs):
    corpus = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(corpus)

def hybrid_search(query, vectorstore, bm25, all_docs, k=8):
    # Semantic leg
    semantic_hits = vectorstore.similarity_search_with_score(query, k=k)

    # Normalize FAISS L2 scores → higher = better
    max_s = max((s for _, s in semantic_hits), default=1e-9) + 1e-9
    semantic_map = {
        d.metadata["chunk"]: {"doc": d, "score": (1 - s / max_s) * 0.6}
        for d, s in semantic_hits
    }

    # BM25 leg
    bm25_scores = bm25.get_scores(query.lower().split())
    max_b = bm25_scores.max() + 1e-9
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k]

    # Merge with weights: 60% semantic, 40% BM25
    combined = dict(semantic_map)
    for idx in bm25_top_idx:
        if bm25_scores[idx] <= 0:
            continue
        doc = all_docs[idx]
        cid = doc.metadata["chunk"]
        bm25_contrib = (bm25_scores[idx] / max_b) * 0.4
        if cid in combined:
            combined[cid]["score"] += bm25_contrib
        else:
            combined[cid] = {"doc": doc, "score": bm25_contrib}

    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:k]]


def rerank(query, docs, reranker, top_k=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def rewrite_query(query, llm):
    messages = [
        SystemMessage(content=(
            "You are a search query optimizer for document retrieval. "
            "Rewrite the user's question to be clearer and more specific "
            "so it retrieves better results from a vector database. "
            "Return ONLY the rewritten query — no explanation, no quotes."
        )),
        HumanMessage(content=query),
    ]
    return llm.invoke(messages).content.strip()

# ---------- ANSWER GENERATION ----------
def generate_answer(question, context, llm):
    messages = [
        SystemMessage(content=(
            "You are a precise document assistant. Answer the user's question "
            "using ONLY the provided document context. "
            "If the answer is not in the context, say: "
            "'I could not find the answer in this document.' "
            "Cite the page number when possible."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content.strip()

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
    query = st.text_input("Your question")

    if query:
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to get answers.")
            st.stop()

        llm = get_llm(groq_api_key)
        reranker = get_reranker()

        # Rewrite query for better retrieval
        with st.spinner("Optimising query..."):
            expanded_query = rewrite_query(query, llm)
        st.caption(f"Expanded query: _{expanded_query}_")

        # Hybrid retrieval: BM25 + semantic search
        with st.spinner("Searching (hybrid BM25 + semantic)..."):
            candidate_docs = hybrid_search(expanded_query, vectorstore, bm25, docs, k=8)

        # Rerank candidates
        with st.spinner("Re-ranking results..."):
            top_docs = rerank(expanded_query, candidate_docs, reranker, top_k=3)

        if not top_docs:
            st.warning("No relevant content found in this document.")
            st.stop()

        # Build context with page citations
        context = "\n\n---\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in top_docs
        )

        # Generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, context, llm)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Source chunks used"):
            for d in top_docs:
                st.markdown(f"**Page {d.metadata.get('page', '?')}**")
                st.write(d.page_content)
