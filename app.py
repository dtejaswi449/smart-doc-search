import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# ---------- BASIC CONFIG ----------
APP_TITLE = "Smart Doc Scan"
APP_ICON = "📄"
SEED_QUERY = "What is this document about?"
CHUNK_SIZE = 800               
CHUNK_OVERLAP = 200
MIN_CHARS_PER_PAGE = 100       # guardrail for scanned PDFs
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-cased-distilled-squad"  

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
st.title(APP_TITLE)

# ---------- CACHED RESOURCES ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

@st.cache_resource(show_spinner=False)
def get_qa_pipeline():
    return pipeline(
    "question-answering",
    model=QA_MODEL,
    handle_impossible_answer=True,
    max_answer_len=60
)


# ---------- HELPERS ----------
def extract_text_pdf(file_bytes):
    """Return full text and per-page character counts."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts, counts = [], []
    for page in doc:
        txt = page.get_text("text")
        parts.append(txt)
        counts.append(len(txt.strip()))
    return "".join(parts), counts

def is_likely_scanned(counts, min_chars=MIN_CHARS_PER_PAGE):
    if not counts:
        return True
    low = sum(c < min_chars for c in counts)
    return low > max(1, len(counts) // 2)

def make_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    docs = splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata["chunk"] = i
    return docs

# ---------- UI FLOW ----------
uploaded = st.file_uploader("Choose a **text-based** PDF", type="pdf")

if uploaded:
    st.success("Document uploaded successfully.")
    st.write("**File name:**", uploaded.name)
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.write("**File size:**", f"{size_mb:.2f} MB")

    # Extract & guardrail
    with st.spinner("Extracting text..."):
        full_text, counts = extract_text_pdf(uploaded.getvalue())

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
        docs = make_chunks(full_text)
    st.write("**Total chunks:**", len(docs))
    with st.expander("Show some chunk previews"):
        for i in range(min(5, len(docs))):
            st.markdown(f"**Chunk {i}**")
            st.write(docs[i].page_content)

    # Index
    with st.spinner("Building vector index..."):
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Sanity peek
    st.divider()
    st.subheader("Document summary peek")
    hits = vectorstore.similarity_search(SEED_QUERY, k=3)
    for i, res in enumerate(hits, 1):
        st.markdown(f"**Hit {i} (chunk {res.metadata.get('chunk', '?')})**")
        st.write(res.page_content)

   
    st.divider()
    st.subheader("Ask a question about this document")
    query = st.text_input("Your question")
    if query:
        with st.spinner("Searching..."):
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=8)
            docs_with_scores.sort(key=lambda x: x[1])

            SCORE_THRESHOLD = 0.8
            TOP_K = 3

            filtered_docs = [
                d for d, score in docs_with_scores if score < SCORE_THRESHOLD
            ][:TOP_K]

        if not filtered_docs:
            st.warning("Answer not found in this document.")
            st.stop()

        context = "\n\n---\n\n".join(d.page_content for d in filtered_docs)

        qa = get_qa_pipeline()
        with st.spinner("Answering..."):
            out = qa(question=query, context=context)

        st.markdown("### Answer")
        st.write(out.get("answer", "").strip())
