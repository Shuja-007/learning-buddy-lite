"""A privacy-conscious Streamlit demo for Private PDF RAG."""

import sys
import uuid
from pathlib import Path

import chromadb
import pymupdf
import streamlit as st

DATABASE_DIR = Path(__file__).resolve().parent / "Database"
sys.path.insert(0, str(DATABASE_DIR))

from model_config import get_embedding_function, get_reranker

MAX_RESULTS = 10
RERANK_TOP_K = 5
DISTANCE_THRESHOLD = 0.6


@st.cache_resource
def get_client() -> chromadb.ClientAPI:
    return chromadb.Client()


@st.cache_resource
def get_embedder():
    return get_embedding_function()


def index_pdf(uploaded_file) -> tuple[chromadb.Collection, int]:
    """Create an in-memory collection from the uploaded PDF's non-empty pages."""
    collection = get_client().create_collection(
        name=f"upload_{uuid.uuid4().hex}",
        embedding_function=get_embedder(),
        metadata={"hnsw:space": "cosine"},
    )

    document = pymupdf.open(stream=uploaded_file.getvalue(), filetype="pdf")
    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                collection.add(
                    ids=[str(page_number)],
                    documents=[text],
                    metadatas=[{"source": uploaded_file.name, "page": page_number}],
                )
    finally:
        document.close()

    return collection, collection.count()


def search(collection: chromadb.Collection, query: str) -> list[dict]:
    result_count = min(MAX_RESULTS, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    matches = [
        {"text": document, "metadata": metadata, "distance": distance}
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if distance <= DISTANCE_THRESHOLD
    ]

    if len(matches) > RERANK_TOP_K:
        scores = get_reranker().predict([[query, match["text"]] for match in matches])
        matches = [
            match
            for match, _ in sorted(
                zip(matches, scores), key=lambda pair: pair[1], reverse=True
            )[:RERANK_TOP_K]
        ]
    return matches


st.set_page_config(page_title="Private PDF RAG", page_icon="🔒", layout="wide")
st.title("🔒 Private PDF RAG")
st.caption("Upload a PDF, index it in memory, and ask questions about it.")

with st.expander("Privacy note", expanded=True):
    st.write(
        "This demo never uses the project author's PDFs or API key. Uploaded PDFs "
        "are processed in memory and are not committed to Git. Do not upload sensitive "
        "documents to a public demo; run the local version from this repository instead."
    )

uploaded_file = st.file_uploader("Choose a PDF to search", type="pdf")
if uploaded_file is not None:
    needs_index = st.session_state.get("uploaded_name") != uploaded_file.name
    if needs_index:
        with st.spinner("Extracting text and creating local embeddings..."):
            try:
                collection, page_count = index_pdf(uploaded_file)
            except Exception as error:
                st.error(f"The PDF could not be indexed: {error}")
            else:
                st.session_state.collection = collection
                st.session_state.page_count = page_count
                st.session_state.uploaded_name = uploaded_file.name

    if "collection" in st.session_state:
        st.success(f"Ready: indexed {st.session_state.page_count} non-empty pages.")
        question = st.text_input("Ask a question about this PDF")
        if question:
            with st.spinner("Searching relevant passages..."):
                matches = search(st.session_state.collection, question)
            if not matches:
                st.info("No strong matches found. Try a more specific question.")
            else:
                st.subheader("Relevant passages")
                for match in matches:
                    page = match["metadata"]["page"]
                    with st.expander(
                        f"Page {page} · similarity distance {match['distance']:.3f}"
                    ):
                        st.write(match["text"])
else:
    st.info("Upload a PDF to begin.")
