"""Build a local Chroma index from PDFs placed in the private data directory."""

from pathlib import Path

import chromadb
import pymupdf

from model_config import get_embedding_function

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name="pdf_documents",
    embedding_function=get_embedding_function(),
    metadata={"hnsw:space": "cosine"},
)


def embedding_logic() -> None:
    """Index non-empty pages from local PDFs.

    Both ``data/*.pdf`` and ``Database/chroma_db`` are gitignored. This keeps
    the submitted repository free of the user's documents and their embeddings.
    """
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}. Add PDFs locally and run again.")
        return

    for pdf_number, pdf_path in enumerate(pdf_paths, start=1):
        print(f"Processing: {pdf_path.name}")
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                collection.upsert(
                    ids=[f"{pdf_number}-{page_number}"],
                    documents=[text],
                    metadatas=[{"source": pdf_path.name, "page": page_number}],
                )


if __name__ == "__main__":
    embedding_logic()
