from pathlib import Path
import chromadb

from model_config import get_embedding_function, get_reranker

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="pdf_documents",
    embedding_function=get_embedding_function(),
    metadata={"hnsw:space": "cosine"},
)


def retrieve(query, metadata=None):
    DISTANCE_THRESHOLD = 0.6
    MAX_RESULTS = 25
    RERANK_TOP_K = 5

    # Build optional metadata filter
    where = None

    if metadata:
        conditions = [
            {key: {"$eq": value}}
            for key, value in metadata.items()
        ]

        if len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

    # Retrieve from the MAIN database
    results = collection.query(
        query_texts=[query],
        n_results=MAX_RESULTS,
        where=where,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Apply cosine-distance threshold
    filtered_documents = [
        document
        for document, distance in zip(documents, distances)
        if distance <= DISTANCE_THRESHOLD
    ]

    # No results
    if not filtered_documents:
        return "No results found."

    # 5 or fewer results → return them directly
    if len(filtered_documents) <= 5:
        return filtered_documents

    # More than 5 → rerank
    pairs = [
        [query, document]
        for document in filtered_documents
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(filtered_documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return only the texts of the top 5
    return [
        document
        for document, score in ranked[:RERANK_TOP_K]
    ]
