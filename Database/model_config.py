"""Central model configuration for the lightweight RAG server.

Both models are compact English models suitable for the English lecture PDFs in
this project.  Keeping their names here prevents the embedding pipeline and
the MCP server from ever using different vector spaces by accident.
"""

from functools import lru_cache

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder

# About 80 MB.  Strong, fast baseline for semantic search in English.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# About 80 MB.  It is loaded only when reranking is actually needed.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the reranker only after a query has more than five candidates."""
    return CrossEncoder(RERANKER_MODEL)
