"""MCP server for querying a local, private PDF collection."""

import logging
import sys
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "Database"
sys.path.append(str(DATABASE_DIR))

from retrieval import retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv(PROJECT_DIR / ".env")
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Private PDF RAG",
    instructions=(
        "Searches the user's locally indexed PDF documents. The documents and "
        "their generated vector database stay on the user's machine."
    ),
)


class QueryMetadata(BaseModel):
    """Optional filters supported by the generic PDF index."""

    model_config = ConfigDict(populate_by_name=True)

    source: Optional[str] = None
    page: Optional[int] = None


class Query(BaseModel):
    text: str
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)


class State(TypedDict):
    query: str
    structured_query: Query | None
    retrieval_result: dict | None


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    reasoning_format="parsed",
    max_retries=2,
)
model_with_structured_output = model.with_structured_output(Query)


def structure_query(user_query: str) -> Query:
    """Extract only explicit filename/page filters; retain the user's query."""
    prompt = f"""Parse the user query into the provided structured output.

Rules:
- `text` must contain the user's query exactly as written.
- Populate `source` only when the user explicitly names a PDF filename.
- Populate `page` only when the user explicitly gives a page number.
- Leave metadata null when uncertain. Never invent filters.

User query:
{user_query}
"""
    return model_with_structured_output.invoke(prompt)


def query_chroma(structured_query: Query) -> dict:
    logger.info("Querying local Chroma index")
    metadata = structured_query.metadata.model_dump(exclude_none=True)
    results = retrieve(query=structured_query.text, metadata=metadata or None)

    if metadata and (not results or results == "No results found."):
        results = retrieve(query=structured_query.text)
        return {
            "results": results,
            "warning": "No match used the requested filter; returning text-only matches.",
        }
    return {"results": results, "warning": None}


def structure_node(state: State) -> dict:
    return {"structured_query": structure_query(state["query"])}


def query_node(state: State) -> dict:
    return {"retrieval_result": query_chroma(state["structured_query"])}


builder = StateGraph(State)
builder.add_node("structure_query", structure_node)
builder.add_node("query_chroma", query_node)
builder.add_edge(START, "structure_query")
builder.add_edge("structure_query", "query_chroma")
builder.add_edge("query_chroma", END)
graph = builder.compile()


@mcp.tool()
def search_private_documents(query: str) -> dict:
    """Search the user's locally indexed PDFs for relevant passages."""
    return graph.invoke({"query": query, "structured_query": None, "retrieval_result": None})
