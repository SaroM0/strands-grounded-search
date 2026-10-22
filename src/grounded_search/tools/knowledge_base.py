"""Tool: search_knowledge_base."""
from __future__ import annotations

from strands import tool

from ..clients.knowledge_base_client import KnowledgeBaseClient
from ..config import settings

_client: KnowledgeBaseClient | None = None


def _get_client() -> KnowledgeBaseClient:
    global _client
    if _client is None:
        _client = KnowledgeBaseClient()
    return _client


@tool
def search_knowledge_base(query: str) -> dict:
    """
    Retrieves relevant passages from the loaded document base (Qdrant) via semantic search.

    Use it when the question explicitly refers to documents uploaded by the user (papers,
    reports, meeting notes) rather than to general knowledge or the web.

    Args:
        query: The question or concept to search for in the documents.

    Returns:
        A dictionary with the retrieved passages ("content") and their sources ("sources").
    """
    result = _get_client().search(query, max_results=settings.kb_max_results)
    return result.to_dict()
