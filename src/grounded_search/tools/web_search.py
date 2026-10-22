"""Tool: search_web."""
from __future__ import annotations

from strands import tool

from ..clients.web_search_client import get_web_search_client
from ..config import settings


@tool
def search_web(query: str) -> dict:
    """
    Searches the web for raw results (title, URL, snippet) without delegating the synthesis to
    another model.

    Use it for specific facts, recent data, or verifiable claims (versions, dates, figures) where
    it's better to see the original sources rather than an already-synthesized answer.

    Args:
        query: The search terms.

    Returns:
        A dictionary with a summary of results ("content") and the list of sources ("sources").
    """
    client = get_web_search_client()
    result = client.search(query, max_results=settings.web_search_max_results)
    return result.to_dict()
