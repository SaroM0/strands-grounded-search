"""Tool: search_web."""
from __future__ import annotations

import logging

from strands import tool

from ..clients.research_cache_client import ResearchCacheClient
from ..clients.web_search_client import get_web_search_client
from ..config import settings

logger = logging.getLogger(__name__)

_cache_client: ResearchCacheClient | None = None


def _get_cache_client() -> ResearchCacheClient:
    global _cache_client
    if _cache_client is None:
        _cache_client = ResearchCacheClient()
    return _cache_client


@tool
def search_web(query: str) -> dict:
    """
    Searches the web for raw results (title, URL, snippet) without delegating the synthesis to
    another model.

    Use it for specific facts, recent data, or verifiable claims (versions, dates, figures) where
    it's better to see the original sources rather than an already-synthesized answer. Results
    are cached automatically so search_research_cache can reuse them later.

    Args:
        query: The search terms.

    Returns:
        A dictionary with a summary of results ("content") and the list of sources ("sources").
    """
    client = get_web_search_client()
    result = client.search(query, max_results=settings.web_search_max_results)

    try:
        _get_cache_client().upsert(query, tool="search_web", result=result)
    except Exception:
        logger.warning("failed to cache search_web result", exc_info=True)

    return result.to_dict()
