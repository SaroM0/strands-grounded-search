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
    logger.info("search_web: querying %s (max %d results)", settings.web_search_provider, settings.web_search_max_results)
    result = client.search(query, max_results=settings.web_search_max_results)
    logger.info("search_web: got %d source(s)", len(result.sources))

    try:
        _get_cache_client().upsert(query, tool="search_web", result=result)
        logger.info("search_web: result cached in %s", settings.research_cache_collection)
    except Exception as exc:
        logger.warning(
            "search_web: failed to cache result at %s (%s)",
            settings.qdrant_url,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )

    return result.to_dict()
