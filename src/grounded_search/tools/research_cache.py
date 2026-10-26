"""Tool: search_research_cache."""
from __future__ import annotations

import logging

from strands import tool

from ..clients.research_cache_client import ResearchCacheClient
from ..config import settings

logger = logging.getLogger(__name__)

_client: ResearchCacheClient | None = None


def _get_client() -> ResearchCacheClient:
    global _client
    if _client is None:
        _client = ResearchCacheClient()
    return _client


@tool
def search_research_cache(query: str) -> dict:
    """
    Looks up research previously gathered by search_web or research_with_sonar that is
    semantically close to this query, so it can be reused instead of searching again.

    Use it first when a question looks like something that may already have been researched.
    A hit here is not automatically trustworthy forever: for time-sensitive or high-stakes
    claims, corroborate the cached result with a fresh search_web or research_with_sonar call
    before relying on it, especially if it looks old (see "age_seconds") or the similarity
    score is borderline.

    Args:
        query: The question to look up in the cache.

    Returns:
        A dictionary with "cache_hit" (bool) and "results": a list of past research entries,
        each with "query" (the original query that produced it), "tool" (which tool produced
        it), "age_seconds", "score" (semantic similarity to this query), "content", and
        "sources".
    """
    logger.info(
        "search_research_cache: querying %s (min_score=%.2f)",
        settings.research_cache_collection,
        settings.research_cache_min_score,
    )
    try:
        hits = _get_client().search(
            query,
            max_results=settings.research_cache_max_results,
            min_score=settings.research_cache_min_score,
        )
    except Exception as exc:
        logger.warning(
            "search_research_cache: cache unavailable at %s (%s) - is Qdrant running? (docker compose up -d)",
            settings.qdrant_url,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return {"cache_hit": False, "results": [], "error": f"research cache unavailable: {exc}"}

    logger.info("search_research_cache: %d hit(s) above threshold", len(hits))
    return {"cache_hit": bool(hits), "results": [hit.to_dict() for hit in hits]}
