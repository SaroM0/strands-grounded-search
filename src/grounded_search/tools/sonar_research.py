"""Tool: research_with_sonar."""
from __future__ import annotations

import logging

from strands import tool

from ..clients.perplexity_client import PerplexityClient
from ..clients.research_cache_client import ResearchCacheClient

logger = logging.getLogger(__name__)

_client = PerplexityClient()
_cache_client: ResearchCacheClient | None = None


def _get_cache_client() -> ResearchCacheClient:
    global _cache_client
    if _cache_client is None:
        _cache_client = ResearchCacheClient()
    return _cache_client


@tool
def research_with_sonar(query: str) -> dict:
    """
    Researches a question by gathering and synthesizing evidence from multiple sources via
    Perplexity Sonar.

    Use it when the question requires comparing positions, weighing evidence for and against, or
    producing a reasoned synthesis (not just a list of links). Results are cached automatically
    so search_research_cache can reuse them later.

    Args:
        query: The research question to synthesize.

    Returns:
        A dictionary with the synthesis ("content") and the cited sources ("sources").
    """
    result = _client.research(query)

    try:
        _get_cache_client().upsert(query, tool="research_with_sonar", result=result)
    except Exception:
        logger.warning("failed to cache research_with_sonar result", exc_info=True)

    return result.to_dict()
