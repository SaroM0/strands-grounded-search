"""Tool: research_with_sonar."""
from __future__ import annotations

import logging

from strands import tool

from ..clients.perplexity_client import PerplexityClient
from ..clients.research_cache_client import ResearchCacheClient
from ..config import settings

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
    logger.info("research_with_sonar: asking Perplexity %s", settings.perplexity_model)
    result = _client.research(query)
    logger.info("research_with_sonar: got %d chars, %d source(s)", len(result.content), len(result.sources))

    try:
        _get_cache_client().upsert(query, tool="research_with_sonar", result=result)
        logger.info("research_with_sonar: result cached in %s", settings.research_cache_collection)
    except Exception as exc:
        logger.warning(
            "research_with_sonar: failed to cache result at %s (%s)",
            settings.qdrant_url,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )

    return result.to_dict()
