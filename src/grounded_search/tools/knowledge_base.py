"""Tool: search_knowledge_base."""
from __future__ import annotations

import logging

from strands import tool

from ..clients.knowledge_base_client import KnowledgeBaseClient
from ..clients.relevance_grader import RelevanceGrader
from ..config import settings
from ..schemas import ToolResult

logger = logging.getLogger(__name__)

_client: KnowledgeBaseClient | None = None
_grader: RelevanceGrader | None = None


def _get_client() -> KnowledgeBaseClient:
    global _client
    if _client is None:
        _client = KnowledgeBaseClient()
    return _client


def _get_grader() -> RelevanceGrader:
    global _grader
    if _grader is None:
        _grader = RelevanceGrader()
    return _grader


def _grade(query: str, result: ToolResult) -> dict:
    """Corrective-RAG step: grades the passages so the agent knows whether to fall back to the web."""
    try:
        grade = _get_grader().grade(query, [source.snippet or "" for source in result.sources])
    except Exception as exc:  # A failed grading shouldn't hide the passages themselves.
        logger.warning("search_knowledge_base: relevance grading failed: %s", exc)
        return {"relevant": "unknown", "error": str(exc)}
    logger.info(
        "search_knowledge_base: relevance %.3f (threshold %.2f) -> %s",
        grade.score,
        grade.threshold,
        "relevant" if grade.is_relevant else "not relevant",
    )
    return grade.to_dict()


@tool
def search_knowledge_base(query: str) -> dict:
    """
    Retrieves relevant passages from the loaded document base (Qdrant) via semantic search, and
    grades how relevant they are to the query.

    Use it when the question explicitly refers to documents uploaded by the user (papers,
    reports, meeting notes) rather than to general knowledge or the web.

    Args:
        query: The question or concept to search for in the documents.

    Returns:
        A dictionary with the retrieved passages ("content"), their sources ("sources") and,
        if grading is enabled, a "relevance" verdict: "relevant" is "yes" / "no" / "unknown",
        "score" the context precision of the passages and "relevant_passages" the 1-based
        numbers of the useful ones.
    """
    logger.info("search_knowledge_base: querying Qdrant collection %s (max %d)", settings.qdrant_collection, settings.kb_max_results)
    result = _get_client().search(query, max_results=settings.kb_max_results, min_score=settings.kb_min_score)
    logger.info("search_knowledge_base: got %d passage(s)", len(result.sources))
    payload = result.to_dict()
    if settings.kb_relevance_grading:
        payload["relevance"] = _grade(query, result)
    return payload
