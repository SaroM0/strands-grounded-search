"""Tool: research_with_sonar."""
from __future__ import annotations

from strands import tool

from ..clients.perplexity_client import PerplexityClient

_client = PerplexityClient()


@tool
def research_with_sonar(query: str) -> dict:
    """
    Researches a question by gathering and synthesizing evidence from multiple sources via
    Perplexity Sonar.

    Use it when the question requires comparing positions, weighing evidence for and against, or
    producing a reasoned synthesis (not just a list of links).

    Args:
        query: The research question to synthesize.

    Returns:
        A dictionary with the synthesis ("content") and the cited sources ("sources").
    """
    result = _client.research(query)
    return result.to_dict()
