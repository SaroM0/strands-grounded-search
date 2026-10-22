"""Herramienta: research_with_sonar."""
from __future__ import annotations

from strands import tool

from ..clients.perplexity_client import PerplexityClient

_client = PerplexityClient()


@tool
def research_with_sonar(query: str) -> dict:
    """
    Investiga una pregunta recopilando y sintetizando evidencia de múltiples fuentes vía
    Perplexity Sonar.

    Úsala cuando la pregunta requiera comparar posturas, ponderar evidencia a favor y en contra,
    o producir una síntesis razonada (no solo una lista de enlaces).

    Args:
        query: La pregunta de investigación a sintetizar.

    Returns:
        Un diccionario con la síntesis ("content") y las fuentes citadas ("sources").
    """
    result = _client.research(query)
    return result.to_dict()
