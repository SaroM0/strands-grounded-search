"""Herramienta: search_web."""
from __future__ import annotations

from strands import tool

from ..clients.web_search_client import get_web_search_client
from ..config import settings


@tool
def search_web(query: str) -> dict:
    """
    Busca en la web resultados crudos (título, URL, fragmento) sin delegar la síntesis a otro
    modelo.

    Úsala para hechos puntuales, datos recientes o verificables (versiones, fechas, cifras)
    donde conviene ver las fuentes originales en lugar de una respuesta ya sintetizada.

    Args:
        query: Los términos de búsqueda.

    Returns:
        Un diccionario con un resumen de resultados ("content") y la lista de fuentes
        ("sources").
    """
    client = get_web_search_client()
    result = client.search(query, max_results=settings.web_search_max_results)
    return result.to_dict()
