"""Herramienta: search_knowledge_base."""
from __future__ import annotations

from strands import tool

from ..clients.knowledge_base_client import KnowledgeBaseClient
from ..config import settings

_client: KnowledgeBaseClient | None = None


def _get_client() -> KnowledgeBaseClient:
    global _client
    if _client is None:
        _client = KnowledgeBaseClient()
    return _client


@tool
def search_knowledge_base(query: str) -> dict:
    """
    Recupera pasajes relevantes de la base documental cargada (Qdrant), vía búsqueda semántica.

    Úsala cuando la pregunta se refiera explícitamente a documentos cargados por el usuario
    (papers, informes, actas) y no a conocimiento general o a la web.

    Args:
        query: La pregunta o concepto a buscar en los documentos.

    Returns:
        Un diccionario con los pasajes recuperados ("content") y sus fuentes ("sources").
    """
    result = _get_client().search(query, max_results=settings.kb_max_results)
    return result.to_dict()
