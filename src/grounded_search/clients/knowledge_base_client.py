"""Cliente para la base de conocimiento documental (usado por search_knowledge_base).

Usa Qdrant para la búsqueda vectorial y un Embedder intercambiable (por defecto, fastembed
corriendo en local, sin dependencia de una API externa).
"""
from __future__ import annotations

from typing import Protocol

from qdrant_client import QdrantClient as _QdrantClient

from ..config import settings
from ..schemas import Source, ToolResult


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Embeddings locales vía fastembed."""

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name or settings.embedding_model)

    def embed(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))


def get_embedder() -> Embedder:
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"Proveedor de embeddings no soportado: {settings.embedding_provider}")


class KnowledgeBaseClient:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._client = _QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._embedder = embedder or get_embedder()
        self._collection = settings.qdrant_collection

    def search(self, query: str, max_results: int) -> ToolResult:
        vector = self._embedder.embed(query)
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=max_results,
        )

        sources = [
            Source(
                title=hit.payload.get("title", hit.payload.get("source", "documento")),
                url=hit.payload.get("url"),
                snippet=(hit.payload.get("text", "") or "")[:500],
            )
            for hit in hits
        ]
        content = "\n\n".join(f"- {source.title}: {source.snippet}" for source in sources)
        return ToolResult(content=content, sources=sources)
