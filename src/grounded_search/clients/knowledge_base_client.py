"""Client for the document knowledge base (used by search_knowledge_base).

Uses Qdrant for vector search and an interchangeable Embedder (defaults to fastembed running
locally, with no dependency on an external API).
"""
from __future__ import annotations

from qdrant_client import QdrantClient as _QdrantClient

from ..config import settings
from ..schemas import Source, ToolResult
from .embedding import Embedder, get_embedder


class KnowledgeBaseClient:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._client = _QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._embedder = embedder or get_embedder()
        self._collection = settings.qdrant_collection

    def search(self, query: str, max_results: int) -> ToolResult:
        vector = self._embedder.embed(query)
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=max_results,
        ).points

        sources = [
            Source(
                title=hit.payload.get("title", hit.payload.get("source", "document")),
                url=hit.payload.get("url"),
                snippet=(hit.payload.get("text", "") or "")[:500],
            )
            for hit in hits
        ]
        content = "\n\n".join(f"- {source.title}: {source.snippet}" for source in sources)
        return ToolResult(content=content, sources=sources)
