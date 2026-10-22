"""Client for the research cache (used by search_research_cache, and written to by search_web
and research_with_sonar).

A Qdrant-backed semantic cache: every result that search_web or research_with_sonar produces is
embedded on its originating query and upserted here with metadata (which tool produced it, when).
A later question that is semantically close to a past query can then be answered by looking this
up instead of hitting Tavily/Brave/Perplexity again.

Modular by design: `upsert` and `search` are independent operations, each usable on its own.
"""
from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import settings
from ..schemas import CachedResearch, Source, ToolResult
from .embedding import Embedder, get_embedder


class ResearchCacheClient:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._client = _QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._embedder = embedder or get_embedder()
        self._collection = settings.research_cache_collection

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, query: str, tool: str, result: ToolResult) -> None:
        """Stores `result` in the cache, embedded on the `query` that produced it via `tool`."""
        vector = self._embedder.embed(query)
        self._ensure_collection(vector_size=len(vector))
        point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{tool}:{query}").hex
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "query": query,
                        "tool": tool,
                        "content": result.content,
                        "sources": [source.to_dict() for source in result.sources],
                        "cached_at": time.time(),
                    },
                )
            ],
        )

    def search(self, query: str, max_results: int, min_score: float) -> list[CachedResearch]:
        """Looks up cached research semantically close to `query`, above `min_score`."""
        if not self._client.collection_exists(self._collection):
            return []

        vector = self._embedder.embed(query)
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=max_results,
            score_threshold=min_score,
        ).points

        return [
            CachedResearch(
                query=hit.payload.get("query", ""),
                tool=hit.payload.get("tool", ""),
                cached_at=hit.payload.get("cached_at", 0.0),
                score=hit.score,
                result=ToolResult(
                    content=hit.payload.get("content", ""),
                    sources=[
                        Source(
                            title=source.get("title", ""),
                            url=source.get("url"),
                            snippet=source.get("snippet"),
                        )
                        for source in hit.payload.get("sources", [])
                    ],
                ),
            )
            for hit in hits
        ]
