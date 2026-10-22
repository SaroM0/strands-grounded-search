"""Shared embedding provider, used by both the research cache and the user knowledge base."""
from __future__ import annotations

from typing import Protocol

from ..config import settings


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local embeddings via fastembed."""

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name or settings.embedding_model)

    def embed(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))


def get_embedder() -> Embedder:
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
