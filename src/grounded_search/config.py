"""Configuración centralizada del sistema, cargada desde variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # LLM principal del agente orquestador
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model_id: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL_ID", "claude-sonnet-5-5"))

    # Herramienta research_with_sonar (Perplexity Sonar)
    perplexity_api_key: str | None = field(default_factory=lambda: _env("PERPLEXITY_API_KEY"))
    perplexity_model: str = field(default_factory=lambda: _env("PERPLEXITY_MODEL", "sonar-reasoning-pro"))

    # Herramienta search_web
    web_search_provider: str = field(default_factory=lambda: _env("WEB_SEARCH_PROVIDER", "tavily"))
    tavily_api_key: str | None = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    brave_api_key: str | None = field(default_factory=lambda: _env("BRAVE_API_KEY"))
    web_search_max_results: int = field(default_factory=lambda: int(_env("WEB_SEARCH_MAX_RESULTS", "5")))

    # Herramienta search_knowledge_base (Qdrant)
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "grounded_search_docs"))
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "fastembed"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    kb_max_results: int = field(default_factory=lambda: int(_env("KB_MAX_RESULTS", "5")))


settings = Settings()
