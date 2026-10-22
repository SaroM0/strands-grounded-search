"""Centralized system configuration, loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    # Main LLM for the orchestrator agent (Amazon Bedrock)
    aws_public_key_bedrock: str | None = field(default_factory=lambda: _env("AWS_PUBLIC_KEY_BEDROCK"))
    aws_private_key_bedrock: str | None = field(default_factory=lambda: _env("AWS_PRIVATE_KEY_BEDROCK"))
    aws_bearer_token_bedrock: str | None = field(default_factory=lambda: _env("AWS_BEARER_TOKEN_BEDROCK"))
    aws_region_bedrock: str = field(default_factory=lambda: _env("AWS_REGION_BEDROCK", "us-east-1"))
    bedrock_model_id: str = field(
        default_factory=lambda: _env("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-5-5")
    )

    # research_with_sonar tool (Perplexity Sonar)
    perplexity_api_key: str | None = field(default_factory=lambda: _env("PERPLEXITY_API_KEY"))
    perplexity_model: str = field(default_factory=lambda: _env("PERPLEXITY_MODEL", "sonar-reasoning-pro"))

    # search_web tool
    web_search_provider: str = field(default_factory=lambda: _env("WEB_SEARCH_PROVIDER", "tavily"))
    tavily_api_key: str | None = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    brave_api_key: str | None = field(default_factory=lambda: _env("BRAVE_API_KEY"))
    web_search_max_results: int = field(default_factory=lambda: int(_env("WEB_SEARCH_MAX_RESULTS", "5")))

    # search_knowledge_base tool (Qdrant)
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "grounded_search_docs"))
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "fastembed"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    kb_max_results: int = field(default_factory=lambda: int(_env("KB_MAX_RESULTS", "5")))

    # search_research_cache tool (Qdrant) - semantic cache of search_web / research_with_sonar results
    research_cache_collection: str = field(
        default_factory=lambda: _env("RESEARCH_CACHE_COLLECTION", "grounded_search_cache")
    )
    research_cache_max_results: int = field(
        default_factory=lambda: int(_env("RESEARCH_CACHE_MAX_RESULTS", "3"))
    )
    research_cache_min_score: float = field(
        default_factory=lambda: float(_env("RESEARCH_CACHE_MIN_SCORE", "0.75"))
    )


settings = Settings()
