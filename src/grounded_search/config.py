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
    # Minimum Qdrant similarity for a passage to be returned at all (empty = no threshold)
    kb_min_score: float | None = field(
        default_factory=lambda: float(_env("KB_MIN_SCORE")) if _env("KB_MIN_SCORE") else None
    )

    # Corrective RAG: an LLM grades the retrieved passages (context precision) so the agent knows
    # when the knowledge base isn't enough and it should fall back to the web
    kb_relevance_grading: bool = field(
        default_factory=lambda: _env("KB_RELEVANCE_GRADING", "true").lower() == "true"
    )
    kb_relevance_threshold: float = field(
        default_factory=lambda: float(_env("KB_RELEVANCE_THRESHOLD", "0.5"))
    )
    # Defaults to BEDROCK_MODEL_ID; a smaller model (e.g. Haiku) is enough for grading
    kb_relevance_model_id: str | None = field(default_factory=lambda: _env("KB_RELEVANCE_MODEL_ID"))

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

    # Hooks
    enforce_cache_first: bool = field(
        default_factory=lambda: _env("ENFORCE_CACHE_FIRST", "true").lower() == "true"
    )
    live_log: bool = field(default_factory=lambda: _env("LIVE_LOG", "true").lower() == "true")

    # Human-in-the-loop (interrupts): ask before external searches / let the agent ask questions
    hitl_approval: bool = field(default_factory=lambda: _env("HITL_APPROVAL", "true").lower() == "true")
    hitl_approval_tools: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            name.strip()
            for name in _env("HITL_APPROVAL_TOOLS", "search_web,research_with_sonar").split(",")
            if name.strip()
        )
    )
    hitl_clarify: bool = field(default_factory=lambda: _env("HITL_CLARIFY", "true").lower() == "true")

    # Session persistence (FileSessionManager): conversation + agent state (e.g. trusted tools)
    session_id: str | None = field(default_factory=lambda: _env("SESSION_ID"))
    session_dir: str = field(default_factory=lambda: _env("SESSION_DIR", ".sessions"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # Observability (OpenTelemetry) - see telemetry.py
    otel_traces_file: str | None = field(default_factory=lambda: _env("OTEL_TRACES_FILE", "traces.jsonl"))
    otel_console_export: bool = field(
        default_factory=lambda: _env("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
    )
    otel_exporter_otlp_endpoint: str | None = field(
        default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    langfuse_public_key: str | None = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str | None = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", "https://cloud.langfuse.com"))


settings = Settings()
