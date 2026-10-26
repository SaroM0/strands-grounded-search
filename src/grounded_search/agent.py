"""Builds the main agent (Strands) with Claude Sonnet 5.5 (Amazon Bedrock) as the orchestrator LLM."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import boto3
from strands import Agent
from strands.hooks import HookProvider
from strands.models.bedrock import BedrockModel

from .config import settings
from .hooks import CacheFirstPolicy, LiveLogger
from .prompts import SYSTEM_PROMPT
from .schemas import GroundedAnswer
from .telemetry import setup_telemetry
from .tools import research_with_sonar, search_knowledge_base, search_research_cache, search_web

logger = logging.getLogger(__name__)


def _build_bedrock_model() -> BedrockModel:
    """Builds the Bedrock model based on the available credentials.

    Prioritizes the bearer token (Bedrock API key) over the access/secret key pair,
    since it's the auth mechanism AWS recommends for this use case.
    """
    if settings.aws_bearer_token_bedrock:
        return BedrockModel(
            model_id=settings.bedrock_model_id,
            region_name=settings.aws_region_bedrock,
            api_key=settings.aws_bearer_token_bedrock,
            max_tokens=4096,
        )

    session = boto3.Session(
        aws_access_key_id=settings.aws_public_key_bedrock,
        aws_secret_access_key=settings.aws_private_key_bedrock,
        region_name=settings.aws_region_bedrock,
    )
    return BedrockModel(
        model_id=settings.bedrock_model_id,
        boto_session=session,
        max_tokens=4096,
    )


def build_agent(hooks: Sequence[HookProvider] = ()) -> Agent:
    """Creates the orchestrator agent.

    The LLM itself decides which tool to use, in what order, whether it needs another query, and
    when to stop. Evaluating the evidence (contradictions, sources, sufficiency) is also the
    agent's own responsibility, not a separate component's: see prompts/system_prompt.py.

    Every invocation returns a validated GroundedAnswer in `result.structured_output`: Strands
    exposes the schema to the model as a tool and re-prompts it if validation fails.

    Args:
        hooks: Extra hook providers (e.g. a TrajectoryRecorder) to observe the agent loop.
    """
    exporters = setup_telemetry()
    model = _build_bedrock_model()
    tools = [research_with_sonar, search_web, search_knowledge_base, search_research_cache]
    all_hooks = list(hooks)
    if settings.live_log:
        all_hooks.append(LiveLogger())
    if settings.enforce_cache_first:
        all_hooks.append(CacheFirstPolicy())

    logger.info("model: %s (region %s)", settings.bedrock_model_id, settings.aws_region_bedrock)
    logger.info("tools: %s", ", ".join(t.tool_name for t in tools))
    logger.info("hooks: %s", ", ".join(type(h).__name__ for h in all_hooks) or "none")
    logger.info("structured output: %s", GroundedAnswer.__name__)
    logger.info("traces: %s", ", ".join(exporters) or "disabled")

    # LiveLogger prints each step; the default handler would also stream raw tokens on top.
    extra = {"callback_handler": None} if settings.live_log else {}
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        hooks=all_hooks,
        structured_output_model=GroundedAnswer,
        trace_attributes={"service.component": "grounded-search-orchestrator"},
        **extra,
    )
