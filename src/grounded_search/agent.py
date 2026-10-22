"""Builds the main agent (Strands) with Claude Sonnet 5.5 (Amazon Bedrock) as the orchestrator LLM."""
from __future__ import annotations

import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel

from .config import settings
from .prompts import SYSTEM_PROMPT
from .tools import research_with_sonar, search_knowledge_base, search_research_cache, search_web


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


def build_agent() -> Agent:
    """Creates the orchestrator agent.

    The LLM itself decides which tool to use, in what order, whether it needs another query, and
    when to stop. Evaluating the evidence (contradictions, sources, sufficiency) is also the
    agent's own responsibility, not a separate component's: see prompts/system_prompt.py.
    """
    model = _build_bedrock_model()
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_with_sonar, search_web, search_knowledge_base, search_research_cache],
    )
