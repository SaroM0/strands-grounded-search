"""Construcción del agente principal (Strands) con Claude Sonnet 5.5 como LLM orquestador."""
from __future__ import annotations

from strands import Agent
from strands.models.anthropic import AnthropicModel

from .config import settings
from .prompts import SYSTEM_PROMPT
from .tools import research_with_sonar, search_knowledge_base, search_web


def build_agent() -> Agent:
    """Crea el agente orquestador.

    El propio LLM decide qué herramienta usar, en qué orden, si necesita otra consulta y cuándo
    detenerse. La evaluación de evidencia (contradicciones, fuentes, suficiencia) también es
    responsabilidad del agente, no de un componente separado: ver prompts/system_prompt.py.
    """
    model = AnthropicModel(
        client_args={"api_key": settings.anthropic_api_key},
        model_id=settings.anthropic_model_id,
        max_tokens=4096,
    )
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_with_sonar, search_web, search_knowledge_base],
    )
