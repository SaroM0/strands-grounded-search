"""Minimal client for the Perplexity Sonar API (used by research_with_sonar)."""
from __future__ import annotations

import httpx

from ..config import settings
from ..schemas import Source, ToolResult

_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.perplexity_api_key
        self._model = model or settings.perplexity_model

    def research(self, query: str) -> ToolResult:
        if not self._api_key:
            raise RuntimeError("PERPLEXITY_API_KEY is not configured.")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research engine. Synthesize the available evidence, point "
                        "out agreements and disagreements between sources, and make clear "
                        "where each claim comes from."
                    ),
                },
                {"role": "user", "content": query},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(timeout=60.0) as client:
            response = client.post(_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        answer = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        sources = [Source(title=url, url=url) for url in citations]
        return ToolResult(content=answer, sources=sources)
