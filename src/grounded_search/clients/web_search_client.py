"""Interchangeable clients for raw web search (used by search_web)."""
from __future__ import annotations

from typing import Protocol

import httpx

from ..config import settings
from ..schemas import Source, ToolResult


class WebSearchClient(Protocol):
    def search(self, query: str, max_results: int) -> ToolResult: ...


class TavilyClient:
    _API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.tavily_api_key

    def search(self, query: str, max_results: int) -> ToolResult:
        if not self._api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self._API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        sources = [
            Source(title=item.get("title", ""), url=item.get("url"), snippet=item.get("content"))
            for item in data.get("results", [])
        ]
        content = "\n\n".join(f"- {source.title}: {source.snippet}" for source in sources)
        return ToolResult(content=content, sources=sources)


class BraveClient:
    _API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.brave_api_key

    def search(self, query: str, max_results: int) -> ToolResult:
        if not self._api_key:
            raise RuntimeError("BRAVE_API_KEY is not configured.")

        headers = {"X-Subscription-Token": self._api_key, "Accept": "application/json"}
        params = {"q": query, "count": max_results}
        with httpx.Client(timeout=30.0) as client:
            response = client.get(self._API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        results = data.get("web", {}).get("results", [])
        sources = [
            Source(title=item.get("title", ""), url=item.get("url"), snippet=item.get("description"))
            for item in results
        ]
        content = "\n\n".join(f"- {source.title}: {source.snippet}" for source in sources)
        return ToolResult(content=content, sources=sources)


def get_web_search_client() -> WebSearchClient:
    provider = settings.web_search_provider.lower()
    if provider == "tavily":
        return TavilyClient()
    if provider == "brave":
        return BraveClient()
    raise ValueError(f"Unsupported web search provider: {provider}")
