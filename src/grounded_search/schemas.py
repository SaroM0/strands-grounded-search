"""Types shared across the research tools."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Source:
    title: str
    url: str | None = None
    snippet: str | None = None

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class ToolResult:
    content: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass
class CachedResearch:
    """A past search_web / research_with_sonar result retrieved from the research cache."""

    query: str
    tool: str
    cached_at: float
    score: float
    result: ToolResult

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tool": self.tool,
            "age_seconds": max(time.time() - self.cached_at, 0),
            "score": self.score,
            **self.result.to_dict(),
        }
