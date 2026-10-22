"""Tipos compartidos entre las tres herramientas de investigación."""
from __future__ import annotations

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
