"""Types shared across the research tools."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


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


class Citation(BaseModel):
    """A claim in the answer and the source that backs it."""

    claim: str = Field(description="The specific claim being backed, in the answer's language.")
    source: str = Field(description="URL or knowledge-base document identifier returned by a tool.")
    tool: Literal["research_with_sonar", "search_web", "search_knowledge_base", "search_research_cache"] = (
        Field(description="Tool that returned this source.")
    )


class GroundedAnswer(BaseModel):
    """Final answer of the agent. Strands forces the model to produce it via structured output."""

    answer: str = Field(description="Direct answer to the question, in the same language it was asked.")
    citations: list[Citation] = Field(
        description="One entry per relevant claim. Only sources actually returned by the tools."
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Parts of the answer that are weaker, contested, or backed by few sources.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="What couldn't be verified or fell outside the scope of the available tools.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Overall confidence given the evidence gathered."
    )

    def render(self) -> str:
        lines = [self.answer, "", "Citations:"]
        lines += [f"  [{i}] {c.claim} - {c.source} ({c.tool})" for i, c in enumerate(self.citations, 1)]
        if self.uncertainties:
            lines += ["", "Uncertainties:"] + [f"  - {u}" for u in self.uncertainties]
        if self.limitations:
            lines += ["", "Limitations:"] + [f"  - {item}" for item in self.limitations]
        lines += ["", f"Confidence: {self.confidence}"]
        return "\n".join(lines)
