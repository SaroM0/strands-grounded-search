"""Hook that keeps what the tools actually returned, so citations can be checked against it."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from strands.hooks import AfterToolCallEvent, BeforeInvocationEvent, HookProvider, HookRegistry

_MAX_EVIDENCE_CHARS = 6000


@dataclass
class ReturnedSource:
    tool: str
    title: str | None
    url: str | None


@dataclass
class Evidence:
    tool: str
    query: str | None
    content: str


def _payloads(result: dict) -> list[dict]:
    payloads = []
    for block in result.get("content", []):
        try:
            payload = json.loads(block.get("text", ""))
        except ValueError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


class SourceCollector(HookProvider):
    """Records, per invocation, every source and content chunk returned by a successful tool call.

    Cache hits are flattened: their sources count as returned by search_research_cache, which is
    the tool the model has to cite for them.
    """

    def __init__(self) -> None:
        self.sources: list[ReturnedSource] = []
        self.evidence: list[Evidence] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeInvocationEvent, self._reset)
        registry.add_callback(AfterToolCallEvent, self._on_tool_result)

    def _reset(self, event: BeforeInvocationEvent) -> None:
        self.sources = []
        self.evidence = []

    def _on_tool_result(self, event: AfterToolCallEvent) -> None:
        if event.cancel_message or event.result.get("status") != "success":
            return
        name = event.tool_use["name"]
        tool_input = event.tool_use.get("input")
        query = tool_input.get("query") if isinstance(tool_input, dict) else None
        for payload in _payloads(event.result):
            for entry in [payload, *payload.get("results", [])]:
                # The model sees the snippets too, so they count as evidence alongside "content".
                parts = [str(entry["content"])] if entry.get("content") else []
                for source in entry.get("sources") or []:
                    self.sources.append(ReturnedSource(name, source.get("title"), source.get("url")))
                    if source.get("snippet"):
                        parts.append(f"- {source.get('title')} <{source.get('url') or 'no url'}>: {source['snippet']}")
                if parts:
                    self.evidence.append(Evidence(name, query, "\n".join(parts)[:_MAX_EVIDENCE_CHARS]))

    def identifiers(self) -> set[str]:
        """Normalized URLs and titles, i.e. everything a citation's `source` may legitimately be."""
        return {_normalize(value) for source in self.sources for value in (source.url, source.title) if value}


def _normalize(value: str) -> str:
    value = value.strip().lower().rstrip("/")
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value.removeprefix("www.")


def is_grounded(citation_source: str, identifiers: set[str]) -> bool:
    return _normalize(citation_source) in identifiers
