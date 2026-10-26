"""Hook that forces the agent to check the research cache before hitting an external API."""
from __future__ import annotations

from strands.hooks import (
    AfterToolsEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    BeforeToolsEvent,
    HookProvider,
    HookRegistry,
)

from .interrupts import is_resume

CACHE_TOOL = "search_research_cache"
EXTERNAL_TOOLS = frozenset({"search_web", "research_with_sonar"})

BLOCKED_MESSAGE = (
    "Not executed yet (temporary, not a prohibition): the research cache must be checked in an "
    "earlier turn than any external search. Read the search_research_cache result first. After "
    "that, you are free to call search_web or research_with_sonar, e.g. if the cache has no good "
    "hit, the hits are old, or the claim is time-sensitive. Don't report this as a limitation."
)


class CacheFirstPolicy(HookProvider):
    """Cancels search_web / research_with_sonar until search_research_cache was called.

    This is an experiment on the writable `cancel_tool` field: the system prompt only *suggests*
    checking the cache first, this hook *enforces* it. The cancelled call reaches the model as a
    tool error carrying BLOCKED_MESSAGE, so you can see how it adapts.

    The cache must have been called in an *earlier* model turn: calling it in parallel with an
    external search (same assistant message) doesn't count, because the model hasn't seen the
    cache result when it decides to search.
    """

    def __init__(self) -> None:
        self.cache_consulted = False
        self._cache_in_batch = False

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeInvocationEvent, self._reset)
        registry.add_callback(BeforeToolsEvent, self._start_batch)
        registry.add_callback(BeforeToolCallEvent, self._gate)
        registry.add_callback(AfterToolsEvent, self._end_batch)

    def _reset(self, event: BeforeInvocationEvent) -> None:
        if is_resume(event):
            return
        self.cache_consulted = False
        self._cache_in_batch = False

    def _start_batch(self, event: BeforeToolsEvent) -> None:
        self._cache_in_batch = False

    def _gate(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        if name == CACHE_TOOL:
            self._cache_in_batch = True
        elif name in EXTERNAL_TOOLS and not self.cache_consulted:
            event.cancel_tool = BLOCKED_MESSAGE

    def _end_batch(self, event: AfterToolsEvent) -> None:
        if self._cache_in_batch:
            self.cache_consulted = True
