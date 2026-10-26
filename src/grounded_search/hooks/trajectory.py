"""Hook that records the agent's trajectory: which tools it picks, in what order, and how often."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    HookProvider,
    HookRegistry,
)

from .interrupts import is_resume


@dataclass
class ToolStep:
    tool: str
    input: dict
    status: str
    duration: float | None
    blocked_reason: str | None = None


@dataclass
class Trajectory:
    """Everything the agent decided during a single invocation."""

    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    model_calls: int = 0
    stop_reasons: list[str] = field(default_factory=list)
    steps: list[ToolStep] = field(default_factory=list)

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.tool] = counts.get(step.tool, 0) + 1
        return counts

    def reformulations(self) -> dict[str, list[str]]:
        """Distinct queries per tool, for tools called with more than one query."""
        queries: dict[str, list[str]] = {}
        for step in self.steps:
            query = step.input.get("query")
            if query is None:
                continue
            seen = queries.setdefault(step.tool, [])
            if query not in seen:
                seen.append(query)
        return {tool: qs for tool, qs in queries.items() if len(qs) > 1}

    def render(self) -> str:
        elapsed = (self.finished_at or time.time()) - self.started_at
        lines = [f"Trajectory: {self.model_calls} model call(s), {len(self.steps)} tool call(s), {elapsed:.1f}s"]
        for index, step in enumerate(self.steps, start=1):
            args = repr(step.input["query"]) if "query" in step.input else ""
            duration = f"{step.duration:.2f}s" if step.duration is not None else "-"
            line = f"  {index}. {step.tool}({args}) -> {step.status} [{duration}]"
            if step.blocked_reason:
                line += f" BLOCKED: {step.blocked_reason}"
            lines.append(line)
        for tool, queries in self.reformulations().items():
            lines.append(f"  {tool} reformulated {len(queries) - 1} time(s)")
        return "\n".join(lines)


class TrajectoryRecorder(HookProvider):
    """Observes the agent loop without changing it.

    A new Trajectory starts on every invocation (each `agent(...)` call); the previous one stays
    available in `history`.
    """

    def __init__(self) -> None:
        self.current: Trajectory | None = None
        self.history: list[Trajectory] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_invocation_start)
        registry.add_callback(BeforeModelCallEvent, self._on_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_model_response)
        registry.add_callback(AfterToolCallEvent, self._on_tool_result)
        registry.add_callback(AfterInvocationEvent, self._on_invocation_end)

    def _on_invocation_start(self, event: BeforeInvocationEvent) -> None:
        if is_resume(event) and self.current:
            return
        self.current = Trajectory()
        self.history.append(self.current)

    def _on_model_call(self, event: BeforeModelCallEvent) -> None:
        if self.current:
            self.current.model_calls += 1

    def _on_model_response(self, event: AfterModelCallEvent) -> None:
        if self.current and event.stop_response:
            self.current.stop_reasons.append(event.stop_response.stop_reason)

    def _on_tool_result(self, event: AfterToolCallEvent) -> None:
        if not self.current:
            return
        tool_input = event.tool_use.get("input")
        self.current.steps.append(
            ToolStep(
                tool=event.tool_use["name"],
                input=tool_input if isinstance(tool_input, dict) else {},
                status=event.result.get("status", "unknown"),
                duration=event.duration,
                blocked_reason=event.cancel_message,
            )
        )

    def _on_invocation_end(self, event: AfterInvocationEvent) -> None:
        if self.current:
            self.current.finished_at = time.time()
