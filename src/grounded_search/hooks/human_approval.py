"""Human-in-the-loop hook: pauses the agent until a human approves an external search."""
from __future__ import annotations

from collections.abc import Iterable

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

INTERRUPT_NAME = "grounded-search-approval"
TRUSTED_TOOLS_KEY = "hitl_trusted_tools"
OPTIONS = "y = approve once, t = trust this tool for the session, n = deny (or type feedback for the agent)"

_APPROVE = {"y", "yes", "s", "si", "sí"}
_TRUST = {"t", "trust"}
_DENY = {"", "n", "no"}


def denied_message(tool: str, feedback: str | None = None) -> str:
    message = (
        f"The user did not approve this {tool} call, so it was not executed. Don't retry the same "
        "call: answer with the evidence you already have, or use another tool."
    )
    if feedback:
        message += f" User feedback: {feedback}"
    return message


class HumanApproval(HookProvider):
    """Raises an interrupt before every call to a gated tool and waits for the human's answer.

    The agent returns with `stop_reason == "interrupt"`; the caller resumes it with
    `agent([{"interruptResponse": {"interruptId": ..., "response": "y"}}])` (see cli.py).
    "t" stores the tool in `agent.state`, so it's auto-approved for the rest of the session (and
    across runs when a session manager persists the state). Anything other than y/t denies the
    call through `cancel_tool`; free text is forwarded to the model as feedback.

    Register it after CacheFirstPolicy: a call that policy already cancelled is not asked about.
    """

    def __init__(self, tools: Iterable[str]) -> None:
        self.tools = frozenset(tools)

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeToolCallEvent, self._approve)

    def _approve(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        if name not in self.tools or event.cancel_tool:
            return
        trusted = event.agent.state.get(TRUSTED_TOOLS_KEY) or []
        if name in trusted:
            return

        response = event.interrupt(
            INTERRUPT_NAME,
            reason={"tool": name, "input": event.tool_use["input"], "options": OPTIONS},
        )
        answer = str(response or "").strip()
        choice = answer.lower()
        if choice in _APPROVE:
            return
        if choice in _TRUST:
            event.agent.state.set(TRUSTED_TOOLS_KEY, [*trusted, name])
            return
        event.cancel_tool = denied_message(name, None if choice in _DENY else answer)
