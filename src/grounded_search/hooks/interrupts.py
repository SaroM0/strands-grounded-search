"""Helpers shared by the hooks that need to tell a fresh invocation from an interrupt resume."""
from __future__ import annotations

from strands.hooks import BeforeInvocationEvent


def is_resume(event: BeforeInvocationEvent) -> bool:
    """True when the invocation resumes an interrupted one (`agent([{"interruptResponse": ...}])`).

    Strands fires BeforeInvocationEvent again on resume, but the interrupt responses are consumed
    before the event, so it carries no new messages. Per-invocation state (cache consulted, the
    current trajectory, turn counters) must survive it: it's still the same question.
    """
    return not event.messages
