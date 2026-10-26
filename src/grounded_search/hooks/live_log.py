"""Hook that logs every step of the agent loop to the console as it happens."""
from __future__ import annotations

import json
import logging
import time

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

logger = logging.getLogger("grounded_search.agent_loop")

_MAX_TEXT = 300
_MAX_SOURCES = 3
_FINAL_ANSWER_TOOL = "GroundedAnswer"


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize_payload(payload: dict) -> list[str]:
    """Human-readable lines for the dicts our tools return."""
    lines: list[str] = []
    if payload.get("error"):
        lines.append(f"ERROR: {payload['error']}")
    if "cache_hit" in payload:
        results = payload.get("results", [])
        lines.append(f"cache_hit={payload['cache_hit']} ({len(results)} entr{'y' if len(results) == 1 else 'ies'})")
        for entry in results:
            age_min = entry.get("age_seconds", 0) / 60
            lines.append(
                f"  - score={entry.get('score', 0):.3f} tool={entry.get('tool')} age={age_min:.0f}min "
                f"query={entry.get('query')!r}"
            )
    if payload.get("content"):
        lines.append(f"content: {_truncate(str(payload['content']))}")
    sources = payload.get("sources") or []
    if sources:
        lines.append(f"{len(sources)} source(s):")
        for source in sources[:_MAX_SOURCES]:
            lines.append(f"  - {source.get('title')} <{source.get('url') or 'no url'}>")
        if len(sources) > _MAX_SOURCES:
            lines.append(f"  - ... {len(sources) - _MAX_SOURCES} more")
    return lines


def _summarize_result(result: dict) -> list[str]:
    lines: list[str] = []
    for block in result.get("content", []):
        text = block.get("text")
        if text is None:
            lines.append(f"<{', '.join(block)} block>")
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            lines += _summarize_payload(payload) or [_truncate(text)]
        else:
            lines.append(_truncate(text))
    return lines


class LiveLogger(HookProvider):
    """Logs what the agent is doing: each model turn, what it said/thought, each tool call and result.

    It replaces Strands' default streaming callback handler in the CLI, so the console shows one
    structured line per step instead of raw token deltas.
    """

    def __init__(self) -> None:
        self._turn = 0
        self._started_at = 0.0
        self._model_started_at = 0.0
        self._tool_started_at: dict[str, float] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_invocation_start)
        registry.add_callback(BeforeModelCallEvent, self._on_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_model_response)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_call)
        registry.add_callback(AfterToolCallEvent, self._on_tool_result)
        registry.add_callback(AfterInvocationEvent, self._on_invocation_end)

    def _on_invocation_start(self, event: BeforeInvocationEvent) -> None:
        self._turn = 0
        self._started_at = time.monotonic()
        logger.info("=" * 72)
        logger.info("INVOCATION START | %d message(s) already in history", len(event.agent.messages))

    def _on_model_call(self, event: BeforeModelCallEvent) -> None:
        self._turn += 1
        self._model_started_at = time.monotonic()
        tokens = f", ~{event.projected_input_tokens} input tokens" if event.projected_input_tokens else ""
        logger.info("-" * 72)
        logger.info("[turn %d] MODEL CALL -> %d message(s) in context%s", self._turn, len(event.agent.messages), tokens)

    def _on_model_response(self, event: AfterModelCallEvent) -> None:
        elapsed = time.monotonic() - self._model_started_at
        if event.exception:
            logger.error("[turn %d] MODEL ERROR after %.1fs: %s", self._turn, elapsed, event.exception)
            return
        if not event.stop_response:
            return
        message = event.stop_response.message
        logger.info("[turn %d] MODEL RESPONSE in %.1fs | stop_reason=%s", self._turn, elapsed, event.stop_response.stop_reason)
        for block in message.get("content", []):
            if "reasoningContent" in block:
                reasoning = block["reasoningContent"].get("reasoningText", {}).get("text")
                if reasoning:
                    logger.info("    thinking: %s", _truncate(reasoning, 600))
            elif "text" in block and block["text"].strip():
                logger.info("    says: %s", _truncate(block["text"], 600))
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_input = tool_use.get("input") or {}
                if tool_use["name"] == _FINAL_ANSWER_TOOL:
                    logger.info(
                        "    decides to answer: %s (%d citation(s), confidence=%s)",
                        _FINAL_ANSWER_TOOL,
                        len(tool_input.get("citations", [])),
                        tool_input.get("confidence"),
                    )
                else:
                    logger.info("    decides to call: %s(%s)", tool_use["name"], _truncate(json.dumps(tool_input, ensure_ascii=False)))

    def _on_tool_call(self, event: BeforeToolCallEvent) -> None:
        self._tool_started_at[event.tool_use["toolUseId"]] = time.monotonic()
        logger.info("  >> TOOL %s START", event.tool_use["name"])

    def _on_tool_result(self, event: AfterToolCallEvent) -> None:
        name = event.tool_use["name"]
        started = self._tool_started_at.pop(event.tool_use["toolUseId"], None)
        elapsed = time.monotonic() - started if started else 0.0
        if event.cancel_message:
            logger.warning("  << TOOL %s BLOCKED by hook: %s", name, event.cancel_message)
            return
        status = event.result.get("status")
        log = logger.info if status == "success" else logger.warning
        log("  << TOOL %s %s in %.2fs", name, status.upper() if status else "?", elapsed)
        if event.exception:
            logger.warning("     exception: %r", event.exception)
        for line in _summarize_result(event.result):
            log("     %s", line)

    def _on_invocation_end(self, event: AfterInvocationEvent) -> None:
        elapsed = time.monotonic() - self._started_at
        logger.info("-" * 72)
        result = event.result
        if result is None:
            logger.info("INVOCATION END in %.1fs (no result)", elapsed)
            return
        usage = result.metrics.accumulated_usage
        logger.info(
            "INVOCATION END in %.1fs | %d model turn(s) | stop_reason=%s | tokens in=%s out=%s (session total)",
            elapsed,
            self._turn,
            result.stop_reason,
            usage.get("inputTokens"),
            usage.get("outputTokens"),
        )
