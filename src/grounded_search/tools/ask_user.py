"""Tool: ask_user (human-in-the-loop clarification, raised from inside the tool)."""
from __future__ import annotations

from strands import tool
from strands.types.tools import ToolContext

INTERRUPT_NAME = "grounded-search-clarification"

NO_ANSWER = (
    "The user didn't answer. Proceed with the most reasonable interpretation and state the "
    "assumption you made in the limitations."
)


@tool(context=True)
def ask_user(question: str, tool_context: ToolContext) -> str:
    """
    Asks the user a clarifying question and waits for the answer.

    Use it only when the question is ambiguous in a way that changes what you would research
    (e.g. which entity, time period, or region is meant), before spending searches on the
    wrong interpretation. Don't use it to ask for permission or to confirm an answer.

    Args:
        question: One short, specific question, in the user's language.

    Returns:
        The user's answer.
    """
    answer = tool_context.interrupt(INTERRUPT_NAME, reason={"question": question})
    answer = str(answer or "").strip()
    return answer or NO_ANSWER
