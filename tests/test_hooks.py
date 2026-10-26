"""Runs a real Strands Agent loop against a scripted fake model to exercise the hooks."""
import json

from strands import Agent, tool
from strands.models.model import Model

from grounded_search.hooks import CacheFirstPolicy, TrajectoryRecorder
from grounded_search.hooks.cache_first import BLOCKED_MESSAGE
from grounded_search.schemas import GroundedAnswer


class _ScriptedModel(Model):
    """Replays one list of tool calls per model turn: [(tool_name, input), ...]."""

    def __init__(self, turns: list[list[tuple[str, dict]]]) -> None:
        self.turns = list(turns)
        self.seen_messages: list = []

    def update_config(self, **model_config: object) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.seen_messages = list(messages)
        calls = self.turns.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        for index, (name, tool_input) in enumerate(calls):
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": f"{name}-{index}-{len(self.turns)}"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


tool_calls: list[str] = []


@tool
def search_research_cache(query: str) -> dict:
    """Fake cache lookup.

    Args:
        query: The question.
    """
    tool_calls.append("search_research_cache")
    return {"hits": []}


@tool
def search_web(query: str) -> dict:
    """Fake web search.

    Args:
        query: The search terms.
    """
    tool_calls.append("search_web")
    return {"content": "result", "sources": [{"title": "t", "url": "https://example.com"}]}


FINAL_ANSWER = (
    "GroundedAnswer",
    {
        "answer": "42",
        "citations": [{"claim": "it is 42", "source": "https://example.com", "tool": "search_web"}],
        "confidence": "medium",
    },
)


def _run(turns, hooks):
    tool_calls.clear()
    agent = Agent(
        model=_ScriptedModel(turns),
        tools=[search_research_cache, search_web],
        hooks=hooks,
        structured_output_model=GroundedAnswer,
        callback_handler=None,
    )
    return agent, agent("question?")


def test_trajectory_records_order_and_reformulations() -> None:
    recorder = TrajectoryRecorder()
    _, result = _run(
        [
            [("search_research_cache", {"query": "q1"})],
            [("search_web", {"query": "q1"})],
            [("search_web", {"query": "q1 rephrased"})],
            [FINAL_ANSWER],
        ],
        [recorder],
    )

    trajectory = recorder.current
    assert [step.tool for step in trajectory.steps] == [
        "search_research_cache",
        "search_web",
        "search_web",
        "GroundedAnswer",
    ]
    assert trajectory.model_calls == 4
    assert trajectory.tool_counts()["search_web"] == 2
    assert trajectory.reformulations() == {"search_web": ["q1", "q1 rephrased"]}
    assert trajectory.finished_at is not None
    assert isinstance(result.structured_output, GroundedAnswer)
    assert result.structured_output.answer == "42"


def test_cache_first_blocks_external_search_until_cache_is_checked() -> None:
    recorder = TrajectoryRecorder()
    agent, _ = _run(
        [
            [("search_web", {"query": "q"})],
            [("search_research_cache", {"query": "q"})],
            [("search_web", {"query": "q"})],
            [FINAL_ANSWER],
        ],
        [recorder, CacheFirstPolicy()],
    )

    assert tool_calls == ["search_research_cache", "search_web"]
    first = recorder.current.steps[0]
    assert first.tool == "search_web"
    assert first.status == "error"
    assert first.blocked_reason == BLOCKED_MESSAGE
    assert recorder.current.steps[2].status == "success"
    # The model saw the block as a tool error and adapted.
    assert BLOCKED_MESSAGE in json.dumps(agent.messages)


def test_cache_first_blocks_search_in_parallel_with_cache() -> None:
    # The model can't have read the cache result yet, so the parallel search is blocked;
    # the next turn, after the cache result is in context, is allowed.
    recorder = TrajectoryRecorder()
    _run(
        [
            [("search_research_cache", {"query": "q"}), ("search_web", {"query": "q"})],
            [("search_web", {"query": "q"})],
            [FINAL_ANSWER],
        ],
        [recorder, CacheFirstPolicy()],
    )

    assert tool_calls == ["search_research_cache", "search_web"]
    statuses = {(step.tool, step.status) for step in recorder.current.steps[:2]}
    assert statuses == {("search_research_cache", "success"), ("search_web", "error")}
    assert recorder.current.steps[2].status == "success"


def test_cache_first_resets_on_each_invocation() -> None:
    policy = CacheFirstPolicy()
    agent, _ = _run(
        [[("search_research_cache", {"query": "q"})], [FINAL_ANSWER]],
        [policy],
    )
    assert policy.cache_consulted

    agent.model.turns = [[("search_web", {"query": "q2"})], [FINAL_ANSWER]]
    tool_calls.clear()
    agent("another question?")
    assert tool_calls == []
