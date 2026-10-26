"""Human-in-the-loop: interrupts raised by HumanApproval (hook) and ask_user (tool), then resumed."""
import json

from strands import Agent
from strands.session import FileSessionManager

from grounded_search.hooks import CacheFirstPolicy, HumanApproval, TrajectoryRecorder
from grounded_search.hooks.human_approval import INTERRUPT_NAME, TRUSTED_TOOLS_KEY, denied_message
from grounded_search.schemas import GroundedAnswer
from grounded_search.tools import ask_user
from grounded_search.tools.ask_user import INTERRUPT_NAME as CLARIFY_INTERRUPT

from .test_hooks import FINAL_ANSWER, _ScriptedModel, search_research_cache, search_web, tool_calls


def _agent(turns, hooks, **kwargs) -> Agent:
    tool_calls.clear()
    return Agent(
        model=_ScriptedModel(turns),
        tools=[search_research_cache, search_web, ask_user],
        hooks=hooks,
        structured_output_model=GroundedAnswer,
        callback_handler=None,
        **kwargs,
    )


def _respond(agent: Agent, result, response: str):
    return agent(
        [{"interruptResponse": {"interruptId": interrupt.id, "response": response}} for interrupt in result.interrupts]
    )


def test_approval_pauses_then_runs_the_tool_on_yes() -> None:
    agent = _agent([[("search_web", {"query": "q"})], [FINAL_ANSWER]], [HumanApproval({"search_web"})])

    result = agent("question?")
    assert result.stop_reason == "interrupt"
    assert tool_calls == []
    (interrupt,) = result.interrupts
    assert interrupt.name == INTERRUPT_NAME
    assert interrupt.reason["tool"] == "search_web"
    assert interrupt.reason["input"] == {"query": "q"}

    result = _respond(agent, result, "y")
    assert tool_calls == ["search_web"]
    assert isinstance(result.structured_output, GroundedAnswer)


def test_denial_cancels_the_tool_and_forwards_feedback() -> None:
    agent = _agent([[("search_web", {"query": "q"})], [FINAL_ANSWER]], [HumanApproval({"search_web"})])

    result = _respond(agent, agent("question?"), "use my documents instead")
    assert tool_calls == []
    assert denied_message("search_web", "use my documents instead") in json.dumps(agent.messages)
    assert result.structured_output is not None


def test_ungated_tools_run_without_asking() -> None:
    agent = _agent([[("search_research_cache", {"query": "q"})], [FINAL_ANSWER]], [HumanApproval({"search_web"})])

    result = agent("question?")
    assert result.stop_reason != "interrupt"
    assert tool_calls == ["search_research_cache"]


def test_trust_auto_approves_later_calls() -> None:
    agent = _agent(
        [[("search_web", {"query": "q1"})], [("search_web", {"query": "q2"})], [FINAL_ANSWER]],
        [HumanApproval({"search_web"})],
    )

    result = _respond(agent, agent("question?"), "t")
    assert result.stop_reason != "interrupt"
    assert tool_calls == ["search_web", "search_web"]
    assert agent.state.get(TRUSTED_TOOLS_KEY) == ["search_web"]


def test_trust_survives_a_new_agent_with_the_same_session(tmp_path) -> None:
    def session():
        return FileSessionManager(session_id="hitl", storage_dir=str(tmp_path))

    agent = _agent([[("search_web", {"query": "q1"})], [FINAL_ANSWER]], [HumanApproval({"search_web"})], session_manager=session())
    _respond(agent, agent("question?"), "t")

    agent = _agent([[("search_web", {"query": "q2"})], [FINAL_ANSWER]], [HumanApproval({"search_web"})], session_manager=session())
    result = agent("another question?")
    assert result.stop_reason != "interrupt"
    assert tool_calls == ["search_web"]


def test_resume_keeps_cache_first_state_and_trajectory() -> None:
    # The cache was checked before the pause: the approved search must not be blocked on resume,
    # and the whole run stays one trajectory. Blocked calls are never sent to the human.
    recorder = TrajectoryRecorder()
    agent = _agent(
        [
            [("search_web", {"query": "q"})],
            [("search_research_cache", {"query": "q"})],
            [("search_web", {"query": "q"})],
            [FINAL_ANSWER],
        ],
        [recorder, CacheFirstPolicy(), HumanApproval({"search_web"})],
    )

    result = agent("question?")
    assert result.stop_reason == "interrupt"
    assert tool_calls == ["search_research_cache"]

    result = _respond(agent, result, "y")
    assert tool_calls == ["search_research_cache", "search_web"]
    assert len(recorder.history) == 1
    assert [(step.tool, step.status) for step in recorder.current.steps] == [
        ("search_web", "error"),
        ("search_research_cache", "success"),
        ("search_web", "success"),
        ("GroundedAnswer", "success"),
    ]


def test_ask_user_interrupts_with_the_question_and_returns_the_answer() -> None:
    agent = _agent([[("ask_user", {"question": "Which Mercury?"})], [FINAL_ANSWER]], [])

    result = agent("Tell me about Mercury")
    assert result.stop_reason == "interrupt"
    (interrupt,) = result.interrupts
    assert interrupt.name == CLARIFY_INTERRUPT
    assert interrupt.reason == {"question": "Which Mercury?"}

    result = _respond(agent, result, "the planet")
    assert "the planet" in json.dumps(agent.messages)
    assert isinstance(result.structured_output, GroundedAnswer)
