"""Eval metrics computed from a real Strands loop driven by the scripted fake model."""
import json

import pytest
from strands import Agent

from grounded_search.evals import EvalCase, SourceCollector, compute_case_metrics, load_dataset, summarize
from grounded_search.evals.collector import is_grounded
from grounded_search.hooks import CacheFirstPolicy, TrajectoryRecorder
from grounded_search.schemas import GroundedAnswer

from .test_hooks import FINAL_ANSWER, _ScriptedModel, search_research_cache, search_web

ANSWER_WITH_FAKE_CITATION = (
    "GroundedAnswer",
    {
        "answer": "42",
        "citations": [
            {"claim": "it is 42", "source": "https://example.com/", "tool": "search_web"},
            {"claim": "made up", "source": "https://not-returned.org", "tool": "search_web"},
        ],
        "confidence": "high",
    },
)


def _evaluate(turns, case, hooks=()):
    recorder, collector = TrajectoryRecorder(), SourceCollector()
    agent = Agent(
        model=_ScriptedModel(turns),
        tools=[search_research_cache, search_web],
        hooks=[recorder, collector, *hooks],
        structured_output_model=GroundedAnswer,
        callback_handler=None,
    )
    result = agent(case.question)
    return compute_case_metrics(
        case,
        "test",
        recorder.current,
        collector,
        result.structured_output,
        result.metrics.accumulated_usage,
        latency_s=1.0,
    )


def test_metrics_from_trajectory_and_citations() -> None:
    case = EvalCase(id="c1", question="question?", expected_tools=["search_web"])
    metrics = _evaluate(
        [
            [("search_web", {"query": "q"})],
            [("search_research_cache", {"query": "q"})],
            [("search_web", {"query": "q"}), ("search_web", {"query": "q again"})],
            [ANSWER_WITH_FAKE_CITATION],
        ],
        case,
        hooks=[CacheFirstPolicy()],
    )

    assert metrics.tool_calls == 4  # GroundedAnswer is not counted as a tool call
    assert metrics.tool_counts == {"search_web": 3, "search_research_cache": 1}
    assert metrics.blocked_calls == 1
    assert metrics.external_calls == 2
    assert metrics.reformulations == 1
    assert metrics.cache_checked_first is False
    assert metrics.expected_tools_used is True
    assert metrics.model_calls == 4
    assert metrics.has_answer and metrics.confidence == "high"
    assert (metrics.citations, metrics.grounded_citations) == (2, 1)
    assert metrics.citation_precision == 0.5
    assert metrics.abstained is None


def test_unanswerable_case_without_abstention() -> None:
    case = EvalCase(id="c2", question="question?", answerable=False, expected_tools=["search_knowledge_base"])
    metrics = _evaluate([[("search_research_cache", {"query": "q"})], [FINAL_ANSWER]], case)

    assert metrics.cache_checked_first is True
    assert metrics.expected_tools_used is False
    assert metrics.citation_precision == 0.0  # the cache hit list was empty, so nothing was returned
    assert metrics.abstained is False  # confidence=medium and no limitations


def test_collector_flattens_cache_hits() -> None:
    collector = SourceCollector()
    payload = {"cache_hit": True, "results": [{"content": "old", "sources": [{"title": "Doc", "url": None}]}]}

    class _Event:
        cancel_message = None
        tool_use = {"name": "search_research_cache", "input": {"query": "q"}}
        result = {"status": "success", "content": [{"text": json.dumps(payload)}]}

    collector._on_tool_result(_Event())
    assert [(s.tool, s.title) for s in collector.sources] == [("search_research_cache", "Doc")]
    assert collector.evidence[0].content == "old"
    assert is_grounded(" doc ", collector.identifiers())


def test_summarize_ignores_metrics_that_do_not_apply() -> None:
    case = EvalCase(id="c3", question="question?")
    ok = _evaluate([[("search_web", {"query": "q"})], [FINAL_ANSWER]], case)
    failed = compute_case_metrics(case, "test", None, SourceCollector(), None, None, 0.0, error="boom")

    summary = summarize([ok, failed])["test"]
    assert summary["cases"] == 2 and summary["errors"] == 1
    assert summary["answer_rate"] == 0.5
    assert summary["citation_precision"] == 1.0
    assert summary["expected_tools_rate"] is None
    assert summary["judge_score"] is None


def test_load_dataset_rejects_duplicates_and_unknown_fields(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "question": "q"}\n{"id": "a", "question": "q2"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)

    path.write_text('{"id": "a", "question": "q", "typo_field": 1}\n')
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(path)


def test_bundled_dataset_loads() -> None:
    assert load_dataset("evals/dataset.jsonl")
