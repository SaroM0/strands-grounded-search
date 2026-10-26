import json

import pytest
from strands.models.model import Model

from grounded_search.clients.relevance_grader import RelevanceGrader, context_precision
from grounded_search.schemas import Source, ToolResult
from grounded_search.tools import knowledge_base


class _VerdictModel(Model):
    """Answers with a single RelevanceVerdicts structured-output tool call."""

    def __init__(self, relevant: list[bool]) -> None:
        self.relevant = relevant
        self.prompts: list[str] = []

    def update_config(self, **model_config: object) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.prompts.append(messages[0]["content"][0]["text"])
        verdicts = [{"index": i, "relevant": r} for i, r in enumerate(self.relevant, 1)]
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {"toolUse": {"name": "RelevanceVerdicts", "toolUseId": "v-1"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"verdicts": verdicts})}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


@pytest.mark.parametrize(
    ("relevant", "expected"),
    [
        ([], 0.0),
        ([False, False], 0.0),
        ([True, True, True], 1.0),
        ([True, False], 1.0),
        ([False, True], 0.5),
        ([True, False, True], (1 + 2 / 3) / 2),
    ],
)
def test_context_precision_rewards_relevant_passages_ranked_first(relevant, expected):
    assert context_precision(relevant) == pytest.approx(expected)


def test_grade_scores_the_llm_verdicts():
    model = _VerdictModel([False, True, False])
    grade = RelevanceGrader(model=model, threshold=0.5).grade("Who is my insurer?", ["a", "b", "c"])

    assert grade.relevant == [False, True, False]
    assert grade.score == pytest.approx(0.5)
    assert grade.to_dict() == {"relevant": "no", "score": 0.5, "threshold": 0.5, "relevant_passages": [2]}
    assert "PASSAGE 3:\nc" in model.prompts[0]


def test_grade_without_passages_is_not_relevant_and_skips_the_llm():
    model = _VerdictModel([])
    grade = RelevanceGrader(model=model, threshold=0.5).grade("q", [])

    assert not grade.is_relevant
    assert model.prompts == []


class _FailingGrader:
    def grade(self, question, passages):
        raise RuntimeError("bedrock down")


def test_failed_grading_is_reported_as_unknown(monkeypatch):
    monkeypatch.setattr(knowledge_base, "_grader", _FailingGrader())
    result = ToolResult(content="[1] doc: text", sources=[Source(title="doc", snippet="text")])

    assert knowledge_base._grade("q", result) == {"relevant": "unknown", "error": "bedrock down"}
