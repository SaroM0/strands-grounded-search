"""Relevance grader for knowledge-base passages: the retrieval evaluator of a corrective RAG.

An LLM judges, passage by passage, whether each retrieved chunk is useful to answer the question.
The verdicts are aggregated into context precision without reference (the RAGAS metric
LLMContextPrecisionWithoutReference): the average precision of the ranking, so relevant passages
ranked first score higher than the same passages ranked last.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..config import settings

GRADER_PROMPT = """\
You grade the passages a retrieval system returned for a question. For each passage, decide \
whether it contains information that is useful to answer the question (fully or partially). A \
passage on the same topic that doesn't help answer this specific question is not relevant. Grade \
every passage independently, by its number. Be strict."""


class PassageVerdict(BaseModel):
    index: int = Field(description="Number of the passage, as given in the input (1-based).")
    relevant: bool = Field(description="Whether the passage is useful to answer the question.")


class RelevanceVerdicts(BaseModel):
    verdicts: list[PassageVerdict] = Field(description="One verdict per passage, in order.")


def context_precision(relevant: Sequence[bool]) -> float:
    """Average precision over the ranked passages (0.0 if none is relevant)."""
    hits, total = 0, 0.0
    for rank, is_relevant in enumerate(relevant, 1):
        if is_relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


@dataclass
class RelevanceGrade:
    score: float
    threshold: float
    relevant: list[bool]

    @property
    def is_relevant(self) -> bool:
        return self.score > self.threshold

    def to_dict(self) -> dict:
        return {
            "relevant": "yes" if self.is_relevant else "no",
            "score": round(self.score, 3),
            "threshold": self.threshold,
            "relevant_passages": [i for i, is_relevant in enumerate(self.relevant, 1) if is_relevant],
        }


def _render(question: str, passages: Sequence[str]) -> str:
    rendered = "\n\n".join(f"PASSAGE {i}:\n{passage}" for i, passage in enumerate(passages, 1))
    return f"QUESTION:\n{question}\n\n{rendered}"


class RelevanceGrader:
    def __init__(self, model: Model | None = None, threshold: float | None = None) -> None:
        if model is None:
            # Imported here: agent.py imports the tools, which import this module.
            from ..agent import build_bedrock_model

            model = build_bedrock_model(settings.kb_relevance_model_id)
        self._model = model
        self._threshold = settings.kb_relevance_threshold if threshold is None else threshold

    def grade(self, question: str, passages: Sequence[str]) -> RelevanceGrade:
        if not passages:
            return RelevanceGrade(score=0.0, threshold=self._threshold, relevant=[])

        # A fresh agent per call, so no verdict leaks into the next one through the history.
        agent = Agent(
            model=self._model,
            system_prompt=GRADER_PROMPT,
            structured_output_model=RelevanceVerdicts,
            callback_handler=None,
            trace_attributes={"service.component": "grounded-search-relevance-grader"},
        )
        verdicts = agent(_render(question, passages)).structured_output.verdicts
        by_index = {verdict.index: verdict.relevant for verdict in verdicts}
        relevant = [by_index.get(i, False) for i in range(1, len(passages) + 1)]
        return RelevanceGrade(score=context_precision(relevant), threshold=self._threshold, relevant=relevant)
