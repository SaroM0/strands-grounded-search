"""LLM-as-judge: grades correctness against the reference and groundedness against the evidence."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from strands import Agent

from ..agent import build_bedrock_model
from ..schemas import GroundedAnswer
from .collector import Evidence
from .dataset import EvalCase

_MAX_EVIDENCE_CHARS = 40000

JUDGE_PROMPT = """\
You grade the answers of a research agent. You get the question, optionally a reference answer \
and key facts, the agent's structured answer, and the evidence its tools actually returned.

Grade these independently:
1. Correctness against the reference answer and key facts. "correct" if it states every key fact \
and contradicts nothing in the reference; "partially_correct" if it misses some key facts or adds \
minor errors; "incorrect" if it misses the main point or contradicts the reference. Wording and \
language don't matter, meaning does. If the question is marked unanswerable, "correct" means the \
agent clearly said it couldn't verify the answer instead of making one up. If there is no \
reference, grade correctness by whether the evidence supports the answer.
2. Key facts: for each key fact, say whether the answer states it.
3. Groundedness: list every factual claim in the answer that is NOT supported by the evidence \
(paraphrases and reasonable inferences are supported). An empty list means fully grounded.

Be strict and concise."""


class FactCheck(BaseModel):
    fact: str = Field(description="One of the key facts, copied as given.")
    present: bool = Field(description="Whether the answer states this fact (in any wording).")


class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    facts: list[FactCheck] = Field(description="One entry per key fact, in order. Empty only if there are none.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims in the answer not backed by the evidence."
    )
    notes: str = Field(description="One or two sentences justifying the verdict.")

    @property
    def score(self) -> float:
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[self.verdict]

    @property
    def facts_covered(self) -> float | None:
        return sum(f.present for f in self.facts) / len(self.facts) if self.facts else None


def _render_evidence(evidence: list[Evidence]) -> str:
    chunks, total = [], 0
    for item in evidence:
        chunk = f"[{item.tool}] query={item.query!r}\n{item.content}"
        if total + len(chunk) > _MAX_EVIDENCE_CHARS:
            chunks.append("... (evidence truncated)")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks) or "(the tools returned no evidence)"


def _render_case(case: EvalCase, answer: GroundedAnswer, evidence: list[Evidence]) -> str:
    facts = "\n".join(f"- {fact}" for fact in case.must_include) or "(none)"
    return (
        f"QUESTION:\n{case.question}\n\n"
        f"ANSWERABLE WITH THE AVAILABLE TOOLS: {'yes' if case.answerable else 'no'}\n\n"
        f"REFERENCE ANSWER:\n{case.reference or '(none)'}\n\n"
        f"KEY FACTS:\n{facts}\n\n"
        f"AGENT ANSWER:\n{answer.render()}\n\n"
        f"EVIDENCE RETURNED BY THE TOOLS:\n{_render_evidence(evidence)}"
    )


class Judge:
    def __init__(self, model_id: str | None = None) -> None:
        self._model = build_bedrock_model(model_id)

    def grade(self, case: EvalCase, answer: GroundedAnswer, evidence: list[Evidence]) -> JudgeVerdict:
        # A fresh agent per case, so no grading leaks into the next one through the history.
        agent = Agent(
            model=self._model,
            system_prompt=JUDGE_PROMPT,
            structured_output_model=JudgeVerdict,
            callback_handler=None,
            trace_attributes={"service.component": "grounded-search-eval-judge"},
        )
        result = agent(_render_case(case, answer, evidence))
        return result.structured_output
