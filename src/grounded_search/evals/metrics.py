"""Per-case metrics (deterministic + judge) and their aggregation per variant."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

from ..hooks.cache_first import CACHE_TOOL, EXTERNAL_TOOLS
from ..hooks.trajectory import Trajectory
from ..schemas import GroundedAnswer
from .collector import SourceCollector, is_grounded
from .dataset import EvalCase

FINAL_ANSWER_TOOL = GroundedAnswer.__name__


@dataclass
class CaseMetrics:
    case_id: str
    variant: str
    tags: list[str]
    error: str | None = None
    latency_s: float = 0.0
    # Trajectory
    model_calls: int = 0
    tool_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    blocked_calls: int = 0
    external_calls: int = 0
    reformulations: int = 0
    cache_checked_first: bool = False
    expected_tools_used: bool | None = None
    # Cost
    input_tokens: int = 0
    output_tokens: int = 0
    # Answer
    has_answer: bool = False
    confidence: str | None = None
    citations: int = 0
    grounded_citations: int = 0
    abstained: bool | None = None
    # Judge (filled by judge.py; None when --no-judge)
    judge_verdict: str | None = None
    judge_score: float | None = None
    facts_covered: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    judge_notes: str | None = None

    @property
    def citation_precision(self) -> float | None:
        return self.grounded_citations / self.citations if self.citations else None

    def to_dict(self) -> dict:
        return {**asdict(self), "citation_precision": self.citation_precision}


def compute_case_metrics(
    case: EvalCase,
    variant: str,
    trajectory: Trajectory | None,
    collector: SourceCollector,
    answer: GroundedAnswer | None,
    usage: dict | None,
    latency_s: float,
    error: str | None = None,
) -> CaseMetrics:
    metrics = CaseMetrics(case_id=case.id, variant=variant, tags=case.tags, error=error, latency_s=latency_s)

    if trajectory is not None:
        steps = [step for step in trajectory.steps if step.tool != FINAL_ANSWER_TOOL]
        metrics.model_calls = trajectory.model_calls
        metrics.tool_calls = len(steps)
        metrics.tool_counts = {name: n for name, n in trajectory.tool_counts().items() if name != FINAL_ANSWER_TOOL}
        metrics.blocked_calls = sum(1 for step in steps if step.blocked_reason)
        metrics.external_calls = sum(
            1 for step in steps if step.tool in EXTERNAL_TOOLS and step.status == "success" and not step.blocked_reason
        )
        metrics.reformulations = sum(len(queries) - 1 for queries in trajectory.reformulations().values())
        metrics.cache_checked_first = bool(steps) and steps[0].tool == CACHE_TOOL
        if case.expected_tools:
            used = {step.tool for step in steps if step.status == "success"}
            metrics.expected_tools_used = set(case.expected_tools) <= used

    if usage:
        metrics.input_tokens = usage.get("inputTokens", 0)
        metrics.output_tokens = usage.get("outputTokens", 0)

    if answer is not None:
        identifiers = collector.identifiers()
        metrics.has_answer = True
        metrics.confidence = answer.confidence
        metrics.citations = len(answer.citations)
        metrics.grounded_citations = sum(1 for c in answer.citations if is_grounded(c.source, identifiers))
        if not case.answerable:
            metrics.abstained = answer.confidence == "low" or bool(answer.limitations)

    return metrics


def _mean(values: list[float | int | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def summarize(results: list[CaseMetrics]) -> dict[str, dict[str, float | int | None]]:
    """Aggregates per variant. Rates are means over the cases where the metric applies."""
    summary: dict[str, dict[str, float | int | None]] = {}
    for variant in dict.fromkeys(r.variant for r in results):
        rows = [r for r in results if r.variant == variant]
        ok = [r for r in rows if r.error is None]
        summary[variant] = {
            "cases": len(rows),
            "errors": len(rows) - len(ok),
            "answer_rate": _mean([r.has_answer for r in rows]),
            "judge_score": _mean([r.judge_score for r in ok]),
            "facts_covered": _mean([r.facts_covered for r in ok]),
            "citation_precision": _mean([r.citation_precision for r in ok]),
            "unsupported_claims": _mean([len(r.unsupported_claims) if r.judge_verdict else None for r in ok]),
            "expected_tools_rate": _mean([r.expected_tools_used for r in ok]),
            "abstention_rate": _mean([r.abstained for r in ok]),
            "cache_first_rate": _mean([r.cache_checked_first for r in ok]),
            "tool_calls": _mean([r.tool_calls for r in ok]),
            "external_calls": _mean([r.external_calls for r in ok]),
            "blocked_calls": _mean([r.blocked_calls for r in ok]),
            "reformulations": _mean([r.reformulations for r in ok]),
            "model_calls": _mean([r.model_calls for r in ok]),
            "input_tokens": _mean([r.input_tokens for r in ok]),
            "output_tokens": _mean([r.output_tokens for r in ok]),
            "latency_s": _mean([r.latency_s for r in ok]),
        }
    return summary


def render_summary(summary: dict[str, dict[str, float | int | None]]) -> str:
    variants = list(summary)
    if not variants:
        return "No results."
    metric_names = list(summary[variants[0]])
    width = max(len(name) for name in metric_names) + 2
    column = max(12, *(len(v) + 2 for v in variants))
    lines = ["".ljust(width) + "".join(v.rjust(column) for v in variants)]
    for name in metric_names:
        cells = []
        for variant in variants:
            value = summary[variant][name]
            if value is None:
                cells.append("-".rjust(column))
            elif isinstance(value, int):
                cells.append(str(value).rjust(column))
            else:
                cells.append(f"{value:.2f}".rjust(column))
        lines.append(name.ljust(width) + "".join(cells))
    return "\n".join(lines)
