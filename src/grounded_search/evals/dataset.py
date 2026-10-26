"""Eval dataset: one JSON object per line (see evals/dataset.jsonl)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    # Reference answer and key facts the answer must contain; the judge grades against them.
    reference: str | None = None
    must_include: list[str] = field(default_factory=list)
    # Tools the agent is expected to use at least once (e.g. search_knowledge_base for KB questions).
    expected_tools: list[str] = field(default_factory=list)
    # False for questions the tools can't answer: the agent should say so instead of guessing.
    answerable: bool = True
    tags: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                cases.append(EvalCase(**json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid eval case ({exc})") from exc
    ids = [case.id for case in cases]
    duplicates = {case_id for case_id in ids if ids.count(case_id) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids: {sorted(duplicates)}")
    return cases
