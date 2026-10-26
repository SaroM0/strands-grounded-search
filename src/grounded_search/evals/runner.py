"""Runs the eval dataset through the real agent and writes per-case results plus a summary.

    python -m grounded_search.evals.runner --variants cache_first,free
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..agent import build_agent
from ..config import settings
from ..hooks import TrajectoryRecorder
from ..telemetry import flush_telemetry
from .collector import SourceCollector
from .dataset import EvalCase, load_dataset
from .judge import Judge
from .metrics import CaseMetrics, compute_case_metrics, render_summary, summarize

logger = logging.getLogger("grounded_search.evals")

# Variant name -> build_agent overrides.
VARIANTS = {
    "cache_first": {"enforce_cache_first": True},
    "free": {"enforce_cache_first": False},
}


def run_case(case: EvalCase, variant: str, judge: Judge | None, live_log: bool) -> tuple[CaseMetrics, dict]:
    recorder, collector = TrajectoryRecorder(), SourceCollector()
    # A fresh agent per case: the conversation history of one case must not help the next one.
    # No human in the loop here: approvals / clarifying questions and session persistence are off.
    agent = build_agent(
        hooks=[recorder, collector],
        live_log=live_log,
        quiet=True,
        hitl_approval=False,
        hitl_clarify=False,
        session_id="",
        **VARIANTS[variant],
    )
    answer, usage, error = None, None, None
    started = time.monotonic()
    try:
        result = agent(case.question)
        answer = result.structured_output
        usage = result.metrics.accumulated_usage
    except Exception as exc:  # one failing case must not abort the whole run
        logger.exception("case %s (%s) failed", case.id, variant)
        error = f"{type(exc).__name__}: {exc}"
    latency = time.monotonic() - started

    metrics = compute_case_metrics(case, variant, recorder.current, collector, answer, usage, latency, error)
    if judge is not None and answer is not None:
        try:
            verdict = judge.grade(case, answer, collector.evidence)
            metrics.judge_verdict = verdict.verdict
            metrics.judge_score = verdict.score
            metrics.facts_covered = verdict.facts_covered
            metrics.unsupported_claims = verdict.unsupported_claims
            metrics.judge_notes = verdict.notes
        except Exception:
            logger.exception("judge failed on case %s (%s)", case.id, variant)

    record = {
        "metrics": metrics.to_dict(),
        "question": case.question,
        "answer": answer.model_dump() if answer is not None else None,
        "trajectory": [asdict(step) for step in recorder.current.steps] if recorder.current else [],
        "returned_sources": [asdict(source) for source in collector.sources],
    }
    return metrics, record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_variant = "cache_first" if settings.enforce_cache_first else "free"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="evals/dataset.jsonl", help="JSONL file with the eval cases")
    parser.add_argument(
        "--variants",
        default=default_variant,
        help=f"Comma-separated agent variants to compare: {', '.join(VARIANTS)} (default: {default_variant}, "
        "from ENFORCE_CACHE_FIRST)",
    )
    parser.add_argument("--ids", help="Comma-separated case ids to run (default: all)")
    parser.add_argument("--tags", help="Comma-separated tags; runs cases that have any of them")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (deterministic metrics only)")
    parser.add_argument("--judge-model", help="Bedrock model id for the judge (default: BEDROCK_MODEL_ID)")
    parser.add_argument("--out", help="Output directory (default: evals/results/<timestamp>)")
    parser.add_argument("--live-log", action="store_true", help="Log every agent-loop step, like the CLI")
    args = parser.parse_args(argv)

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(sorted(unknown))}")
    return args


def _select(cases: list[EvalCase], ids: str | None, tags: str | None) -> list[EvalCase]:
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        cases = [c for c in cases if c.id in wanted]
    if tags:
        wanted = {t.strip() for t in tags.split(",")}
        cases = [c for c in cases if wanted & set(c.tags)]
    return cases


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("grounded_search").setLevel(settings.log_level)
    args = _parse_args(argv)

    cases = _select(load_dataset(args.dataset), args.ids, args.tags)
    if not cases:
        raise SystemExit("No eval cases selected.")
    out_dir = Path(args.out or f"evals/results/{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = None if args.no_judge else Judge(args.judge_model)

    logger.warning(
        "Running %d case(s) x %d variant(s). search_web / research_with_sonar write to the research "
        "cache (%s), so later runs and variants can hit results cached by earlier ones.",
        len(cases),
        len(args.variants),
        settings.research_cache_collection,
    )
    results: list[CaseMetrics] = []
    try:
        with open(out_dir / "results.jsonl", "w", encoding="utf-8") as handle:
            for variant in args.variants:
                for index, case in enumerate(cases, start=1):
                    print(f"[{variant} {index}/{len(cases)}] {case.id}", flush=True)
                    metrics, record = run_case(case, variant, judge, args.live_log)
                    results.append(metrics)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
    finally:
        flush_telemetry()

    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + render_summary(summary))
    print(f"\nPer-case results: {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
