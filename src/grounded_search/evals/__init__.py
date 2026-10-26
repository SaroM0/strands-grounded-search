"""Offline evaluation of the agent: run a dataset of questions, score each answer and trajectory."""
from .collector import SourceCollector
from .dataset import EvalCase, load_dataset
from .metrics import CaseMetrics, compute_case_metrics, summarize

__all__ = ["CaseMetrics", "EvalCase", "SourceCollector", "compute_case_metrics", "load_dataset", "summarize"]
