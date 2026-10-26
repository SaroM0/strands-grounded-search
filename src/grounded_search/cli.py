"""Command-line entry point."""
from __future__ import annotations

import logging
import sys

from strands import Agent

from .agent import build_agent
from .config import settings
from .hooks import TrajectoryRecorder
from .telemetry import flush_telemetry


def _ask(agent: Agent, recorder: TrajectoryRecorder, query: str) -> None:
    result = agent(query)
    usage = result.metrics.accumulated_usage

    print("\n\n" + "=" * 72)
    if result.structured_output is not None:
        print(result.structured_output.render())
    else:
        print(result)
    print("-" * 72)
    if recorder.current:
        print(recorder.current.render())
    print(f"Tokens (session total): in={usage['inputTokens']} out={usage['outputTokens']}")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Our own loggers follow LOG_LEVEL; third-party ones (strands, httpx, botocore) stay at WARNING
    # unless LOG_LEVEL=DEBUG.
    logging.getLogger("grounded_search").setLevel(settings.log_level)
    if settings.log_level == "DEBUG":
        logging.getLogger("strands").setLevel(logging.DEBUG)


def main() -> None:
    _setup_logging()
    args_query = " ".join(sys.argv[1:]).strip()
    recorder = TrajectoryRecorder()
    agent = build_agent(hooks=[recorder])

    try:
        if args_query:
            _ask(agent, recorder, args_query)
            return

        print("Type a research question. Empty input or 'exit' to quit.")
        while True:
            query = input("\n> ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break
            _ask(agent, recorder, query)
    finally:
        flush_telemetry()


if __name__ == "__main__":
    main()
