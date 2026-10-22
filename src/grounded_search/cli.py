"""Command-line entry point."""
from __future__ import annotations

import sys

from .agent import build_agent


def main() -> None:
    args_query = " ".join(sys.argv[1:]).strip()
    agent = build_agent()

    if args_query:
        agent(args_query)
        return

    print("Type a research question. Empty input or 'exit' to quit.")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        agent(query)


if __name__ == "__main__":
    main()
