"""Command-line entry point."""
from __future__ import annotations

import sys

from .agent import build_agent


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Research question: ").strip()

    if not query:
        print("No question was provided.")
        return

    agent = build_agent()
    agent(query)


if __name__ == "__main__":
    main()
