"""Punto de entrada de línea de comandos."""
from __future__ import annotations

import sys

from .agent import build_agent


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Pregunta de investigación: ").strip()

    if not query:
        print("No se proporcionó ninguna pregunta.")
        return

    agent = build_agent()
    agent(query)


if __name__ == "__main__":
    main()
