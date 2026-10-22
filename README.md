# strands-grounded-search

Experiment with [Strands Agents](https://strandsagents.com) to observe real autonomy: a single
orchestrator agent (Claude Sonnet 5.5) decides which tool to use, in what order, whether it needs
another query, and evaluates the gathered evidence itself before answering. There is no separate
"evidence evaluation" node: it's the agent's own responsibility, within its own reasoning.

## Structure

```
src/grounded_search/
  config.py              # Centralized settings (env vars)
  schemas.py              # Source / ToolResult shared across tools
  agent.py                 # Builds the Agent (Strands) with Claude Sonnet 5.5
  cli.py                    # Command-line entry point
  prompts/
    system_prompt.py        # Role, tool-choice heuristic, evidence evaluation, response format
  clients/
    perplexity_client.py    # Perplexity Sonar API
    web_search_client.py    # Tavily / Brave (interchangeable)
    knowledge_base_client.py  # Qdrant + interchangeable embedder (fastembed by default)
  tools/
    sonar_research.py       # @tool research_with_sonar
    web_search.py            # @tool search_web
    knowledge_base.py        # @tool search_knowledge_base
tests/                       # Unit tests per client, with HTTP/Qdrant mocked
```

## The three tools

- **`research_with_sonar(query)`** — Perplexity Sonar. Multi-source research with reasoned
  synthesis. For open-ended or controversial questions.
- **`search_web(query)`** — Tavily or Brave. Raw results (URL, title, snippet), with no synthesis
  from another model. For specific or verifiable facts.
- **`search_knowledge_base(query)`** — Qdrant + embeddings. Semantic retrieval over documents
  uploaded by the user.

The agent decides which one(s) to use, how many times, and when to stop — see
`prompts/system_prompt.py`.

## Usage

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in the API keys you need
python -m grounded_search.cli "What evidence exists for and against sodium-ion batteries?"
```

## Tests

```bash
pytest
```
