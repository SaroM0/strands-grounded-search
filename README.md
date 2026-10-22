# strands-grounded-search

Experiment with [Strands Agents](https://strandsagents.com) to observe real autonomy: a single
orchestrator agent (Claude Sonnet 5.5) decides which tool to use, in what order, whether it needs
another query, and evaluates the gathered evidence itself before answering. There is no separate
"evidence evaluation" node: it's the agent's own responsibility, within its own reasoning.

## Structure

```
src/grounded_search/
  config.py              # Centralized settings (env vars)
  schemas.py              # Source / ToolResult / CachedResearch shared across tools
  agent.py                 # Builds the Agent (Strands) with Claude Sonnet 5.5
  cli.py                    # Command-line entry point
  prompts/
    system_prompt.py        # Role, tool-choice heuristic, evidence evaluation, response format
  clients/
    embedding.py            # Shared Embedder (fastembed by default), used by both Qdrant clients
    perplexity_client.py    # Perplexity Sonar API
    web_search_client.py    # Tavily / Brave (interchangeable)
    knowledge_base_client.py  # Qdrant: search over documents the user uploaded
    research_cache_client.py # Qdrant: upsert/search a semantic cache of past tool results
  tools/
    sonar_research.py       # @tool research_with_sonar (writes to the research cache)
    web_search.py            # @tool search_web (writes to the research cache)
    knowledge_base.py        # @tool search_knowledge_base
    research_cache.py        # @tool search_research_cache
tests/                       # Unit tests per client, with HTTP/Qdrant mocked
```

## The four tools

- **`research_with_sonar(query)`** — Perplexity Sonar. Multi-source research with reasoned
  synthesis. For open-ended or controversial questions.
- **`search_web(query)`** — Tavily or Brave. Raw results (URL, title, snippet), with no synthesis
  from another model. For specific or verifiable facts.
- **`search_knowledge_base(query)`** — Qdrant + embeddings. Semantic retrieval over documents
  uploaded by the user.
- **`search_research_cache(query)`** — Qdrant + embeddings. Semantic retrieval over past
  `search_web` / `research_with_sonar` results.

The agent decides which one(s) to use, how many times, and when to stop — see
`prompts/system_prompt.py`.

## Research cache

Every result from `search_web` and `research_with_sonar` is embedded on its originating query
and upserted into a dedicated Qdrant collection (`RESEARCH_CACHE_COLLECTION`, separate from the
user's own knowledge base collection). `search_research_cache` looks up that collection
semantically, so a question close to one already researched can be answered from the cache
instead of hitting an external API again.

The system prompt tells the agent to check the cache first for likely-already-researched
questions, but to corroborate cache hits with a fresh search when freshness or confidence
matters — the cache is a shortcut, not a replacement for ground truth. `RESEARCH_CACHE_MIN_SCORE`
controls how semantically close a cached entry must be to count as a hit.

### Running Qdrant locally

```bash
docker compose up -d
```

This starts Qdrant on `localhost:6333` (HTTP/REST) and `localhost:6334` (gRPC), persisting data
to `./qdrant_storage/`. `QDRANT_URL` in `.env` already points at `http://localhost:6333`.

## Usage

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in the API keys you need
docker compose up -d   # start local Qdrant
python -m grounded_search.cli "What evidence exists for and against sodium-ion batteries?"
```

## Tests

```bash
pytest
```
