# strands-grounded-search

Experiment with [Strands Agents](https://strandsagents.com) to observe real autonomy: a single
orchestrator agent (Claude Sonnet 5.5) decides which tool to use, in what order, whether it needs
another query, and evaluates the gathered evidence itself before answering. There is no separate
"evidence evaluation" node: it's the agent's own responsibility, within its own reasoning.

## Structure

```
src/grounded_search/
  config.py              # Centralized settings (env vars)
  schemas.py              # Source / ToolResult / CachedResearch + GroundedAnswer (Pydantic, final output)
  telemetry.py            # OpenTelemetry exporters (console / Langfuse / any OTLP backend)
  agent.py                 # Builds the Agent (Strands) with Claude Sonnet 5.5
  cli.py                    # Command-line entry point
  evals/                    # Offline evaluation: dataset loader, metrics, LLM judge, runner
  hooks/
    live_log.py             # LiveLogger: logs each agent-loop step to the console as it happens
    trajectory.py           # TrajectoryRecorder: logs model calls, tool calls, order, reformulations
    cache_first.py          # CacheFirstPolicy: blocks external searches until the cache was checked
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

## Observing the agent

### Hooks (`hooks/`)

Strands fires typed events around every step of the agent loop (`BeforeInvocationEvent`,
`BeforeModelCallEvent`, `BeforeToolCallEvent`, `AfterToolCallEvent`, ...).

- **`LiveLogger`** (`LIVE_LOG=true`, default) logs every step in real time: each model turn (context
  size, stop reason, what the model said or thought, which tools it decided to call), each tool
  call (status, duration, cache hits/scores, sources) and the token totals. It replaces Strands'
  default token-streaming handler. Tools also log what they do (provider, Qdrant collection,
  caching). `LOG_LEVEL=DEBUG` adds Strands' internal logs.
- **`TrajectoryRecorder`** only observes. The CLI attaches it and prints, after each answer, the
  tools called in order with their query, status and duration, plus how many times each tool was
  reformulated.
- **`CacheFirstPolicy`** (`ENFORCE_CACHE_FIRST=true`, default) writes to `BeforeToolCallEvent.cancel_tool`:
  it cancels `search_web` / `research_with_sonar` until `search_research_cache` was called in the
  same invocation. The model receives the cancellation as a tool error, so the trajectory shows how
  it adapts to a rule the prompt only *suggests*.

### Structured output (`schemas.GroundedAnswer`)

The agent is built with `structured_output_model=GroundedAnswer`. Strands exposes the Pydantic
schema to the model as a tool, validates the call and re-prompts the model on validation errors.
`result.structured_output` is a typed object (answer, citations, uncertainties, limitations,
confidence) instead of free-form text. It shows up in the trajectory as a `GroundedAnswer` tool
call.

### Traces (`telemetry.py`)

Strands emits OpenTelemetry spans: `invoke_agent` > `execute_event_loop_cycle` > `chat` (model
call, with tokens) / `execute_tool <name>`. Exporters:

- `OTEL_TRACES_FILE` (default `traces.jsonl`): one JSON span per line, appended to a file. On by
  default and kept out of the console so the live logs stay readable.
- `OTEL_CONSOLE_EXPORT=true`: also print raw spans to stdout.

Langfuse and OTLP need the exporter: `pip install -e ".[otel]"`.

- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`): send them to Langfuse.
- `OTEL_EXPORTER_OTLP_ENDPOINT` (+ `OTEL_EXPORTER_OTLP_HEADERS`): any OTLP backend, e.g. Arize
  Phoenix at `http://localhost:6006/v1/traces`.

## Evaluating the agent

The hooks and traces show *what* the agent did; `evals/` measures *how well*. The runner sends
every case in `evals/dataset.jsonl` to a fresh agent (no shared history), records the trajectory
plus the sources the tools actually returned, and has an LLM judge grade the answer.

```bash
make eval                                                        # both variants, every case
python -m grounded_search.evals.runner --variants free --no-judge --ids einstein-nobel
```

Each case is one JSON line:

```json
{"id": "einstein-nobel", "question": "...", "reference": "...", "must_include": ["photoelectric effect"],
 "expected_tools": ["search_web"], "answerable": true, "tags": ["false_premise"]}
```

Only `id` and `question` are required. Use `answerable: false` for questions the tools can't
answer (the agent should say so) and `expected_tools` for e.g. questions about your own documents
(`search_knowledge_base`).

Variants: `cache_first` (`CacheFirstPolicy` on) and `free` (the agent decides on its own). The
default is whatever `ENFORCE_CACHE_FIRST` says; `make eval` compares both.

Metrics per variant (printed as a table, saved to `evals/results/<timestamp>/summary.json`; every
case with its answer, trajectory and returned sources goes to `results.jsonl`):

- **Quality (judge)**: `judge_score` (correct=1, partial=0.5, incorrect=0), `facts_covered`
  (share of `must_include` facts in the answer), `unsupported_claims` (claims not backed by the
  evidence the tools returned).
- **Citations (deterministic)**: `citation_precision`, the share of citations whose source is a
  URL or title a tool actually returned. Below 1.0 means hallucinated or mangled sources.
- **Behavior**: `expected_tools_rate`, `abstention_rate` (unanswerable cases), `cache_first_rate`,
  `tool_calls`, `external_calls`, `blocked_calls`, `reformulations`.
- **Cost**: `model_calls`, `input_tokens`, `output_tokens`, `latency_s`.

The judge uses `BEDROCK_MODEL_ID` unless you pass `--judge-model`; grading with the same model
the agent uses is cheaper but less independent. `--no-judge` keeps only the deterministic metrics.

Caveat: `search_web` / `research_with_sonar` write to the research cache, so later cases, variants
and runs can hit what earlier ones cached. That biases `external_calls` and cache metrics. For a
clean comparison, point `RESEARCH_CACHE_COLLECTION` at a scratch collection and delete it between
runs.

## Usage

```bash
pip install -e ".[dev,otel]"
cp .env.example .env   # fill in the API keys you need
docker compose up -d   # start local Qdrant
python -m grounded_search.cli "What evidence exists for and against sodium-ion batteries?"
```

## Tests

```bash
pytest
```

`tests/test_evals.py` checks the eval metrics against a scripted fake model; no API calls.
