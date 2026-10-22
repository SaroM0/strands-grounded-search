# strands-grounded-search

Experimento con [Strands Agents](https://strandsagents.com) para observar autonomía real: un único
agente orquestador (Claude Sonnet 5.5) decide qué herramienta usar, en qué orden, si necesita otra
consulta, y evalúa la evidencia reunida él mismo antes de responder. No hay un nodo separado de
"evaluación de evidencia": es responsabilidad del agente, dentro de su propio razonamiento.

## Estructura

```
src/grounded_search/
  config.py              # Settings centralizados (env vars)
  schemas.py              # Source / ToolResult compartidos entre herramientas
  agent.py                 # Construcción del Agent (Strands) con Claude Sonnet 5.5
  cli.py                    # Punto de entrada por línea de comandos
  prompts/
    system_prompt.py        # Rol, heurística de elección de herramienta, evaluación de evidencia, formato de respuesta
  clients/
    perplexity_client.py    # API de Perplexity Sonar
    web_search_client.py    # Tavily / Brave (intercambiables)
    knowledge_base_client.py  # Qdrant + embedder intercambiable (fastembed por defecto)
  tools/
    sonar_research.py       # @tool research_with_sonar
    web_search.py            # @tool search_web
    knowledge_base.py        # @tool search_knowledge_base
tests/                       # Tests unitarios por cliente, con HTTP/Qdrant mockeados
```

## Las tres herramientas

- **`research_with_sonar(query)`** — Perplexity Sonar. Investigación multi-fuente con síntesis razonada.
  Para preguntas abiertas o controvertidas.
- **`search_web(query)`** — Tavily o Brave. Resultados crudos (URL, título, fragmento), sin síntesis
  de otro modelo. Para hechos puntuales o verificables.
- **`search_knowledge_base(query)`** — Qdrant + embeddings. Recuperación semántica sobre documentos
  cargados por el usuario.

El agente decide cuál(es) usar, cuántas veces, y cuándo detenerse — ver `prompts/system_prompt.py`.

## Uso

```bash
pip install -e ".[dev]"
cp .env.example .env   # completar las API keys necesarias
python -m grounded_search.cli "¿Qué evidencia hay a favor y en contra de las baterías de sodio?"
```

## Tests

```bash
pytest
```
