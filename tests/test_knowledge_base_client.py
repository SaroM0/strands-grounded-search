from grounded_search.clients.knowledge_base_client import KnowledgeBaseClient
from grounded_search.schemas import ToolResult


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _FakeHit:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


class _FakeQdrantClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def search(self, **kwargs: object) -> list[_FakeHit]:
        return [
            _FakeHit({"title": "Paper X", "url": "https://example.com/paper-x", "text": "Limitación: muestra pequeña."}),
        ]


def test_knowledge_base_search_builds_tool_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_search.clients.knowledge_base_client._QdrantClient",
        _FakeQdrantClient,
    )

    client = KnowledgeBaseClient(embedder=_FakeEmbedder())
    result = client.search("¿cuáles son las limitaciones?", max_results=5)

    assert isinstance(result, ToolResult)
    assert len(result.sources) == 1
    assert result.sources[0].title == "Paper X"
