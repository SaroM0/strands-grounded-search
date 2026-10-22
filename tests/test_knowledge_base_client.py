from grounded_search.clients.knowledge_base_client import KnowledgeBaseClient
from grounded_search.schemas import ToolResult


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _FakeHit:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


class _FakeQueryResponse:
    def __init__(self, points: list[_FakeHit]) -> None:
        self.points = points


class _FakeQdrantClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def query_points(self, **kwargs: object) -> _FakeQueryResponse:
        return _FakeQueryResponse(
            [
                _FakeHit({"title": "Paper X", "url": "https://example.com/paper-x", "text": "Limitation: small sample size."}),
            ]
        )


def test_knowledge_base_search_builds_tool_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_search.clients.knowledge_base_client._QdrantClient",
        _FakeQdrantClient,
    )

    client = KnowledgeBaseClient(embedder=_FakeEmbedder())
    result = client.search("what are the limitations?", max_results=5)

    assert isinstance(result, ToolResult)
    assert len(result.sources) == 1
    assert result.sources[0].title == "Paper X"
