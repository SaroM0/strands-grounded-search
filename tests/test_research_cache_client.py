from grounded_search.clients.research_cache_client import ResearchCacheClient
from grounded_search.schemas import CachedResearch, ToolResult


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _FakeHit:
    def __init__(self, payload: dict, score: float) -> None:
        self.payload = payload
        self.score = score


class _FakeQueryResponse:
    def __init__(self, points: list[_FakeHit]) -> None:
        self.points = points


class _FakeQdrantClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.collection_created = False
        self.upserted_points: list[object] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_created

    def create_collection(self, **kwargs: object) -> None:
        self.collection_created = True

    def upsert(self, *, collection_name: str, points: list[object]) -> None:
        self.upserted_points.extend(points)

    def query_points(self, **kwargs: object) -> _FakeQueryResponse:
        return _FakeQueryResponse(
            [
                _FakeHit(
                    {
                        "query": "latest version of Python?",
                        "tool": "search_web",
                        "content": "- Python 3.15.1: latest stable release.",
                        "sources": [
                            {"title": "python.org", "url": "https://python.org", "snippet": "Latest stable release."}
                        ],
                        "cached_at": 1_700_000_000.0,
                    },
                    score=0.92,
                ),
            ]
        )


def test_upsert_creates_collection_and_stores_point(monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_search.clients.research_cache_client._QdrantClient",
        _FakeQdrantClient,
    )

    client = ResearchCacheClient(embedder=_FakeEmbedder())
    client.upsert("latest version of Python?", tool="search_web", result=ToolResult(content="x"))

    assert client._client.collection_created is True
    assert len(client._client.upserted_points) == 1


def test_search_returns_cached_research_above_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_search.clients.research_cache_client._QdrantClient",
        _FakeQdrantClient,
    )

    client = ResearchCacheClient(embedder=_FakeEmbedder())
    client._client.collection_created = True

    hits = client.search("what's the newest Python release?", max_results=3, min_score=0.5)

    assert len(hits) == 1
    assert isinstance(hits[0], CachedResearch)
    assert hits[0].tool == "search_web"
    assert hits[0].score == 0.92
    assert hits[0].result.sources[0].url == "https://python.org"


def test_search_returns_empty_when_collection_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_search.clients.research_cache_client._QdrantClient",
        _FakeQdrantClient,
    )

    client = ResearchCacheClient(embedder=_FakeEmbedder())

    assert client.search("anything", max_results=3, min_score=0.5) == []
