from grounded_search.clients.web_search_client import TavilyClient
from grounded_search.schemas import ToolResult


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "results": [
                {"title": "Python 3.13", "url": "https://python.org", "content": "Última versión estable."},
            ]
        }


class _FakeHTTPClient:
    def __enter__(self) -> "_FakeHTTPClient":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()


def test_tavily_search_builds_tool_result(monkeypatch) -> None:
    client = TavilyClient(api_key="fake-key")

    monkeypatch.setattr(
        "grounded_search.clients.web_search_client.httpx.Client",
        lambda **kwargs: _FakeHTTPClient(),
    )

    result = client.search("¿última versión de Python?", max_results=5)

    assert isinstance(result, ToolResult)
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://python.org"
