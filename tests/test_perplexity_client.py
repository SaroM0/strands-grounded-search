from grounded_search.clients.perplexity_client import PerplexityClient
from grounded_search.schemas import ToolResult


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "Síntesis de prueba."}}],
            "citations": ["https://example.com/a", "https://example.com/b"],
        }


class _FakeHTTPClient:
    def __enter__(self) -> "_FakeHTTPClient":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()


def test_research_builds_tool_result_from_api_response(monkeypatch) -> None:
    client = PerplexityClient(api_key="fake-key")

    monkeypatch.setattr(
        "grounded_search.clients.perplexity_client.httpx.Client",
        lambda **kwargs: _FakeHTTPClient(),
    )

    result = client.research("¿pregunta de prueba?")

    assert isinstance(result, ToolResult)
    assert result.content == "Síntesis de prueba."
    assert [source.url for source in result.sources] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_research_requires_api_key() -> None:
    client = PerplexityClient(api_key=None)
    try:
        client.research("x")
    except RuntimeError as exc:
        assert "PERPLEXITY_API_KEY" in str(exc)
    else:
        raise AssertionError("se esperaba RuntimeError sin API key")
