import httpx
import pytest

from lmaana_assistant.ingestion import sources


def test_every_redirect_is_revalidated(monkeypatch):
    validated = []

    def validate(url, allowed):
        validated.append(url)
        if "127.0.0.1" in url:
            raise ValueError("private target")

    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "https://127.0.0.1/secret"})
    )
    original_client = httpx.Client
    monkeypatch.setattr(sources, "validate_url", validate)
    monkeypatch.setattr(
        sources.httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs)
    )
    with pytest.raises(ValueError, match="private"):
        sources.fetch_bytes("https://example.org/document", ["example.org"])
    assert validated == ["https://example.org/document", "https://127.0.0.1/secret"]


def test_large_download_is_rejected(monkeypatch):
    original_client = httpx.Client
    monkeypatch.setattr(sources, "MAX_BYTES", 4)
    monkeypatch.setattr(sources, "validate_url", lambda *args: None)
    monkeypatch.setattr(
        sources.httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"12345")),
            **kwargs,
        ),
    )
    with pytest.raises(ValueError, match="limit"):
        sources.fetch_bytes("https://example.org/document", ["example.org"])
