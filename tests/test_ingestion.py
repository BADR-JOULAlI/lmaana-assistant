import json
import socket

import pytest

from lmaana_assistant.contracts import Source
from lmaana_assistant.errors import DependencyUnavailable
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.ingestion.sources import extract_sections, load_chunks, validate_url
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder
from lmaana_assistant.retrieval.store import VectorStore


def test_ids_are_stable_and_html_locations_preserved(source_manifest):
    _, first = load_chunks(source_manifest, LexicalEmbedder())
    _, second = load_chunks(source_manifest, LexicalEmbedder())
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].location == "Inscription du projet"
    assert "pied de page" not in " ".join(chunk.text for chunk in first)


def test_failed_ingestion_preserves_active_release(settings, source_manifest, indexed):
    source_manifest.write_text('{"sources": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        ingest(source_manifest, settings, LexicalEmbedder())
    assert json.loads(settings.active_path.read_text())["release"] == indexed["release"]


def test_two_owners_cannot_open_local_storage(settings):
    store = VectorStore(settings)
    try:
        with pytest.raises(DependencyUnavailable, match="locked"):
            VectorStore(settings)
    finally:
        store.close()


def test_source_must_be_reviewed(source_manifest):
    manifest = json.loads(source_manifest.read_text())
    manifest["sources"][0]["reviewed"] = False
    source_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Review"):
        load_chunks(source_manifest, LexicalEmbedder())


def test_local_source_cannot_escape_manifest_directory(source_manifest):
    manifest = json.loads(source_manifest.read_text())
    manifest["sources"][0]["local_path"] = "../outside.txt"
    source_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        load_chunks(source_manifest, LexicalEmbedder())


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/data",
        "https://other.example/data",
        "https://user:password@example.org/data",
        "https://example.org:8080/data",
    ],
)
def test_fetch_url_must_use_approved_https_host(url):
    with pytest.raises(ValueError, match="HTTPS"):
        validate_url(url, ["example.org"])


def test_private_dns_resolution_rejected(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="public"):
        validate_url("https://example.org/data", ["example.org"])


def test_empty_extraction_rejected():
    source = Source(
        id="empty", title="Empty", publisher="Test", kind="html", url="https://example.org/"
    )
    with pytest.raises(ValueError, match="review"):
        extract_sections(source, b"<html><main></main></html>")


def test_chunk_overlap_preserves_original_characters():
    text = "one  two\nthree four five six"
    assert LexicalEmbedder().split(text, 4, 1) == ["one  two\nthree four", "four five six"]
