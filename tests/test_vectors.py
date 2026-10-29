import math

import pytest
from fastapi.testclient import TestClient

from lmaana_assistant.api.app import create_app
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.ingestion.sources import load_chunks
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder
from lmaana_assistant.retrieval.store import VectorStore


@pytest.mark.parametrize("problem", ["nan", "infinity", "zero", "dimension", "count", "type"])
def test_invalid_embeddings_cannot_create_a_collection_or_change_activation(
    settings, source_manifest, indexed, problem
):
    class BrokenEmbedder(LexicalEmbedder):
        def encode_documents(self, texts):
            vectors = super().encode_documents(texts)
            if problem == "count":
                return vectors[:-1]
            if problem == "dimension":
                vectors[-1] = [1.0]
            elif problem == "zero":
                vectors[-1] = [0.0] * self.dimension
            else:
                vectors[-1][0] = {"nan": float("nan"), "infinity": float("inf"), "type": "invalid"}[
                    problem
                ]
            return vectors

    previous = settings.active_path.read_bytes()
    with pytest.raises(ValueError, match="Embedding"):
        ingest(source_manifest, settings, BrokenEmbedder())
    assert settings.active_path.read_bytes() == previous
    store = VectorStore(settings)
    try:
        assert {item.name for item in store.client.get_collections().collections} == {
            indexed["release"]
        }
    finally:
        store.close()


def test_zero_query_has_no_results_without_asking_qdrant_for_a_ranking(
    settings, indexed, monkeypatch
):
    store = VectorStore(settings)
    try:
        monkeypatch.setattr(
            store.client, "query_points", lambda *a, **kw: pytest.fail("Zero query was ranked")
        )
        assert store.search(indexed["release"], [0.0] * 4096, 5) == []
    finally:
        store.close()


@pytest.mark.parametrize("problem", ["nan", "dimension", "boolean"])
def test_bad_query_is_a_503_and_releases_the_inference_slot(settings, indexed, problem):
    class FlakyEmbedder(LexicalEmbedder):
        failed = False

        def encode_query(self, text):
            if self.failed:
                return super().encode_query(text)
            self.failed = True
            if problem == "dimension":
                return [1.0]
            vector = [0.0] * self.dimension
            vector[0] = float("nan") if problem == "nan" else True
            return vector

    with TestClient(create_app(settings, embedder=FlakyEmbedder())) as client:
        response = client.post("/v1/answers", json={"question": "inscrire projet documentation"})
        assert response.status_code == 503
        assert "invalid query vector" in response.json()["detail"]
        recovered = client.post("/v1/answers", json={"question": "inscrire projet documentation"})
        assert recovered.status_code == 200
        assert recovered.json()["citations"]


def test_extreme_finite_vectors_are_normalized_before_float32_storage(settings, source_manifest):
    _, chunks = load_chunks(source_manifest, LexicalEmbedder())
    assert len(chunks) >= 2
    store = VectorStore(settings)
    try:
        store.build("numeric-test", chunks[:2], [[1e300, 0.0], [0.0, 1e-300]], 2)
        results = store.search("numeric-test", [1e300, 0.0], 2)
        assert results[0].chunk.id == chunks[0].id
        assert results[0].score == pytest.approx(1.0)
        assert all(math.isfinite(item.score) for item in results)
    finally:
        store.close()
