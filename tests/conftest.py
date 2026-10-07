import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import Chunk, Evidence, Source
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder


@pytest.fixture
def evidence():
    """Shared software passages, never fabricated administrative requirements."""
    source = Source(
        id="test", title="Test", publisher="Test", kind="text", url="https://example.org/guide"
    )
    return [
        Evidence(
            chunk=Chunk(
                id=f"chunk-{index}",
                source=source,
                text=f"Software project passage {index}",
                location=f"section {index}",
                content_hash="123",
                fetched_at=datetime.now(UTC),
            ),
            score=0.9,
        )
        for index in range(2)
    ]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", min_score=0.1)


@pytest.fixture
def source_manifest(tmp_path):
    # Authored software-project fixtures, not invented administrative requirements.
    (tmp_path / "guide.html").write_text(
        "<html><main><h1>Lmaana documentation</h1><h2>Inscription du projet</h2>"
        "<p>Pour inscrire un projet dans ce test, fournir le nom du projet et le lien "
        "de sa documentation. Ce scénario est une fixture de test logiciel.</p>"
        "<h2>Questions sans réponse</h2><p>Le projet affiche ses sources quand elles "
        "sont disponibles.</p></main><footer>Ne pas indexer le pied de page</footer></html>",
        encoding="utf-8",
    )
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "allowed_hosts": [],
                "sources": [
                    {
                        "id": "project-guide",
                        "title": "Lmaana test guide",
                        "publisher": "Lmaana tests",
                        "url": "https://example.org/project-guide",
                        "kind": "html",
                        "reviewed": True,
                        "local_path": "guide.html",
                        "selector": "main",
                        "freshness_note": "Software test fixture; not administrative guidance.",
                        "verification": {
                            "status": "verified",
                            "checked_on": datetime.now(UTC).date().isoformat(),
                            "review_due": (
                                datetime.now(UTC).date() + timedelta(days=30)
                            ).isoformat(),
                            "evidence_urls": ["https://example.org/project-guide"],
                            "note": "Authored software fixture reviewed for testing only.",
                            "content_sha256": hashlib.sha256(
                                (tmp_path / "guide.html").read_bytes()
                            ).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def indexed(settings, source_manifest):
    return ingest(source_manifest, settings, LexicalEmbedder())
