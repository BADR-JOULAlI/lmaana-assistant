"""Build and validate a new collection before atomically activating its manifest."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import (
    CORPUS_SCHEMA_VERSION,
    QUERY_RULES_VERSION,
    SOURCE_POLICY_VERSION,
    Embedder,
)
from lmaana_assistant.ingestion.sources import CHUNKER_VERSION, load_chunks
from lmaana_assistant.retrieval.store import VectorStore


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest(manifest_path: Path, settings: Settings, embedder: Embedder) -> dict:
    # Acquire the local DB lock before network calls, downloads, or index mutations.
    store = VectorStore(settings)
    try:
        source_manifest, chunks = load_chunks(manifest_path, embedder)
        fingerprint = {
            "sources": source_manifest.model_dump(mode="json"),
            "chunks": [(chunk.id, chunk.content_hash) for chunk in chunks],
            "embedding": embedder.fingerprint,
            "chunker": CHUNKER_VERSION,
            "query_rules": QUERY_RULES_VERSION,
            "source_policy": SOURCE_POLICY_VERSION,
        }
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        # Unique release allows retrying a failed build without touching its collection.
        release = f"corpus_{digest[:12]}_{uuid4().hex[:8]}"
        vectors = embedder.encode_documents([chunk.text for chunk in chunks])
        store.build(release, chunks, vectors, embedder.dimension)
        manifest = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "source_policy": SOURCE_POLICY_VERSION,
            "release": release,
            "content_fingerprint": digest,
            "embedding": embedder.fingerprint,
            "dimension": embedder.dimension,
            "distance": "cosine",
            "query_rules": QUERY_RULES_VERSION,
            "chunker": CHUNKER_VERSION,
            "chunk_count": len(chunks),
            "source_count": len(source_manifest.sources),
            "created_at": datetime.now(UTC).isoformat(),
            "sources": source_manifest.model_dump(mode="json")["sources"],
        }
        atomic_json(settings.corpus_dir / f"{release}.json", manifest)
        atomic_json(settings.active_path, manifest)
        return manifest
    finally:
        store.close()
