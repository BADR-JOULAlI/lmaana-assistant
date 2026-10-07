"""Qdrant local store. One client owns the directory for its full lifetime."""

from qdrant_client import QdrantClient, models

from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import Chunk, Evidence
from lmaana_assistant.errors import DependencyUnavailable


class VectorStore:
    def __init__(self, settings: Settings):
        directory = settings.data_dir / "indexes" / "qdrant"
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client = QdrantClient(path=str(directory))
        except RuntimeError as exc:
            raise DependencyUnavailable(
                "Corpus storage is locked. Stop the API or ingestion process first."
            ) from exc

    def exists(self, release: str) -> bool:
        return self.client.collection_exists(release)

    def build(
        self, release: str, chunks: list[Chunk], vectors: list[list[float]], dimension: int
    ) -> None:
        if len(chunks) != len(vectors) or any(len(vector) != dimension for vector in vectors):
            raise ValueError("Embedding count or dimension does not match the corpus.")
        if self.exists(release):
            raise ValueError("Refusing to overwrite an existing corpus release.")
        self.client.create_collection(
            release,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), 64):
            self.client.upsert(
                release,
                points=[
                    models.PointStruct(
                        id=chunk.id, vector=vector, payload=chunk.model_dump(mode="json")
                    )
                    for chunk, vector in zip(
                        chunks[start : start + 64], vectors[start : start + 64], strict=True
                    )
                ],
            )
        count = self.client.count(release, exact=True).count
        if count != len(chunks):
            raise ValueError("Corpus validation failed: stored chunk count does not match.")

    def search(self, release: str, vector: list[float], limit: int) -> list[Evidence]:
        points = self.client.query_points(
            release, query=vector, limit=limit, with_payload=True
        ).points
        return [
            Evidence(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in points
        ]

    def close(self) -> None:
        self.client.close()
