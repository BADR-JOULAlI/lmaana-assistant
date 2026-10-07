import json
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from lmaana_assistant.contracts import Source, SourceVerification
from lmaana_assistant.errors import InvalidGeneration
from lmaana_assistant.generation.adapters import ExcerptGenerator
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.ingestion.sources import extract_sections
from lmaana_assistant.pipeline import AnswerPipeline
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder
from lmaana_assistant.source_policy import verification_status


def reviewed_source(**updates):
    review = {
        "status": "verified",
        "checked_on": "2026-10-03",
        "review_due": "2026-11-02",
        "evidence_urls": ["https://example.org/review"],
        "note": "Authored software fixture.",
        "content_sha256": "a" * 64,
    }
    review.update(updates)
    return Source(
        id="test",
        title="Test",
        publisher="Test",
        kind="text",
        url="https://example.org",
        reviewed=True,
        verification=review,
    )


@pytest.mark.parametrize(
    "missing",
    [
        "checked_on",
        "review_due",
        "evidence_urls",
        "content_sha256",
        "note",
    ],
)
def test_verified_status_alone_cannot_approve_a_source(missing):
    source = reviewed_source().model_dump()
    del source["verification"][missing]
    with pytest.raises(ValidationError):
        Source.model_validate(source)


@pytest.mark.parametrize("due", ["2026-10-03", "2026-10-02", "2030-01-01"])
def test_review_window_is_bounded(due):
    with pytest.raises(ValidationError):
        reviewed_source(review_due=due)


def test_future_expired_and_changed_reviews_are_not_eligible():
    source = reviewed_source()
    assert verification_status(source, "a" * 64, date(2026, 10, 3)) == "verified"
    assert verification_status(source, "a" * 64, date(2026, 10, 2)) == "not_yet_valid"
    assert verification_status(source, "a" * 64, date(2026, 11, 2)) == "expired"
    assert verification_status(source, "b" * 64, date(2026, 10, 3)) == "content_changed"


@pytest.mark.parametrize("status", ["historical", "unverified"])
def test_recent_inspection_does_not_make_source_current(status):
    source = reviewed_source()
    source.verification = SourceVerification(status=status, checked_on=date(2026, 10, 3))
    assert verification_status(source, "a" * 64, date(2026, 10, 3)) == status


class MustNotGenerate(ExcerptGenerator):
    def generate(self, query, evidence):
        raise AssertionError("Unverified source reached the generator")


@pytest.mark.parametrize("status", ["historical", "unverified", "expired"])
def test_unverified_source_is_withheld_before_generation(settings, source_manifest, status):
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if status == "expired":
        today = datetime.now(UTC).date()
        manifest["sources"][0]["verification"]["checked_on"] = (
            today - timedelta(days=5)
        ).isoformat()
        manifest["sources"][0]["verification"]["review_due"] = today.isoformat()
    else:
        manifest["sources"][0]["verification"] = {"status": status}
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    ingest(source_manifest, settings, LexicalEmbedder())
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), MustNotGenerate())
    try:
        answer = pipeline.answer("Quels documents pour inscrire le projet ?")
        assert answer.outcome == "verification_required"
        assert answer.citations == []
        assert answer.related_sources[0].verification_status == status
        assert "fournir" not in answer.answer
        assert all(not statement.citation_ids for statement in answer.statements)
        assert (
            pipeline.answer("What about quantum zebras pineapple?").outcome
            == "insufficient_evidence"
        )
    finally:
        pipeline.close()


def test_currency_expires_without_restarting_the_server(settings, indexed, monkeypatch):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        question = "Quels documents pour inscrire le projet ?"
        assert pipeline.answer(question).outcome == "source_excerpts"
        due = date.fromisoformat(indexed["sources"][0]["verification"]["review_due"])
        monkeypatch.setattr("lmaana_assistant.pipeline.utc_today", lambda: due)
        assert pipeline.answer(question).outcome == "verification_required"
    finally:
        pipeline.close()


def test_changed_reviewed_content_does_not_replace_active_corpus(
    settings, source_manifest, indexed
):
    document = source_manifest.parent / "guide.html"
    document.write_text(document.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since verification"):
        ingest(source_manifest, settings, LexicalEmbedder())
    assert json.loads(settings.active_path.read_text())["release"] == indexed["release"]


def test_historical_duplicate_does_not_hide_verified_source(settings, source_manifest):
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    historical = {
        **manifest["sources"][0],
        "id": "historical",
        "verification": {"status": "historical"},
    }
    manifest["sources"].insert(0, historical)
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    ingest(source_manifest, settings, LexicalEmbedder())
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        answer = pipeline.answer("Quels documents pour inscrire le projet ?")
        assert answer.outcome == "source_excerpts"
        assert all(c.verification_status == "verified" for c in answer.citations)
        assert answer.related_sources[0].verification_status == "historical"
    finally:
        pipeline.close()


def test_generation_failure_is_not_misreported_as_unverified_evidence(settings, source_manifest):
    class RejectedGenerator(ExcerptGenerator):
        def generate(self, query, evidence):
            raise InvalidGeneration("Wrong response language")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["sources"].append(
        {
            **manifest["sources"][0],
            "id": "historical",
            "verification": {"status": "historical"},
        }
    )
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    ingest(source_manifest, settings, LexicalEmbedder())
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), RejectedGenerator())
    try:
        result = pipeline.answer("Quels documents pour inscrire le projet ?")
        assert result.generation_failed
        assert result.outcome == "insufficient_evidence"
        assert "contrôles" in result.answer
        assert not result.citations
        assert result.related_sources[0].verification_status == "historical"
    finally:
        pipeline.close()


def test_html_section_filter_is_exact_and_rejects_missing_heading():
    source = reviewed_source()
    source.kind = "html"
    source.section_heading = "Selected"
    data = (
        b"<main><h2>Ignore</h2><p>Unrelated content</p>"
        b"<h2>Selected</h2><p>Relevant content</p></main>"
    )
    assert extract_sections(source, data) == [("Selected", "Relevant content")]
    source.section_heading = "Absent"
    with pytest.raises(ValueError, match="review"):
        extract_sections(source, data)
