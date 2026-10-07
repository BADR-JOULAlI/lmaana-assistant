"""Deterministic source-currency gate, shared by all answer generators."""

from datetime import UTC, date, datetime

from lmaana_assistant.contracts import Chunk, Citation, Source

VERIFICATION_REQUIRED = (
    "لقيت مراجع على هاد الموضوع، ولكن ما تأكدتش واش المعلومات ديالها ما زالت صالحة دابا. "
    "ما نقدرش نعطيك جواب مؤكد اعتماداً عليها. تأكد من الجهة الرسمية قبل ما تعتمد عليها."
)


def utc_today() -> date:
    return datetime.now(UTC).date()


def verification_status(source: Source, content_hash: str, today: date) -> str:
    review = source.verification
    if not source.reviewed:
        return "unverified"
    if review.status != "verified":
        return review.status
    if review.content_sha256 != content_hash:
        return "content_changed"
    if today < review.checked_on:
        return "not_yet_valid"
    if today >= review.review_due:
        return "expired"
    return "verified"


def source_reference(chunk: Chunk, today: date) -> Citation:
    review = chunk.source.verification
    return Citation(
        id=chunk.id,
        title=chunk.source.title,
        publisher=chunk.source.publisher,
        url=str(chunk.source.url),
        location=chunk.location,
        excerpt=chunk.text,
        fetched_at=chunk.fetched_at,
        freshness_note=chunk.source.freshness_note,
        verification_status=verification_status(chunk.source, chunk.content_hash, today),
        checked_on=review.checked_on,
        review_due=review.review_due,
        verification_note=review.note,
    )
