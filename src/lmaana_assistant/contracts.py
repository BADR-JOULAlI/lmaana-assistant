"""Shared data contracts, independent of HTTP and model runtimes."""

from datetime import date, datetime, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

QUERY_RULES_VERSION = "darija-rules-v4"
CORPUS_SCHEMA_VERSION = 2
SOURCE_POLICY_VERSION = "reviewed-currency-v1"
LANGUAGE_POLICY_VERSION = "language-routing-v2"
GENERATION_POLICY_VERSION = "generation-validation-v2"
ReplyLanguage = Literal["fr", "ar", "ary", "ary-Latn", "en", "und"]
LanguagePreference = Literal["auto", "fr", "ar", "ary", "ary-Latn", "en"]
AnswerOutcome = Literal[
    "answered",
    "source_excerpts",
    "clarification_required",
    "insufficient_evidence",
    "verification_required",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LanguageDecision(StrictModel):
    detected: ReplyLanguage = "und"
    reply: ReplyLanguage = "und"
    confidence: Literal["low", "medium", "high"] = "low"
    mixed: bool = False
    method: Literal["rules", "user_override"] = "rules"
    version: str = LANGUAGE_POLICY_VERSION


class SourceVerification(StrictModel):
    """Operator review, never inferred from download time or the site's hostname."""

    status: Literal["unverified", "historical", "verified"] = "unverified"
    checked_on: date | None = None
    review_due: date | None = None
    evidence_urls: list[HttpUrl] = Field(default_factory=list, max_length=8)
    note: str = ""
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_review(self):
        if self.status == "verified":
            if not all(
                (
                    self.checked_on,
                    self.review_due,
                    self.evidence_urls,
                    self.note.strip(),
                    self.content_sha256,
                )
            ):
                raise ValueError("Verified sources need dated, content-bound review evidence.")
            if not self.checked_on < self.review_due <= self.checked_on + timedelta(days=90):
                raise ValueError("Review due must be 1 to 90 days after the verification date.")
        elif self.review_due is not None or self.content_sha256 is not None:
            raise ValueError("Only verified sources may carry a currency approval.")
        return self


class PdfSection(StrictModel):
    """Reviewed, bounded section spanning consecutive PDF pages.

    Header and folio removal is opt-in and exact, not a general PDF repair rule.
    Markers match normalized whitespace only; body characters are preserved.
    """

    start_marker: str = Field(min_length=1)
    end_marker: str = Field(min_length=1)
    running_header: str = Field(min_length=1)
    printed_page_offset: int = Field(default=0, ge=-20, le=20)

    @field_validator("start_marker", "end_marker", "running_header")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF section markers cannot be blank.")
        return value


class Source(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: HttpUrl
    language: str = "fr"
    topic: str = "general"
    kind: Literal["html", "pdf", "text"]
    reviewed: bool = False
    reuse_status: str = "Check source terms before redistribution"
    freshness_note: str = "Current applicability has not been independently verified."
    verification: SourceVerification = Field(default_factory=SourceVerification)
    local_path: str | None = None
    selector: str | None = None
    section_heading: str | None = None
    pages: list[int] | None = None
    pdf_section: PdfSection | None = None

    @model_validator(mode="after")
    def valid_pdf_section(self):
        if self.pdf_section is not None:
            if (
                self.kind != "pdf"
                or not self.pages
                or self.section_heading is not None
                or self.selector is not None
            ):
                raise ValueError("PDF sections require explicit PDF pages and no HTML selector.")
            if self.pages != list(range(self.pages[0], self.pages[-1] + 1)):
                raise ValueError("PDF section pages must be consecutive, unique, and ordered.")
        return self

    @field_validator("pages")
    @classmethod
    def valid_pages(cls, value):
        if value is not None and (not value or any(page < 1 for page in value)):
            raise ValueError("PDF page numbers must be positive and one-based.")
        return value


class SourceManifest(StrictModel):
    allowed_hosts: list[str]
    sources: list[Source] = Field(min_length=1)


class Chunk(StrictModel):
    id: str
    source: Source
    text: str
    location: str
    content_hash: str
    fetched_at: datetime


class Evidence(StrictModel):
    chunk: Chunk
    score: float


class ProcessedQuery(StrictModel):
    original: str
    normalized: str
    search_text: str
    intent: str
    version: str = QUERY_RULES_VERSION
    language: LanguageDecision = Field(default_factory=LanguageDecision)


class AnswerRequest(StrictModel):
    question: str = Field(min_length=1, max_length=1500)
    response_language: LanguagePreference = "auto"

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty.")
        return value


class TranscriptionResponse(StrictModel):
    id: str
    text: str
    model: str
    revision: str
    audio_duration_ms: int
    confidence: float | None
    segments: list[dict]
    warnings: list[str]
    timings_ms: dict[str, float]


class Statement(StrictModel):
    text: str = Field(min_length=1, max_length=3000)
    citation_ids: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A statement cannot be blank.")
        return value.strip()


class GeneratedAnswer(StrictModel):
    outcome: AnswerOutcome
    statements: list[Statement] = Field(min_length=1, max_length=10)


class Citation(StrictModel):
    id: str
    title: str
    publisher: str
    url: str
    location: str
    excerpt: str
    fetched_at: datetime
    freshness_note: str
    verification_status: str
    checked_on: date | None
    review_due: date | None
    verification_note: str


class AnswerResponse(StrictModel):
    request_id: str
    outcome: AnswerOutcome
    answer: str
    statements: list[Statement]
    citations: list[Citation]
    related_sources: list[Citation]
    source_policy: str = SOURCE_POLICY_VERSION
    language_policy: str = LANGUAGE_POLICY_VERSION
    answer_language: str = "source"
    generation_failed: bool = False
    query: ProcessedQuery
    corpus_release: str
    embedding: str
    generator: str
    timings_ms: dict[str, float]
    warnings: list[str]


class Embedder(Protocol):
    fingerprint: str
    dimension: int

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...

    def split(self, text: str, size: int, overlap: int) -> list[str]: ...


class Generator(Protocol):
    name: str

    def ready(self) -> bool: ...

    def generate(self, query: ProcessedQuery, evidence: list[Evidence]) -> GeneratedAnswer: ...

    def close(self) -> None: ...
