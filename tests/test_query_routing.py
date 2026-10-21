"""Written-question regressions; no model server or administrative claims."""

import hashlib
import json

import pytest

from lmaana_assistant.generation.adapters import ExcerptGenerator
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.normalization import process_query, topic_matches
from lmaana_assistant.pipeline import AnswerPipeline
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder


@pytest.mark.parametrize(
    "question, language, intent",
    [
        (
            "شــنو الوثــائق لي خاصني باش ندير auto-entrepreneur؟",
            "ary",
            "required_documents",
        ),
        ("شحال من وثيقة خاصني باش نسجل؟", "ary", "required_documents"),
        ("شحال من ورقة خاصني باش نسجل؟", "ary", "required_documents"),
        (
            "ch7al mn nhar bach nakhod carte auto-entrepreneur?",
            "ary-Latn",
            "card_delivery_deadline",
        ),
        ("chhal mn simana bach nkemmel inscription?", "ary-Latn", "deadlines"),
        ("ch7al khassni nkhelles pour inscription?", "ary-Latn", "fees"),
        ("شحال غادي نخلص باش نسجل؟", "ary", "fees"),
        ("How can I register as an auto-entrepreneur?", "en", "procedure_steps"),
        (
            "What are the eligibility requirements for auto-entrepreneur registration?",
            "en",
            "eligibility",
        ),
        ("What fees do I have to pay?", "en", "fees"),
        ("Quels sont les coûts d’inscription ?", "fr", "fees"),
        ("Quel est le délai de dépôt du dossier ?", "fr", "deadlines"),
        ("How long do I have to submit the card application?", "en", "deadlines"),
    ],
)
def test_written_question_language_and_intent(question, language, intent):
    query = process_query(question)
    assert query.original == question
    assert query.language.detected == query.language.reply == language
    assert query.intent == intent


@pytest.mark.parametrize(
    "question",
    [
        "chno lwra9 li khasni bach nsajjel auto entrepreneur?",
        "chno lwra9 li khasni bach nsjel lmo9awil dati?",
        "شنو الوثــيقة المطلوبة للتســجيل للمقــاول الذاتي؟",
    ],
)
def test_search_aliases_cover_spelling_variants_without_rewriting_display(question):
    query = process_query(question)
    assert query.normalized == question
    assert query.intent == "required_documents"
    assert "documents pièces dossier" in query.search_text
    assert "inscription التسجيل" in query.search_text
    assert "auto entrepreneur المقاول الذاتي" in query.search_text
    assert topic_matches(question, "auto-entrepreneur-registration")
    assert not topic_matches(question, "passport")


@pytest.mark.parametrize("word", ["vaccin", "cinéma", "médecine", "cinnamon", "carte bancaire"])
def test_identity_alias_does_not_match_substrings_or_generic_cards(word):
    assert "CNIE identité بطاقة" not in process_query(f"Question sur {word} ?").search_text


@pytest.mark.parametrize("word", ["CIN", "cnie", "بطاقة التعريف", "البطاقة الوطنية"])
def test_identity_alias_still_recognizes_identity_documents(word):
    assert "CNIE identité بطاقة" in process_query(f"Question sur {word} ?").search_text


@pytest.mark.parametrize("question", ["documents: ch7al?", "شحال من وثيقة؟", "ch7al?", "شحال؟"])
def test_count_or_unspecified_quantity_does_not_assert_a_price(question):
    assert process_query(question).intent != "fees"


@pytest.mark.parametrize(
    "question, language",
    [
        ("شــنو الوثــائق ديال المشروع؟", "ary"),
        ("chno lwra9 li khasni bach nsajjel projet?", "ary-Latn"),
        ("What documents do I need to register the projet?", "en"),
        ("ما هي الوثيقة المطلوبة للتسجيل في projet؟", "ar"),
        ("Quels documents pour inscrire le projet ?", "fr"),
    ],
)
def test_multilingual_documents_questions_reach_explicit_document_fixture(
    settings, source_manifest, question, language
):
    # This lexical fixture intentionally names its document section. It does not
    # demonstrate semantic equivalence for passages with no shared vocabulary.
    path = source_manifest.parent / "guide.html"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "<p>Pour inscrire", "<p>Documents du dossier pour inscrire"
        ),
        encoding="utf-8",
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["sources"][0]["verification"]["content_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    ingest(source_manifest, settings, LexicalEmbedder())
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        result = pipeline.answer(question)
        assert result.query.language.reply == language
        assert result.outcome == "source_excerpts"
        # Do not mislabel French excerpts as translations.
        assert result.answer_language == "source"
        assert result.citations[0].url == "https://example.org/project-guide"
        assert all(statement.text in result.citations[0].excerpt for statement in result.statements)
    finally:
        pipeline.close()


def test_lexical_profile_cannot_assume_cross_language_semantic_equivalence(settings, indexed):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        # The French fixture says "fournir" and "documentation", not "documents".
        # Query expansion is not a trained multilingual semantic retriever.
        result = pipeline.answer("شــنو الوثــائق ديال المشروع؟")
        assert result.outcome == "insufficient_evidence"
        assert not result.citations
        assert any("semantic quality is untested" in warning for warning in result.warnings)
    finally:
        pipeline.close()


@pytest.mark.parametrize(
    "question",
    [
        "What are the eligibility requirements for the projet documentation?",
        "ch7al mn nhar bach nakhod carte dyal projet?",
    ],
)
@pytest.mark.parametrize("backend", ["excerpt", "ollama"])
def test_missing_requested_detail_withholds_even_before_neural_generation(
    settings, indexed, question, backend
):
    class MustNotGenerate(ExcerptGenerator):
        def generate(self, *args):
            raise AssertionError("Unsupported detail must be withheld before calling the LLM")

    settings.generator_backend = backend
    generator = MustNotGenerate() if backend != "excerpt" else ExcerptGenerator()
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), generator)
    try:
        # Prove the safety check is not just passing because retrieval found nothing.
        query = process_query(question)
        hits = pipeline.store.search(
            pipeline.release,
            pipeline.embedder.encode_query(query.search_text),
            settings.retrieval_k,
        )
        assert hits and hits[0].score >= settings.min_score
        result = pipeline.answer(question)
        assert result.outcome == "insufficient_evidence"
        assert not result.citations
        assert not result.generation_failed
    finally:
        pipeline.close()
