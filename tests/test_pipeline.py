import json

import pytest

from lmaana_assistant.contracts import GeneratedAnswer, Statement
from lmaana_assistant.errors import DependencyUnavailable
from lmaana_assistant.generation.adapters import ExcerptGenerator
from lmaana_assistant.normalization import lexical_tokens, process_query, topic_matches
from lmaana_assistant.pipeline import AnswerPipeline
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder


def test_normalization_preserves_numbers_negation_and_french():
    raw = "  ما بغيتش  ندير auto-entrepreneur فـ 2026  "
    result = process_query(raw)
    assert result.original == raw
    assert "ما بغيتش" in result.normalized
    assert "auto-entrepreneur" in result.normalized
    assert "2026" in result.normalized


def test_arabic_contracted_topic_and_darija_variants():
    result = process_query("الوثايق المطلوبة للمقاول الذاتي؟")
    assert "auto entrepreneur" in result.search_text
    assert result.intent == "required_documents"
    assert result.version == "darija-rules-v4"


def test_duration_question_is_not_mistaken_for_a_fee():
    assert process_query("شحال من نهار للحصول على البطاقة؟").intent == "card_delivery_deadline"


def test_arabic_typographic_stretching_is_search_only():
    raw = "ملــف التســجيل"
    assert lexical_tokens(raw) == lexical_tokens("ملف التسجيل")
    assert process_query(raw).original == raw


def test_excerpt_round_trip_and_unanswerable_question(settings, indexed):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        result = pipeline.answer("Quels documents pour inscrire un projet ?")
        assert result.outcome == "source_excerpts"
        assert result.citations[0].url == "https://example.org/project-guide"
        assert result.citations[0].location == "Inscription du projet"
        assert "[1]" in result.answer
        assert result.corpus_release == indexed["release"]
        assert (
            pipeline.answer("What about quantum zebras pineapple?").outcome
            == "insufficient_evidence"
        )
    finally:
        pipeline.close()


class InvalidCitationGenerator(ExcerptGenerator):
    calls = 0

    def generate(self, query, evidence):
        self.calls += 1
        return GeneratedAnswer(
            outcome="answered",
            statements=[Statement(text="An invented answer.", citation_ids=["invented-id"])],
        )


def test_invalid_citation_retried_once_then_withheld(settings, indexed):
    generator = InvalidCitationGenerator()
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), generator)
    try:
        result = pipeline.answer("inscrire projet documentation")
        assert generator.calls == 2
        assert result.outcome == "insufficient_evidence"
        assert result.generation_failed
        assert "contrôles" in result.answer
        assert result.citations == []
        assert "invented" not in result.answer
    finally:
        pipeline.close()


def test_embedding_mismatch_fails_before_querying(settings, indexed):
    indexed["embedding"] = "some-other-embedding"
    settings.active_path.write_text(json.dumps(indexed))
    with pytest.raises(DependencyUnavailable, match="differ"):
        AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())


def test_old_pdf_chunking_is_rejected_until_reingestion(settings, indexed):
    indexed["chunker"] = "source-boundaries-v1-size350-overlap50"
    settings.active_path.write_text(json.dumps(indexed))
    with pytest.raises(DependencyUnavailable, match="differ"):
        AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())


@pytest.mark.parametrize(
    "question",
    [
        "Quels documents pour l’inscription universitaire ?",
        "الوثائق للتسجيل بالجامعة؟",
        "Quels documents pour le passeport ?",
        "شنو الوثائق ديال جواز السفر؟",
        "Comparer le dossier universitaire et auto-entrepreneur",
    ],
)
def test_explicit_other_domain_cannot_reuse_registration_word_overlap(question):
    assert not topic_matches(question, "auto-entrepreneur-registration")


def test_known_topic_and_unknown_topic_routes():
    assert topic_matches("الوثائق للمقاول الذاتي؟", "auto-entrepreneur-registration")
    assert topic_matches("inscription universitaire", "university-registration")
    assert topic_matches("documents du projet", "general")


def test_topic_mismatch_withholds_verified_answers_and_archives(settings, indexed):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        result = pipeline.answer("Quels documents pour inscrire un projet universitaire ?")
        assert result.outcome == "insufficient_evidence"
        assert result.citations == [] and result.related_sources == []
    finally:
        pipeline.close()
