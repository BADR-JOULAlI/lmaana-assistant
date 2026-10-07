import pytest
from fastapi.testclient import TestClient

from lmaana_assistant.api.app import create_app
from lmaana_assistant.generation.adapters import ExcerptGenerator
from lmaana_assistant.language import detect_language, message, output_matches_language
from lmaana_assistant.normalization import process_query
from lmaana_assistant.pipeline import AnswerPipeline
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("شنو الوثائق لي خاصني باش ندير auto-entrepreneur؟", "ary"),
        ("بغيت نعرف الوثائق ديال التسجيل", "ary"),
        ("Quels documents pour devenir auto-entrepreneur ?", "fr"),
        ("ما هي الوثائق المطلوبة للتسجيل؟", "ar"),
        ("chno lwra9 li khassni bach ndir auto-entrepreneur?", "ary-Latn"),
        ("Bonjour, wach khassni une photo?", "ary-Latn"),
        ("What documents do I need to register?", "en"),
        ("واش je dois fournir une photo ديالي؟", "ary"),
        ("12345", "und"),
        ("auto-entrepreneur", "und"),
        ("https://example.org/fr", "und"),
        ("文書が必要です", "und"),
    ],
)
def test_supported_language_and_ambiguous_inputs(question, expected):
    decision = detect_language(question)
    assert decision.detected == decision.reply == expected
    assert decision.method == "rules"


def test_override_changes_reply_but_not_detected_language():
    decision = detect_language("شنو خاصني باش نسجل؟", "fr")
    assert decision.detected == "ary" and decision.reply == "fr"
    assert decision.method == "user_override"


def test_mixed_language_and_typographic_arabic():
    assert detect_language("شنو خاصني pour inscription?").mixed
    assert detect_language("شــنو خــاصني؟").detected == "ary"
    assert detect_language("كيف يمكنني التسجيل؟").detected == "ar"


@pytest.mark.parametrize("code", ["fr", "ar", "ary", "ary-Latn", "en"])
@pytest.mark.parametrize(
    "outcome", ["insufficient_evidence", "verification_required", "generation_failed"]
)
def test_localized_non_answers_have_expected_language(code, outcome):
    assert output_matches_language(message(outcome, code), code)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Vous devez fournir une copie.", "ary"),
        ("يجب تقديم نسخة من الطلب.", "fr"),
        ("يجب تقديم نسخة من الطلب.", "ary"),
        ("You need a copy of the form.", "fr"),
        ("خاصك نسخة ديال الطلب.", "ary-Latn"),
        ("خاصك نسخة ديال الطلب.", "ar"),
        ("يجب تقديم صورة شخصية و фото شخصي.", "ar"),
        ("You need a photo and 资料.", "en"),
    ],
)
def test_obvious_wrong_language_is_rejected(text, code):
    assert not output_matches_language(text, code)


def test_arabizi_expansion_preserves_original_question():
    question = "chno lwra9 khassni bach nsjel?"
    query = process_query(question)
    assert query.original == question
    assert query.intent == "required_documents"
    assert "copie" in query.search_text and "inscription" in query.search_text


def test_no_evidence_is_localized_even_without_llm(settings, indexed):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        for question, code in [
            ("Quels frais pour le projet ?", "fr"),
            ("شنو ثمن المشروع؟", "ary"),
            ("What fees for the project?", "en"),
        ]:
            result = pipeline.answer(question)
            assert result.outcome == "insufficient_evidence"
            assert result.answer_language == code
            assert result.answer == message("insufficient_evidence", code)
    finally:
        pipeline.close()


def test_unknown_language_requests_a_choice_without_generation(settings, indexed):
    class MustNotGenerate(ExcerptGenerator):
        def generate(self, *args):
            raise AssertionError("Ambiguous language must not invoke generation")

    pipeline = AnswerPipeline(settings, LexicalEmbedder(), MustNotGenerate())
    try:
        result = pipeline.answer("1234")
        assert result.outcome == "clarification_required"
        assert result.citations == []
    finally:
        pipeline.close()


def test_api_language_override_and_validation(settings, indexed):
    with TestClient(create_app(settings)) as client:
        result = client.post(
            "/v1/answers",
            json={"question": "Quels frais pour le projet ?", "response_language": "ar"},
        ).json()
        assert result["query"]["language"]["detected"] == "fr"
        assert result["answer_language"] == "ar"
        assert (
            client.post(
                "/v1/answers", json={"question": "bonjour", "response_language": "invalid"}
            ).status_code
            == 422
        )


def test_excerpt_mode_never_labels_original_source_as_translated(settings, indexed):
    pipeline = AnswerPipeline(settings, LexicalEmbedder(), ExcerptGenerator())
    try:
        result = pipeline.answer("Quels documents pour inscrire le projet ?", "ary")
        assert result.answer_language == "source"
        assert result.query.language.reply == "ary"
        assert result.outcome == "source_excerpts"
    finally:
        pipeline.close()
