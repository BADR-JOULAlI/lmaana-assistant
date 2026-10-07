from datetime import date

from lmaana_assistant.contracts import LANGUAGE_POLICY_VERSION, SOURCE_POLICY_VERSION
from lmaana_assistant.presentation import can_display_answer, paragraphs_html


def test_display_reflows_pdf_wraps_without_rewriting_words():
    result = paragraphs_html(
        "Formulaire de pré-\ninscription, une copie\net une photo.", reflow=True
    )
    assert "pré-inscription, une copie et une photo." in result
    assert "<br>" not in result


def test_html_is_escaped_and_mixed_script_paragraphs_are_isolated():
    result = paragraphs_html('مرحبا <script>alert(1)</script>\n\nFrench & "quoted" text.')
    assert '<p dir="rtl">' in result
    assert '<p dir="ltr">' in result
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&amp;" in result


def test_generated_answer_lists_keep_line_breaks():
    assert "أولا<br>ثانيا" in paragraphs_html("أولا\nثانيا")


def test_cached_answer_is_hidden_on_upgrade_corpus_change_or_review_expiry():
    health = {
        "ready": True,
        "source_policy": SOURCE_POLICY_VERSION,
        "corpus_release": "test",
        "language_policy": LANGUAGE_POLICY_VERSION,
        "generator": "excerpt",
    }
    answer = {
        "source_policy": SOURCE_POLICY_VERSION,
        "corpus_release": "test",
        "language_policy": LANGUAGE_POLICY_VERSION,
        "generator": "excerpt",
        "citations": [
            {
                "verification_status": "verified",
                "checked_on": "2026-10-03",
                "review_due": "2026-10-10",
            }
        ],
    }
    assert can_display_answer(answer, health, date(2026, 10, 3))
    assert not can_display_answer(answer, health, date(2026, 10, 10))
    assert not can_display_answer(answer, {**health, "ready": False}, date(2026, 10, 3))
    assert not can_display_answer(answer, {**health, "corpus_release": "new"}, date(2026, 10, 3))
    assert not can_display_answer({**answer, "source_policy": "old"}, health, date(2026, 10, 3))
    assert not can_display_answer({**answer, "language_policy": "old"}, health, date(2026, 10, 3))
    assert not can_display_answer(answer, {**health, "generator": "llamacpp"}, date(2026, 10, 3))
