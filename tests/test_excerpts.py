from datetime import UTC, datetime

import pytest

from lmaana_assistant.contracts import Chunk, Evidence, GeneratedAnswer, Source, Statement
from lmaana_assistant.errors import InvalidGeneration
from lmaana_assistant.generation.adapters import ExcerptGenerator, validate_citations
from lmaana_assistant.generation.excerpts import passage_blocks
from lmaana_assistant.normalization import process_query


def passages(text):
    return [
        Evidence(
            chunk=Chunk(
                id="software-fixture",
                source=Source(
                    id="software-fixture",
                    title="Test",
                    publisher="Test",
                    kind="text",
                    url="https://example.org/test",
                ),
                text=text,
                location="page 1",
                content_hash="fixture",
                fetched_at=datetime.now(UTC),
            ),
            score=0.8,
        )
    ]


def test_excerpt_targets_documents_instead_of_unrelated_intro():
    # Authored software fixture; no administrative requirements are invented here.
    text = (
        "Bienvenue dans le projet de test.\n"
        "Le projet propose des exemples et des démonstrations.\n"
        "Documents pour le test : fournir le nom du projet\n"
        "et le lien de la documentation.\n"
        "الوثائق ديال هاد الاختبار: سميّة المشروع ورابط التوثيق.\n"
        "Fin du guide logiciel."
    )
    evidence = passages(text)
    answer = ExcerptGenerator().generate(process_query("شنو الوثائق ديال المشروع؟"), evidence)
    assert answer.outcome == "source_excerpts"
    assert len(answer.statements) == 2
    assert all(statement.text in text for statement in answer.statements)
    assert all("Bienvenue" not in statement.text for statement in answer.statements)
    assert any("الوثائق" in statement.text for statement in answer.statements)
    validate_citations(answer, evidence)


@pytest.mark.parametrize(
    "question",
    [
        "شحال رسوم المشروع؟",
        "Quels frais pour le projet ?",
        "ما هي مدة المشروع؟",
        "Quel délai pour le projet ?",
        "Quelles conditions pour le projet ?",
    ],
)
def test_topic_match_without_requested_information_abstains(question):
    answer = ExcerptGenerator().generate(
        process_query(question), passages("Documents pour le projet : le nom et sa documentation.")
    )
    assert answer.outcome == "insufficient_evidence"
    assert all(not statement.citation_ids for statement in answer.statements)


@pytest.mark.parametrize(
    ("question", "text"),
    [
        ("Quels frais pour le logiciel ?", "Le logiciel de test est gratuit."),
        ("Quel délai pour le test ?", "Le test logiciel s’arrête après deux jours."),
    ],
)
def test_requested_information_is_not_always_withheld(question, text):
    answer = ExcerptGenerator().generate(process_query(question), passages(text))
    assert answer.outcome == "source_excerpts"
    assert answer.statements[0].text == text


def test_negative_clause_and_adjacent_exception_are_preserved():
    text = (
        "Les documents ne sont pas obligatoires dans ce test.\n"
        "Toutefois, fournir la documentation si le test active cette option."
    )
    answer = ExcerptGenerator().generate(
        process_query("Quels documents pour ce test ?"), passages(text)
    )
    assert answer.statements[0].text == text


def test_long_list_is_not_arbitrarily_cut():
    text = "Documents : " + "élément de test; " * 200 + "sauf si le test est désactivé."
    answer = ExcerptGenerator().generate(process_query("Quels documents ?"), passages(text))
    assert answer.outcome == "insufficient_evidence"


def test_blocks_keep_original_substrings_and_ignore_number_only_blocks():
    text = "1\n\nDocuments du projet\nUne ligne de test.\n2\nالمشروع ديال الاختبار."
    blocks = passage_blocks(text)
    assert all(block in text for block in blocks)
    assert all(any(char.isalpha() for char in block) for block in blocks)
    assert len(blocks) == 2


def test_pdf_step_markers_do_not_join_unrelated_steps():
    text = "عنوان الاختبار3\nملــف الاختبار فيه التوثيق.\n2\nخطوة أخرى."
    answer = ExcerptGenerator().generate(process_query("شنو الوثائق؟"), passages(text))
    assert answer.statements[0].text == "ملــف الاختبار فيه التوثيق."


@pytest.mark.parametrize("amount", ["1", "10", "100"])
def test_numeric_value_on_its_own_line_is_not_discarded(amount):
    text = f"Tarif du logiciel de test :\n{amount}\nunités fictives."
    answer = ExcerptGenerator().generate(
        process_query("Quels frais pour le logiciel ?"), passages(text)
    )
    assert answer.statements[0].text == text


def test_excerpt_with_real_citation_but_invented_text_is_rejected():
    evidence = passages("Documents de test.")
    answer = GeneratedAnswer(
        outcome="source_excerpts",
        statements=[Statement(text="Invented content", citation_ids=["software-fixture"])],
    )
    with pytest.raises(InvalidGeneration, match="verbatim"):
        validate_citations(answer, evidence)


@pytest.mark.parametrize(
    "question",
    [
        "Quel délai pour recevoir la carte de test ?",
        "ما هي مدة الحصول على بطاقة الاختبار؟",
        "Quel délai pour la carte de test ?",
    ],
)
def test_card_delivery_question_does_not_return_deposit_deadline(question):
    query = process_query(question)
    assert query.intent == "card_delivery_deadline"
    text = "Déposer la demande de test avec la copie de la carte sous trois jours."
    assert ExcerptGenerator().generate(query, passages(text)).outcome == "insufficient_evidence"


def test_card_delivery_answer_requires_event_and_duration():
    query = process_query("Quel délai pour recevoir la carte de test ?")
    text = "La carte de test est\ndélivrée sous deux jours."
    answer = ExcerptGenerator().generate(query, passages(text))
    assert answer.statements[0].text == text
    assert answer.outcome == "source_excerpts"
    assert (
        ExcerptGenerator()
        .generate(query, passages("La carte de test est délivrée à l’équipe."))
        .outcome
        == "insufficient_evidence"
    )


def test_deposit_deadline_query_stays_distinct_from_card_delivery():
    assert process_query("Quel délai de dépôt de la carte de test ?").intent == "deadlines"
