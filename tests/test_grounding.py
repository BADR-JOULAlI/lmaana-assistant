import pytest

from lmaana_assistant.contracts import GeneratedAnswer, Statement
from lmaana_assistant.errors import InvalidGeneration
from lmaana_assistant.generation.grounding import validate_source_constraints
from lmaana_assistant.normalization import process_query


def test_passport_must_not_replace_an_identity_card(evidence):
    evidence[0].chunk.text = "Une copie de la carte nationale d'identité."
    for text in ("You need a passport.", "نسخة من جواز السفر.", "Un passeport."):
        answer = GeneratedAnswer(
            outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
        )
        with pytest.raises(InvalidGeneration, match="identity document"):
            validate_source_constraints(answer, process_query("What documents?"), evidence)


@pytest.mark.parametrize("text", ["The test lasts 99 days.", "المدة ٩٩ يوما."])
def test_unseen_numbers_are_rejected(evidence, text):
    evidence[0].chunk.text = "The software test lasts 12 days."
    answer = GeneratedAnswer(
        outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
    )
    with pytest.raises(InvalidGeneration, match="number"):
        validate_source_constraints(answer, process_query("What is the test duration?"), evidence)


@pytest.mark.parametrize(
    ("source", "good", "bad"),
    [
        ("Signer le formulaire.", "The form must be signed.", "You need a form."),
        ("Une photo personnelle.", "You need a personal photo.", "You need an identity card."),
        (
            "Une copie de la carte nationale d'identité.",
            "A copy of your national ID.",
            "Your national ID.",
        ),
        (
            "La carte de séjour pour les étrangers.",
            "A residence card for foreigners.",
            "An identity card.",
        ),
        (
            "Banques partenaires de Barid Al-Maghrib.",
            "Partner banks of Barid Al-Maghrib.",
            "Any bank.",
        ),
        ("Déposer dans 30 jours.", "Submit within 30 days.", "Submit the form."),
    ],
)
def test_source_conditions_are_retained(evidence, source, good, bad):
    # Isolated translation-regression fixtures, not administrative advice.
    evidence[0].chunk.text = source
    query = process_query("What documents are required?")
    for text in (good, bad):
        answer = GeneratedAnswer(
            outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
        )
        if text == good:
            validate_source_constraints(answer, query, evidence)
        else:
            with pytest.raises(InvalidGeneration):
                validate_source_constraints(answer, query, evidence)


def test_constraints_do_not_invent_requirements(evidence):
    evidence[0].chunk.text = "The software project only requires its name."
    answer = GeneratedAnswer(
        outcome="answered",
        statements=[Statement(text="You need the project name.", citation_ids=["chunk-0"])],
    )
    validate_source_constraints(answer, process_query("What documents?"), evidence)


def test_list_labels_are_not_treated_as_new_factual_numbers(evidence):
    evidence[0].chunk.text = "The project needs a name and a link."
    answer = GeneratedAnswer(
        outcome="answered",
        statements=[Statement(text="You need: 1. a name, 2. a link.", citation_ids=["chunk-0"])],
    )
    validate_source_constraints(answer, process_query("What documents?"), evidence)


@pytest.mark.parametrize(
    "text",
    ["You need an email address.", "خاصك ايميل ديالك.", "Khassk un email dyalk."],
)
def test_email_requirement_is_source_triggered(evidence, text):
    evidence[0].chunk.text = "Avoir une adresse électronique."
    query = process_query("What documents are required?")
    answer = GeneratedAnswer(
        outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
    )
    validate_source_constraints(answer, query, evidence)
    answer.statements[0].text = "You need a form."
    with pytest.raises(InvalidGeneration, match="source condition"):
        validate_source_constraints(answer, query, evidence)


@pytest.mark.parametrize(
    "text",
    [
        "You must fill the application online.",
        "Une demande remplie électroniquement sur le portail.",
        "خاصك تعمّر الطلب فالمنصة الإلكترونية.",
        "Khassk t3mmer demande online.",
    ],
)
def test_online_application_cannot_be_omitted(evidence, text):
    evidence[0].chunk.text = "Remplir la demande d'une manière électronique via le portail."
    query = process_query("What documents are required?")
    answer = GeneratedAnswer(
        outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
    )
    validate_source_constraints(answer, query, evidence)
    answer.statements[0].text = "Submit the form to the office."
    with pytest.raises(InvalidGeneration, match="online application"):
        validate_source_constraints(answer, query, evidence)


@pytest.mark.parametrize(
    "text",
    [
        "Submit the application within 12 days from its creation.",
        "Déposez la demande dans les 12 jours à partir de sa création.",
        "خاصك تدفع الطلب فمدة 12 يوم من نهار تسجلتيه.",
        "Khassk tdeffe3 talab f 12 iyam men nhar tsjjelti demande.",
    ],
)
def test_submission_deadline_keeps_its_start_event(evidence, text):
    evidence[
        0
    ].chunk.text = (
        "Déposer la demande dans 12 jours à compter de la date de création de la demande."
    )
    query = process_query("What documents are required?")
    answer = GeneratedAnswer(
        outcome="answered", statements=[Statement(text=text, citation_ids=["chunk-0"])]
    )
    validate_source_constraints(answer, query, evidence)
    for invalid in ("You receive the card in 12 days.", "Submit within 12 days."):
        answer.statements[0].text = invalid
        with pytest.raises(InvalidGeneration, match="start event"):
            validate_source_constraints(answer, query, evidence)
    answer.statements[0].text = "Submit the application within 12 days after receiving your card."
    with pytest.raises(InvalidGeneration, match="changed its starting event"):
        validate_source_constraints(answer, query, evidence)
