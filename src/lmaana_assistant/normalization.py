"""Conservative query processing; the user's original wording remains available."""

import re
import unicodedata

from lmaana_assistant.contracts import LanguagePreference, ProcessedQuery
from lmaana_assistant.language import detect_language

# Patterns run on a search-only form; displayed questions and passages stay intact.
AUTO_ENTREPRENEUR = (
    r"\bauto[\s-]?entrepreneur\b|(?:المقاول|للمقاول|مقاول)\s+(?:ال)?ذاتي"
    r"|\bl?m[oa][9q]aw[iy]l\s+(?:dati|ddati|dhati)\b"
)
DOCUMENTS = (
    r"وثايق|وثائق|وثيقة|ورقة|الوراق|لوراق"
    r"|\b(?:documents?|pieces?|papiers?|lwra9|wra9|lwrak|wathai9|papers?)\b"
)
REGISTRATION = (
    r"نسجل|تسجيل|ندير"
    r"|\b(?:inscription|inscrire|register(?:ing)?|registration|nsjel|nsaj+el|ndir)\b"
)
IDENTITY_DOCUMENT = r"\b(?:cin|cnie)\b|(?:ال)?بطاقة\s+(?:التعريف|الوطنية)"
ALIASES = (
    (AUTO_ENTREPRENEUR, "auto entrepreneur المقاول الذاتي"),
    (DOCUMENTS, "documents pièces dossier formulaire demande copie الوثائق"),
    (REGISTRATION, "inscription التسجيل"),
    (IDENTITY_DOCUMENT, "CNIE identité بطاقة"),
)
INTENTS = (
    (
        "deadlines",
        r"\b(?:اجل|مدة|وقت|delais?|duree|date\s+limite|deadlines?|how\s+long)\b"
        r"|(?:شحال|ch7al|chhal)\b.{0,32}\b(?:نهار|ايام|يوم|شهر|ساعة|اسبوع"
        r"|nhar|iyam|ayam|simana|chher|sa3a|days?|weeks?|months?|hours?)\b",
    ),
    (
        "fees",
        r"ثمن|رسوم|نخلص|تكلفة|درهم"
        r"|\b(?:frais|tarifs?|couts?|prix|fees?|costs?|price|dirhams?|dh"
        r"|nkhelles|nkhlles|nkhless|nkhalas)\b",
    ),
    ("required_documents", DOCUMENTS),
    ("eligibility", r"شروط|موهل|\b(?:eligib\w*|conditions?|qualify)\b"),
    ("procedure_steps", rf"كيفاش|مراحل|\betapes?\b|{REGISTRATION}"),
)
TOPIC_CUES = {
    "auto-entrepreneur": AUTO_ENTREPRENEUR,
    "passport": r"\bpasseport\b|\bpassport\b|جواز\s+السفر|باسبور",
    "university": r"\buniversit\w*|\bfacult[eé]\b|الجامع[ةي]|بالجامع[ةي]",
}


def topic_matches(question: str, source_topic: str) -> bool:
    """Conservative routing for the few named domains; not a topic classifier.

    Every explicitly recognized domain must be covered, otherwise the passage
    cannot answer the whole request. Unknown domains still need relevance checks.
    """
    searchable = _search_form(question)
    topics = [topic for topic, cue in TOPIC_CUES.items() if re.search(cue, searchable)]
    return all(source_topic == topic or source_topic.startswith(topic + "-") for topic in topics)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _search_form(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char) and char != "ـ"
    )


def process_query(text: str, response_language: LanguagePreference = "auto") -> ProcessedQuery:
    cleaned = normalize(text)
    searchable = _search_form(cleaned)
    aliases = [alias for pattern, alias in ALIASES if re.search(pattern, searchable)]
    intent = next((name for name, pattern in INTENTS if re.search(pattern, searchable)), "unknown")
    # A deposit deadline is not a card-delivery estimate. Keep that distinction
    # explicit even in the lexical development profile.
    if intent == "deadlines" and re.search(r"\b(?:carte|card)\b|بطاقة", searchable):
        if not re.search(r"depot|deposer|submit|ايداع|ندفع|دفع", searchable):
            intent = "card_delivery_deadline"
    return ProcessedQuery(
        original=text,
        normalized=cleaned,
        search_text=" ".join([cleaned, *aliases]),
        intent=intent,
        language=detect_language(cleaned, response_language),
    )


def lexical_tokens(text: str) -> list[str]:
    # Search-only normalization; never rewrite source excerpts or displayed questions.
    text = _search_form(text)
    stop = {
        "ما",
        "هي",
        "هو",
        "هل",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "do",
        "does",
        "quel",
        "quels",
        "quelle",
        "quelles",
        "comment",
        "chno",
        "chnou",
        "bach",
        "li",
        "khassni",
        "khasni",
        "de",
        "la",
        "le",
        "les",
        "des",
        "du",
        "un",
        "une",
        "et",
        "en",
        "pour",
        "the",
        "a",
        "is",
        "شنو",
        "لي",
        "باش",
        "من",
        "في",
        "على",
        "واش",
    }
    return [word for word in re.findall(r"[^\W_]+", text) if len(word) > 1 and word not in stop]
