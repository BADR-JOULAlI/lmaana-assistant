"""Conservative query processing; the user's original wording remains available."""

import re
import unicodedata

from lmaana_assistant.contracts import LanguagePreference, ProcessedQuery
from lmaana_assistant.language import detect_language

ALIASES = (
    (
        r"auto[\s-]?entrepreneur|(?:المقاول|للمقاول|مقاول)\s+(?:ال)?ذاتي",
        "auto entrepreneur المقاول الذاتي",
    ),
    (
        r"وثائق|وثايق|الوراق|لوراق|documents|pièces|papiers|\b(?:lwra9|wra9|lwrak|wathai9|papers)\b",
        "documents pièces dossier formulaire demande copie الوثائق",
    ),
    (
        r"نسجل|التسجيل|ندير|inscription|\b(?:register|registration|nsjel|nsajel|ndir)\b",
        "inscription التسجيل",
    ),
    (r"البطاقة|بطاقة|cnie|cin", "CNIE identité بطاقة"),
)
INTENTS = (
    (
        "deadlines",
        r"أجل|مدة|وقت|délai|date limite|deadline|how long|شحال.{0,20}(?:نهار|يوم|شهر|ساعة)",
    ),
    ("fees", r"شحال|ثمن|رسوم|frais|tarif|\b(?:fees?|cost|ch7al)\b"),
    (
        "required_documents",
        r"وثائق|وثايق|الوراق|لوراق|documents|pièces|papiers|\b(?:lwra9|wra9|lwrak|wathai9|papers)\b",
    ),
    ("eligibility", r"شروط|مؤهل|éligib|conditions"),
    ("procedure_steps", r"كيفاش|مراحل|étapes|inscription"),
)
TOPIC_CUES = {
    "auto-entrepreneur": ALIASES[0][0],
    "passport": r"\bpasseport\b|\bpassport\b|جواز\s+السفر|باسبور",
    "university": r"\buniversit\w*|\bfacult[eé]\b|الجامع[ةي]|بالجامع[ةي]",
}


def topic_matches(question: str, source_topic: str) -> bool:
    """Conservative routing for the few named domains; not a topic classifier.

    Every explicitly recognized domain must be covered, otherwise the passage
    cannot answer the whole request. Unknown domains still need relevance checks.
    """
    topics = [topic for topic, cue in TOPIC_CUES.items() if re.search(cue, question, re.I)]
    return all(source_topic == topic or source_topic.startswith(topic + "-") for topic in topics)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def process_query(text: str, response_language: LanguagePreference = "auto") -> ProcessedQuery:
    cleaned = normalize(text)
    aliases = [alias for pattern, alias in ALIASES if re.search(pattern, cleaned, re.I)]
    intent = next(
        (name for name, pattern in INTENTS if re.search(pattern, cleaned, re.I)), "unknown"
    )
    # A deposit deadline is not a card-delivery estimate. Keep that distinction
    # explicit even in the lexical development profile.
    if intent == "deadlines" and re.search(r"\b(?:carte|card)\b|بطاقة|البطاقة", cleaned, re.I):
        if not re.search(r"dép[oô]t|déposer|submit|إيداع|ايداع|ندفع|دفع", cleaned, re.I):
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
    text = "".join(
        c
        for c in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(c) and c != "ـ"
    )
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
