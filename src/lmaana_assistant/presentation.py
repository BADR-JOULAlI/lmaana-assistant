"""Safe, model-independent formatting for the bilingual interface."""

import html
import re
from datetime import date

from lmaana_assistant.contracts import LANGUAGE_POLICY_VERSION, SOURCE_POLICY_VERSION
from lmaana_assistant.generation.excerpts import script_direction


def can_display_answer(answer: dict, health: dict, today: date) -> bool:
    """Invalidate old sessions after an upgrade, corpus change, outage or review expiry."""
    if not (
        health.get("ready")
        and answer.get("source_policy") == health.get("source_policy") == SOURCE_POLICY_VERSION
        and answer.get("corpus_release") == health.get("corpus_release")
        and answer.get("language_policy")
        == health.get("language_policy")
        == LANGUAGE_POLICY_VERSION
        and answer.get("generator") == health.get("generator")
    ):
        return False
    try:
        return all(
            citation["verification_status"] == "verified"
            and date.fromisoformat(citation["checked_on"])
            <= today
            < date.fromisoformat(citation["review_due"])
            for citation in answer["citations"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def paragraphs_html(text: str, *, reflow: bool = False) -> str:
    """Escape untrusted content and isolate each paragraph's reading direction.

    Reflow only display whitespace; the API's verbatim source text stays intact.
    Single line breaks in generated answers/lists are preserved.
    """
    result = []
    for part in re.split(r"\n\s*\n", text):
        if not part.strip():
            continue
        if reflow:
            # A wrapped hyphenated word keeps the original hyphen, but no extra space.
            part = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "-", part)
            part = " ".join(part.split())
        direction = script_direction(part) or "ltr"
        escaped = html.escape(part).replace("\n", "<br>")
        result.append(f'<p dir="{direction}">{escaped}</p>')
    return "".join(result)
