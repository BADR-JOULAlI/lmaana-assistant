"""Query-focused, verbatim excerpts; these heuristics are not answer synthesis."""

import re
import unicodedata

from lmaana_assistant.contracts import Evidence, GeneratedAnswer, ProcessedQuery, Statement
from lmaana_assistant.normalization import lexical_tokens

ABSTENTION = "ما لقيتش معلومات كافية فالمصادر المتوفرة باش نجاوبك. وضّح السؤال أو زيد مصدر مناسب."

# Conservative relevance cues, not a claim that a passage answers the whole question.
INTENT_CUES = {
    "required_documents": (
        r"\b(documents?|pieces?|dossier\w*|formulaires?|copies?|fournir|joindre|cnie?)\b"
        r"|وثائق|استمار|نسخة|ملف|صورة"
    ),
    "fees": r"\b(frais|tarifs?|cout\w*|prix|gratuit\w*|dirhams?|dh)\b|رسوم|ثمن|درهم|مجاني|تكلف",
    "deadlines": r"\b(delai\w*|duree|jours?|semaines?|mois|heures?)\b|مدة|أجل|ايام|أيام|شهر|ساعة",
    "card_delivery_deadline": (
        r"\b(?:delivr\w*|recev\w*|reception|obtention|remise)\b.{0,80}\bcarte\b"
        r"|\bcarte\b.{0,80}\b(?:delivr\w*|recue|prete|disponible|remise)\b"
        r"|(?:تسليم|استلام|الحصول على|توصل).{0,40}بطاقة"
        r"|بطاقة.{0,40}(?:تسليم|استلام|توصلك|جاهزة)"
    ),
    "eligibility": r"\b(conditions?|eligible\w*|eligibilite|autorise\w*|exclu\w*)\b|شروط|مؤهل|يحق",
}


def search_form(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char) and char != "ـ"
    )


def script_direction(text: str) -> str:
    """Use the first strong character, like HTML dir=auto (ignore digits/punctuation)."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in {"R", "AL"}:
            return "rtl"
        if direction == "L":
            return "ltr"
    return ""


def passage_blocks(text: str) -> list[str]:
    """Keep contiguous source spans, soft line wraps, and adjacent qualifications.

    Blank lines, language switches and sentence-ending PDF lines give candidate
    boundaries. Isolated step numbers are excluded from the selected spans.
    No spelling repair, translation or arbitrary character truncation is performed.
    """
    blocks: list[str] = []
    start = None
    end = 0
    direction = ""
    previous = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = line.strip()
        line_direction = script_direction(clean)
        qualifier = re.match(
            r"^(sauf|cependant|toutefois|mais|except|however|unless)\b|^(إلا|لكن|باستثناء)",
            clean,
            re.I,
        )
        # Only an isolated, one-digit marker after a completed sentence is a
        # likely step label. Keep numeric values in lists (e.g. a price on its own line).
        marker = clean.isdecimal() and len(clean) == 1 and re.search(r"[.!?؟]$", previous)
        boundary = (
            not clean
            or marker
            or (
                start is not None
                and line_direction
                and (
                    (direction and direction != line_direction)
                    or (re.search(r"[.!?؟][\d٠-٩]*$", previous) and not qualifier)
                    or re.search(r"[^\W\d_][\d٠-٩]{1,2}$", previous)
                )
            )
        )
        if boundary and start is not None:
            block = text[start:end].strip()
            if any(char.isalpha() for char in block):
                blocks.append(block)
            start, direction = None, ""
        if clean and not marker:
            if start is None:
                start = offset
            end = offset + len(line)
            direction = line_direction or direction
            previous = clean
        offset += len(line)
    if start is not None:
        block = text[start:end].strip()
        if any(char.isalpha() for char in block):
            blocks.append(block)
    return blocks


def focused_excerpts(query: ProcessedQuery, evidence: list[Evidence]) -> GeneratedAnswer:
    query_terms = set(lexical_tokens(search_form(query.search_text)))
    cue = search_form(INTENT_CUES.get(query.intent, ""))
    ranked = []
    for item_index, item in enumerate(evidence):
        for block_index, block in enumerate(passage_blocks(item.chunk.text)):
            # Never truncate a condition/list to make it fit the response contract.
            if len(block) > 2800:
                continue
            searchable = search_form(block)
            if query.intent == "card_delivery_deadline":
                # Match event cues across PDF soft wraps, but also require a
                # duration cue. Neither a card mention nor a deposit deadline suffices.
                searchable = " ".join(searchable.split())
                if not re.search(search_form(INTENT_CUES["deadlines"]), searchable):
                    continue
            terms = set(lexical_tokens(searchable))
            overlap = len(terms & query_terms)
            matches = set(re.findall(cue, searchable)) if cue else set()
            if (cue and not matches) or (not cue and not overlap):
                continue
            rank = 3 * len(matches) + overlap + item.score
            ranked.append((rank, item_index, block_index, block, item.chunk.id))
    selected, seen = [], set()
    for candidate in sorted(ranked, key=lambda row: (-row[0], row[1], row[2])):
        key = " ".join(candidate[3].split())
        if key not in seen:
            selected.append(candidate)
            seen.add(key)
        if len(selected) == 2:
            break
    if not selected:
        return GeneratedAnswer(
            outcome="insufficient_evidence", statements=[Statement(text=ABSTENTION)]
        )
    # Preserve source order once relevance has selected the two best passages.
    selected.sort(key=lambda row: (row[1], row[2]))
    return GeneratedAnswer(
        outcome="source_excerpts",
        statements=[Statement(text=row[3], citation_ids=[row[4]]) for row in selected],
    )
