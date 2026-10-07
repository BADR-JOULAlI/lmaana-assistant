"""Conservative checks for observed translation failures, not an entailment model.

The initial French registration corpus has explicit conditions that a document
checklist must not lose. Activate each check from source text, never from a fixed
administrative answer. Unknown paraphrases can be rejected; passing is not proof
that all claims are true. Wider corpora need evaluated multilingual entailment.
"""

import re
import unicodedata

from lmaana_assistant.contracts import Evidence, GeneratedAnswer, ProcessedQuery
from lmaana_assistant.errors import InvalidGeneration


def _form(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    # List labels are formatting, not factual quantities (including Arabic digits).
    text = re.sub(r"(^|[\n:;,،])\s*\d+[.)]\s+", r"\1 ", text)
    return " ".join(
        "".join(
            str(unicodedata.digit(char)) if char.isdecimal() else char
            for char in unicodedata.normalize("NFD", text.casefold())
            if not unicodedata.combining(char) and char != "ـ"
        ).split()
    )


_PASSPORT = r"\b(?:passports?|passeports?|passport|baspor)\b|جواز|باسبور"
_CONDITIONS = (
    (
        r"adresse electronique",
        r"\b(?:e-?mail|courriel|adresse electronique)\b|بريد\s+الكتروني|عنوان\s+الكتروني|ايميل",
    ),
    (r"\bsigner\b", r"\bsign\w*|\bmsign\w*|وقع|وقيع"),
    (r"photo personnelle", r"\bphoto\w*|\bpicture\w*|\btswira\b|صورة|تصويرة"),
    (r"copie de la carte nationale", r"\bcop(?:ie|y)\b|\bcopy\b|نسخ"),
    (r"carte nationale d.identite", r"\b(?:identity|id|cin|cnie)\b|identit|تعريف|هوية"),
    (r"carte de sejour", r"sejour|residen\w*|اقامة"),
    (r"\betrangers\b", r"etranger\w*|foreign\w*|ajanib|ajani|اجنب|اجانب"),
    (r"banques partenaires", r"partenair\w*|partner\w*|chouraka|shoraka|شريك|شرك|شراكة"),
    (r"barid al\s*-?\s*maghrib", r"barid\s+al\s*-?\s*maghrib|بريد\s+المغرب"),
)

_ONLINE_APPLICATION = r"remplir.{0,80}demande.{0,160}electronique.{0,100}portail"
_FILL = r"\b(?:rempli\w*|fill\w*|complet\w*|t?kat3mmer|t?3mmer|ta3mir)\b|عم.?ر|تعب.ة|ملء"
_ONLINE = r"\b(?:electroni\w*|online|internet)\b|الكتروني|لانترنت|انترنت"
_SUBMIT = (
    r"\b(?:submit\w*|deposit\w*|depos\w*|depot|tdeff\w*|tdef\w*|t9ddem\w*)\b"
    r"|دفع|ديع|تقديم|قدم"
)
_FROM_EVENT = (
    r"\b(?:from|after|following|a compter|a partir|depuis|men nhar|mn nhar)\b"
    r"|من\s+(?:نهار|تاريخ|يوم)|ابتداء|بعد"
)
_REQUEST = r"\b(?:application|request|demande|talab)\b|طلب"
_CREATION = (
    r"\b(?:creat\w*|establish\w*|register\w*|etabli\w*|tsjjel\w*|tsejjel\w*)\b|نشا|عداد|تسجيل|تسجل"
)
_SOURCE_REQUEST_CREATION = r"creat\w*.{0,80}demande|demande.{0,120}(?:etabli\w*|creat\w*)"


def validate_source_constraints(
    answer: GeneratedAnswer, query: ProcessedQuery, evidence: list[Evidence]
) -> None:
    if answer.outcome != "answered":
        return
    by_id = {item.chunk.id: item.chunk.text for item in evidence}
    for statement in answer.statements:
        source = _form(" ".join(by_id.get(key, "") for key in statement.citation_ids))
        text = _form(statement.text)
        if re.search(_PASSPORT, text) and not re.search(_PASSPORT, source):
            raise InvalidGeneration("The response introduces an unsupported identity document.")
        if not set(re.findall(r"\b\d+\b", text)).issubset(set(re.findall(r"\b\d+\b", source))):
            raise InvalidGeneration("The response introduces a number absent from its citations.")
    if query.intent != "required_documents":
        return
    cited = {key for statement in answer.statements for key in statement.citation_ids}
    source = _form(" ".join(by_id[key] for key in cited))
    text = _form(" ".join(statement.text for statement in answer.statements))
    for source_cue, answer_cue in _CONDITIONS:
        if re.search(source_cue, source) and not re.search(answer_cue, text):
            raise InvalidGeneration("A source condition was lost during checklist generation.")
    if re.search(_ONLINE_APPLICATION, source) and not (
        re.search(_FILL, text) and re.search(_ONLINE, text)
    ):
        raise InvalidGeneration("The checklist omitted the source's online application.")
    for days in re.findall(r"\b(\d+)\s+jours?\b", source):
        if not re.search(rf"(?<!\d){days}(?!\d)", text):
            raise InvalidGeneration("The checklist omitted the source's submission deadline.")
        # The initial reviewed section specifies a submission deadline starting
        # from application creation. A number alone must not stand in for that
        # event. These cues remain conservative, not a general entailment check.
        if re.search(r"depos\w*.*?\b" + days + r"\s+jours?.*?a compter", source):
            deadline_statements = [
                _form(statement.text)
                for statement in answer.statements
                if re.search(rf"(?<!\d){days}(?!\d)", _form(statement.text))
            ]
            if not any(
                re.search(_SUBMIT, line) and re.search(_FROM_EVENT, line)
                for line in deadline_statements
            ):
                raise InvalidGeneration("The checklist lost the submission deadline's start event.")
            if re.search(_SOURCE_REQUEST_CREATION, source):
                # A starting-event word alone could attach the deadline to card
                # receipt. Preserve the stated application-creation event too.
                def retains_creation(line: str) -> bool:
                    start = re.search(_FROM_EVENT, line)
                    return bool(
                        start
                        and re.search(_SUBMIT, line)
                        and re.search(_REQUEST, line)
                        and re.search(_CREATION, line[start.end() :])
                    )

                if not any(retains_creation(line) for line in deadline_statements):
                    raise InvalidGeneration("The submission deadline changed its starting event.")
