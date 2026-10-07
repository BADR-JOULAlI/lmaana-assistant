"""Auditable language routing, not a learned detector or a semantic-quality score.

Short, ambiguous and code-switched text can be misclassified. Always expose the
decision and allow an explicit correction; never treat script as dialect proof.
"""

import re
import unicodedata

from lmaana_assistant.contracts import LanguageDecision, LanguagePreference, ReplyLanguage

LABELS = {
    "fr": "Français",
    "ar": "العربية الفصحى",
    "ary": "الدارجة",
    "ary-Latn": "Darija (latin / Arabizi)",
    "en": "English",
    "und": "Indéterminée",
}
INSTRUCTIONS = {
    "fr": "Write the answer in French, not Arabic or Darija.",
    "ar": "اكتب الجواب بالعربية الفصحى المبسطة، وليس بالدارجة أو الفرنسية.",
    "ary": (
        "Write in Moroccan Darija using Arabic letters, not Modern Standard Arabic. "
        "جاوب بالدارجة المغربية بحروف عربية. استعمل كلام مغربي طبيعي بحال خاصك، ديالك، باش. "
        "ما تجاوبش بالفصحى ولا بالفرنسية. خلي أسماء المؤسسات والمصطلحات الفرنسية المفيدة كيف ما هي."
    ),
    "ary-Latn": (
        "Write in Moroccan Darija using Latin letters and Arabizi digits only. "
        "Jawb b darija maghribiya b l7orof latin / Arabizi (b7al: khassk, dyalk, bach). "
        "No Arabic script, no French-only answer. Keep useful French administrative terms."
    ),
    "en": "Write the answer in English, not French or Arabic.",
}
_DAR_AR = set(
    (
        "شنو اشنو أشنو كيفاش واش بغيت بغيتي خاصني خاصك خاصو خصني خصك ديال ديالي ديالك باش ندير "
        "نسجل شحال فين فوقاش عافاك كاين كاينة كاينين هاد هادي هادا ليك ليا تاخد ناخد "
        "الوراق لوراق وثايق الوثايق"
    ).split()
)
_DAR_LAT = set(
    (
        "chno chnou shno chnu ach wach kifach bghit bghiti khassni khasni khassk khassak khass "
        "khassek dyal dial dyali dyalk bach ndir nsjel nsajel ch7al chhal wa9tach fin 3afak afak "
        "kayn kayna had hadi hada lma9awil wra9 lwra9 wraqi lwrak"
    ).split()
)
_FR = set(
    (
        "je tu il elle nous vous quels quelles quel quelle comment pourquoi combien dois besoin "
        "pour avec une des les est sont documents pieces inscription inscrire projet documentation "
        "dossier bonjour merci peux pouvez voudrais reponds repondre francais delai frais "
        "conditions "
        "obtenir carte demande envoyer copie faut"
    ).split()
)
_EN = set(
    (
        "what which how why when where do does are is the this that need required requirements "
        "documents for with please answer english hello thanks project register registration fees "
        "deadline card can i get"
    ).split()
)


def _form(text: str) -> str:
    # Preserve hamza: stripping it turns formal "وثائق" into dialectal "وثايق",
    # which is not evidence that the user wrote Darija.
    return unicodedata.normalize(
        "NFC",
        "".join(
            c
            for c in unicodedata.normalize("NFD", text.casefold())
            if (not unicodedata.combining(c) or c in "\u0654\u0655") and c != "ـ"
        ),
    )


def detect_language(text: str, preference: LanguagePreference = "auto") -> LanguageDecision:
    # Product names and URLs do not establish the question's language.
    clean = re.sub(r"https?://\S+|auto[ -]?entrepreneur|\b(?:cnie?|rnae)\b", " ", _form(text))
    tokens = set(re.findall(r"[^\W_]+", clean))
    da, dl = len(tokens & _DAR_AR), len(tokens & _DAR_LAT)
    fr, en = len(tokens & _FR), len(tokens & _EN)
    arabic = len(re.findall(r"[\u0600-\u06ff]", clean))
    latin = len(re.findall(r"[a-z]", clean))
    detected: ReplyLanguage = "und"
    confidence = "low"
    if da and (fr < 3 or da >= 2):
        detected, confidence = "ary", "high" if da >= 2 else "medium"
    elif dl and (fr < 3 or dl >= 2):
        detected, confidence = "ary-Latn", "high" if dl >= 2 else "medium"
    elif arabic >= 2 and (fr + en < 3 or arabic > latin):
        detected, confidence = "ar", "medium" if arabic >= 12 else "low"
    elif fr > en and fr >= 1:
        detected, confidence = "fr", "high" if fr >= 3 else "medium"
    elif en > fr and en >= 1:
        detected, confidence = "en", "high" if en >= 3 else "medium"
    mixed = bool((arabic >= 2 and (fr or en or dl)) or (dl and fr) or (fr >= 2 and en >= 2))
    return LanguageDecision(
        detected=detected,
        reply=detected if preference == "auto" else preference,
        confidence=confidence,
        mixed=mixed,
        method="rules" if preference == "auto" else "user_override",
    )


MESSAGES = {
    "generation_failed": {
        "fr": "La réponse générée n’a pas passé les contrôles. Elle n’est pas affichée. "
        "Reformulez la question ou choisissez une autre langue.",
        "ar": "لم تجتز الإجابة المولّدة فحوص التحقق، لذلك لم أعرضها. "
        "يرجى إعادة صياغة السؤال أو اختيار لغة أخرى.",
        "ary": "الجواب لي خرج الموديل ما دازش المراقبة، وما غاديش نعرضو ليك. "
        "عاود صيغ السؤال ولا اختار لغة أخرى.",
        "ary-Latn": "Ljawab li khrej lmodel ma dazch lmora9aba, ma ghadi nchowrohch. "
        "3awed so2al dyalk wlla khtar logha khra.",
        "en": "The generated answer did not pass validation and was withheld. "
        "Please rephrase the question or choose another language.",
    },
    "insufficient_evidence": {
        "fr": (
            "Je n’ai pas trouvé assez d’informations dans les sources disponibles pour répondre "
            "à cette question. Précisez la demande ou ajoutez une source pertinente."
        ),
        "ar": (
            "لم أجد معلومات كافية في المصادر المتاحة للإجابة عن هذا السؤال. "
            "يرجى توضيح الطلب أو إضافة مصدر مناسب."
        ),
        "ary": "ما لقيتش معلومات كافية فالمصادر المتوفرة باش نجاوبك. وضّح السؤال أو زيد مصدر مناسب.",
        "ary-Latn": (
            "Ma l9itch ma3loumat kafya f lmasadir bach njawbk. "
            "Wdde7 so2al dyalk wlla zid chi masdar monasib."
        ),
        "en": (
            "I could not find enough information in the available sources to answer. "
            "Please clarify the question or add a relevant source."
        ),
    },
    "verification_required": {
        "fr": (
            "J’ai trouvé des références sur ce sujet, mais leur actualité n’est pas confirmée. "
            "Je ne peux pas en tirer une réponse certaine. Vérifiez auprès de l’organisme officiel."
        ),
        "ar": (
            "وجدت مراجع حول هذا الموضوع، لكن لم يتم التأكد من أن معلوماتها ما زالت سارية. "
            "لا يمكنني تقديم جواب مؤكد بناء عليها. يرجى التحقق لدى الجهة الرسمية."
        ),
        "ary": (
            "لقيت مراجع على هاد الموضوع، ولكن ما تأكدتش واش المعلومات ديالها ما زالت صالحة دابا. "
            "ما نقدرش نعطيك جواب مؤكد اعتماداً عليها. "
            "تأكد من الجهة الرسمية قبل ما تعتمد عليها."
        ),
        "ary-Latn": (
            "L9it maraji3 3la had lmawdo3, walakin ma t2kkedtch wach ba9i sal7in daba. "
            "Ma n9drch n3tik jawab mo2akkad. T2kked m3a ljiha rasmia."
        ),
        "en": (
            "I found references on this topic, but their current applicability is unconfirmed. "
            "I cannot give a definite answer from them. "
            "Please check with the official organization."
        ),
    },
}
LANGUAGE_CLARIFICATION = (
    "Quelle langue préférez-vous : darija, arabe, français ou anglais ? / بأي لغة بغيتي الجواب؟"
)


def message(outcome: str, language: ReplyLanguage) -> str:
    return MESSAGES[outcome].get(language, LANGUAGE_CLARIFICATION)


def output_matches_language(text: str, language: ReplyLanguage) -> bool:
    """Coarse script/lexical guard; not a language-quality or dialect-certification metric."""
    clean = re.sub(r"https?://\S+", "", text)
    # Reject stray Cyrillic/CJK/etc. tokens observed in local generation. This
    # does not assess fluency and may reject legitimate names in other scripts.
    if any(
        char.isalpha()
        and not any(script in unicodedata.name(char, "") for script in ("LATIN", "ARABIC"))
        for char in clean
    ):
        return False
    decision = detect_language(clean)
    ar = len(re.findall(r"[\u0600-\u06ff]", clean))
    lat = len(re.findall(r"[a-zA-Z]", clean))
    if language in {"ar", "ary"}:
        if ar < 3 or ar < lat / 2:
            return False
        return decision.detected != "ary" if language == "ar" else decision.detected == "ary"
    if language in {"fr", "en", "ary-Latn"}:
        if lat < 3 or ar > max(2, lat / 10):
            return False
        return decision.detected == language
    return False
