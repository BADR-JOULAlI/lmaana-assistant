"""A thin bilingual text interface; retrieval and inference stay in the API."""

import html
import os

import httpx
import streamlit as st

from lmaana_assistant.contracts import LANGUAGE_POLICY_VERSION, SOURCE_POLICY_VERSION
from lmaana_assistant.language import LABELS
from lmaana_assistant.presentation import can_display_answer, paragraphs_html
from lmaana_assistant.source_policy import utc_today

API_URL = os.environ.get("LMAANA_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLE = "شنو الوثائق لي خاصني باش ندير auto-entrepreneur؟"
STATUS_LABELS = {
    "historical": "Référence historique",
    "unverified": "Actualité non confirmée",
    "verified": "Revue de fraîcheur valide",
    "expired": "Revue de fraîcheur expirée",
    "content_changed": "Contenu modifié depuis la revue",
    "not_yet_valid": "Date de revue invalide",
}
st.set_page_config(page_title="Lmaana Assistant", layout="wide")
st.html("""
<style>
body, .stApp, h1, h2, h3, p, input, textarea, button,
.answer-text, .scope, .source-heading, .question-summary {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}
.stApp {background: #FFFFFF; color: #151515;}
.block-container {max-width: 1150px; padding-top: 2.7rem; padding-bottom: 2rem;}
h1 {font-size: clamp(2.3rem, 5vw, 3.3rem) !important;
  letter-spacing: -0.055em; font-weight: 600 !important;}
h3 {font-size: 1.3rem !important;}
.intro {font-size: 1.45rem; line-height: 1.8; margin-bottom: .2rem;}
.answer-text {font-size: 1.08rem; line-height: 1.9; overflow-wrap: anywhere;}
.answer-text p {margin: 0 0 .7rem; unicode-bidi: plaintext;}
.answer-text p[dir="rtl"] {font-size: 1.26rem;}
.excerpt {border-left: 3px solid #002FA7; padding: .15rem 0 .25rem 1rem; margin: 0 0 1.2rem;}
.reference {font-size: .84rem; color: #002FA7; text-decoration: underline;}
.scope {background: #F7F7F8; border-top: 1px solid #D8D8DD;
  padding: .9rem 1rem; font-size: .92rem; line-height: 1.6;}
.scope strong {color: #002FA7;}
.source-heading {font-size: 1.05rem; font-weight: 600; line-height: 1.6;}
.source-heading bdi {color: #002FA7; margin-right: .5rem;}
.source-heading {scroll-margin-top: 5rem; overflow-wrap: anywhere;}
.question-summary {font-size: 1rem; color: #555; line-height: 1.8;}
div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 0 !important;}
button[kind="primaryFormSubmit"], button[kind="primary"] {
  background: #002FA7; border-color: #002FA7; color: #FFFFFF;
  border-radius: 0; min-height: 2.75rem;}
textarea {unicode-bidi: plaintext; font-size: 1.1rem !important;}
@media (max-width: 640px) {
  .block-container {padding: 2.2rem 1.1rem 1.5rem;}
  .intro {font-size: 1.25rem;}
  .excerpt {padding-left: .7rem;}
}
</style>
""")

st.title("Lmaana Assistant")
st.html('<p class="intro" dir="rtl">سول بالدارجة، وشوف شنو كاين فالمصادر.</p>')
st.caption("Recherche locale dans des documents marocains · Prototype texte")

try:
    health_response = httpx.get(f"{API_URL}/health/ready", timeout=5, trust_env=False)
    health = health_response.json()
    if not isinstance(health, dict):
        raise ValueError("Invalid health response")
except (httpx.HTTPError, ValueError):
    health = {"ready": False, "reason": "API inaccessible. Lancez lmaana serve."}

if health.get("ready") and (
    health.get("source_policy") != SOURCE_POLICY_VERSION
    or health.get("language_policy") != LANGUAGE_POLICY_VERSION
):
    health = {"ready": False, "reason": "Réindexez le corpus et redémarrez l’API mise à jour."}
cached_answer = st.session_state.get("answer")
if cached_answer and not can_display_answer(cached_answer, health, utc_today()):
    st.session_state.pop("answer", None)

is_excerpt = health.get("generator_backend") == "excerpt"
if not health.get("ready"):
    st.warning("Le service n’est pas prêt. Vérifiez l’API et les documents indexés.")
    with st.expander("Détails du service"):
        st.text(health.get("reason") or "Corpus indisponible.")
elif is_excerpt:
    count = health.get("source_count", 0)
    noun = "document indexé" if count == 1 else "documents indexés"
    st.html(
        '<div class="scope"><strong>Mode extraits</strong> · '
        f"{count} {noun}<br>Les passages sont cités dans leur langue d’origine. "
        "La langue est détectée, mais ce mode ne traduit pas les sources.</div>"
    )
    reviewed = health.get("verified_source_count", 0)
    st.caption(
        f"{reviewed} / {count} document(s) avec une revue de fraîcheur valide. "
        "Un document indexé n’est pas forcément une procédure actuelle."
    )
else:
    st.caption(
        "Génération locale · Langue détectée ou choisie · Réponses à vérifier avec les sources."
    )
    st.caption(
        "La qualité en darija et en Arabizi reste expérimentale. "
        "Un contrôle échoué bloque la réponse."
    )

if health.get("sources"):
    with st.expander("Quels documents puis-je interroger ?"):
        for source in health["sources"]:
            st.write(f"{source['title']} — {source['publisher']}")
            st.caption(
                STATUS_LABELS.get(source.get("verification_status"), "Actualité non confirmée")
            )
        st.caption("La présence d’un document ne garantit pas qu’il réponde à toute question.")

if st.button("Essayer la question sur l’auto-entrepreneur", disabled=not health.get("ready")):
    st.session_state.question = EXAMPLE

with st.form("question_form"):
    question = st.text_area(
        "السؤال / Votre question",
        key="question",
        height=105,
        max_chars=1500,
        placeholder=EXAMPLE,
    )
    response_language = st.selectbox(
        "Langue de réponse / لغة الجواب",
        options=["auto", "ary", "ary-Latn", "ar", "fr", "en"],
        format_func=lambda code: (
            "Automatique — selon la question" if code == "auto" else LABELS[code]
        ),
        help=(
            "La détection est une estimation. Vous pouvez corriger la langue, "
            "surtout pour une question courte ou mélangée."
        ),
    )
    submitted = st.form_submit_button(
        "سول / Rechercher", type="primary", disabled=not health.get("ready")
    )

if submitted:
    st.session_state.pop("answer", None)
    if not question.strip():
        st.warning("كتب السؤال أولا. Écrivez une question pour commencer.")
    else:
        with st.spinner("كنقلب فالمصادر… Recherche dans les documents…"):
            try:
                response = httpx.post(
                    f"{API_URL}/v1/answers",
                    json={"question": question, "response_language": response_language},
                    timeout=650,
                    trust_env=False,
                )
                if response.status_code == 200:
                    st.session_state.answer = response.json()
                else:
                    messages = {
                        429: "Une autre recherche est en cours. Réessayez dans un instant.",
                        503: "Le corpus ou le modèle est indisponible. Vérifiez l’API.",
                        504: "Le modèle a dépassé le délai. Vérifiez-le puis redémarrez l’API.",
                    }
                    st.error(messages.get(response.status_code, "La recherche a échoué."))
            except (httpx.HTTPError, ValueError):
                st.error("Impossible de joindre l’API. Vérifiez qu’elle est lancée.")

answer = st.session_state.get("answer")
if answer:
    st.divider()
    st.caption("Résultat pour")
    language = answer["query"]["language"]
    confidence = {"low": "faible", "medium": "moyenne", "high": "forte"}[language["confidence"]]
    st.caption(
        f"Langue détectée : {LABELS[language['detected']]} · certitude estimée : {confidence} · "
        f"Langue de réponse demandée : {LABELS[language['reply']]}"
        + (" · choix manuel" if language["method"] == "user_override" else "")
        + (" · question multilingue" if language["mixed"] else "")
    )
    st.html(
        '<div class="question-summary">' + paragraphs_html(answer["query"]["original"]) + "</div>"
    )
    labels = {citation["id"]: index for index, citation in enumerate(answer["citations"], 1)}
    titles = {
        "source_excerpts": "Extraits pertinents",
        "answered": "الجواب / Réponse",
        "insufficient_evidence": "Informations insuffisantes",
        "clarification_required": "Une précision est nécessaire",
        "verification_required": "Actualité non confirmée",
    }
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader(
            "Réponse non validée"
            if answer.get("generation_failed")
            else titles.get(answer["outcome"], "Réponse")
        )
        if answer["outcome"] == "verification_required":
            st.html(
                '<div class="scope"><strong>Vérification nécessaire</strong><br>'
                "Les références trouvées ne suffisent pas à confirmer la procédure actuelle. "
                "Leurs extraits ne sont pas présentés comme une réponse.</div>"
            )
        if answer["outcome"] == "source_excerpts":
            st.caption(
                "Texte original : la langue demandée n’est pas appliquée aux citations. "
                "Activez le modèle local pour obtenir une réponse reformulée."
            )
        for statement in answer["statements"]:
            references = " ".join(
                f'<a class="reference" href="#source-{labels[key]}" '
                f'aria-label="Voir la source {labels[key]}">Source [{labels[key]}]</a>'
                for key in statement["citation_ids"]
            )
            css = "answer-text excerpt" if statement["citation_ids"] else "answer-text"
            st.html(
                f'<div class="{css}">'
                + paragraphs_html(statement["text"], reflow=answer["outcome"] == "source_excerpts")
                + references
                + "</div>"
            )
        if answer["outcome"] == "insufficient_evidence" and not answer.get("generation_failed"):
            st.caption(
                "Aucun passage suffisamment pertinent pour cette question. "
                "Le corpus peut être incomplet ; cela ne signifie pas "
                "que l’information n’existe pas."
            )
        st.caption(f"Traitement API : {answer['timings_ms']['total']:.0f} ms")
    with right:
        related = answer.get("related_sources", [])
        st.subheader("المصادر / Sources" if answer["citations"] else "Références à vérifier")
        if not answer["citations"] and not related:
            st.write("Aucune source citée pour cette question.")
        elif answer["citations"]:
            st.caption(
                "Vérifiez l’actualité auprès de l’organisme. "
                "La date d’indexation ne prouve pas que la procédure est encore valable."
            )
        for index, citation in enumerate(answer["citations"], 1):
            with st.container(border=True):
                st.html(
                    f'<div class="source-heading" id="source-{index}"><bdi>[{index}]</bdi> '
                    f"{html.escape(citation['title'])}</div>"
                )
                st.caption(f"{citation['publisher']} · {citation['location']}")
                st.link_button("Consulter le document", citation["url"])
                st.caption(citation["freshness_note"])
                st.caption(
                    f"Revue du {citation['checked_on']} · à renouveler le {citation['review_due']}"
                )
                st.caption(f"Indexé le {citation['fetched_at'][:10]}")
                with st.expander("Voir le passage original complet"):
                    st.html(
                        '<div class="answer-text">'
                        + paragraphs_html(citation["excerpt"])
                        + "</div>"
                    )
        if related:
            st.caption("Ces documents ne sont pas utilisés comme preuves d’une réponse actuelle.")
        for reference in related:
            with st.container(border=True):
                st.html('<div class="source-heading">' + html.escape(reference["title"]) + "</div>")
                st.write(STATUS_LABELS[reference["verification_status"]])
                st.caption(f"{reference['publisher']} · {reference['location']}")
                st.caption(reference["verification_note"])
                st.caption(reference["freshness_note"])
                st.link_button("Consulter la référence", reference["url"])
                with st.expander("Lire l’archive — pas une réponse actuelle"):
                    st.caption("Texte d’origine, dont l’applicabilité actuelle n’est pas validée.")
                    st.html(
                        '<div class="answer-text">'
                        + paragraphs_html(reference["excerpt"])
                        + "</div>"
                    )
    with st.expander("Détails techniques"):
        st.json(
            {
                key: answer[key]
                for key in (
                    "request_id",
                    "corpus_release",
                    "embedding",
                    "generator",
                    "source_policy",
                    "language_policy",
                    "answer_language",
                    "generation_failed",
                    "timings_ms",
                    "warnings",
                )
            }
        )
else:
    st.caption("Posez une question pour afficher les extraits et leurs sources côte à côte.")

st.divider()
st.caption("Texte uniquement · L’entrée vocale avec Lmaana 2.4 n’est pas encore disponible.")
