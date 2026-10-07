# Source review and answer policy

The prototype must not turn a retrieved historical paragraph into a current
administrative instruction. Citation validity and factual/current applicability
are different properties. Software tests do not certify administrative accuracy.

## Default behavior

All sources start as `unverified`. `reviewed: true` only means the operator inspected
the source for ingestion; it does **not** certify that its requirements are current.

The API applies `reviewed-currency-v1` before either excerpt or neural generation:

- Only sources with a complete, unexpired `verified` review enter answer context.
- Historical, unverified, future-dated, changed, and expired sources are withheld.
- If relevant withheld references are found, the API returns `verification_required`
  with a short Darija explanation and no factual answer or answer citations.
- Withheld references appear separately in `related_sources`. Their original text
  is available under an explicitly labelled archive expander, not as the answer.
- If there is no sufficiently relevant passage, the result is `insufficient_evidence`.
- Checking dates happens on each request, so an approval expires without a restart.

The gate limits which evidence can support an answer. It does not prove that a
generated claim is semantically entailed, detect every contradiction, or guarantee
that an official procedure has not changed since review. The lexical relevance
heuristics and source-level reviews are not claim-level verification.

## Evidence required for a freshness review

To set `verification.status` to `verified`, an operator must provide:

1. `checked_on`: the date of the actual applicability check, not just a download.
2. `review_due`: the exclusive renewal date, 1–90 days after the check. This is an
   internal risk-control window, not a legal validity period.
3. `evidence_urls`: the primary evidence used to establish present applicability.
4. `note`: what was checked, for which scope, and any remaining limitations.
5. `content_sha256`: SHA-256 of the exact downloaded/local source bytes reviewed.

An approval is an operator attestation; the schema cannot establish whether the
operator's evidence is correct. Do not mark sources verified just to get an answer.
For a changing HTML page, re-download and inspect the new snapshot before issuing
a new approval. A mismatched hash makes ingestion fail without replacing the active
corpus. Retrieval checks the approval against the stored content hash as well.

Then re-ingest and restart the API (and Streamlit after a code upgrade).
Never hand-edit Qdrant payloads. Corpus schema 2
records the source policy; a schema-1 index is rejected until re-ingested. Source
metadata, verification records, and selected HTML sections are versioned with the corpus.

## Review log — 2026-10-03

| Source | Finding | Status in the prototype |
| --- | --- | --- |
| [DGI auto-entrepreneur guide, edition 2026](https://www.tax.gov.ma/wps/wcm/connect/631b0232-3f9d-4d12-a8c9-053ccc921bce/Guide__AE%2B_FR%2B_2026.pdf?CACHEID=ROOTWORKSPACE-631b0232-3f9d-4d12-a8c9-053ccc921bce-q0pxu4M&MOD=AJPERES) | Downloaded from the official DGI host; edition stated on the cover. Registration section checked visually on PDF pages 11–12 (printed pages 10–11). The guide's final page says it does not replace legislation/regulations. | `verified`, narrowly for retrieval of the registration section; review expires 2026-11-02. |
| [Maroc PME brochure, page 2](https://marocpme.gov.ma/wp-content/uploads/2020/05/AE-D%C3%A9pliant-FINAL-1.pdf) | URL contains a 2020 upload path. The French paragraph specifies partner banks; its Arabic counterpart omits that qualifier. Present applicability was not established. | `historical`; withheld from answer generation. |
| [CRI Casablanca-Settat procedure page](https://www.casainvest.ma/ar/الإجراءات) | The selected auto-entrepreneur section mentions banks partnered with Poste Maroc, but gives no detailed document checklist or section update date. | `unverified`; consultation is not a current-applicability approval. |
| [RNAE registration portal](https://rn.ae.gov.ma/registration) | Could not be retrieved during the verification attempt. This does not establish that it is unavailable to all users. | Not ingested; no current requirements inferred from the failed request. |

Only the exact CRI section `إجراءات الحصول على صفة المقاول الذاتي` is ingested.
Other company forms on the page must not contaminate this small auto-entrepreneur corpus.
The corpus has **one scope-limited approved source out of three** during its review
window. This is an evidence review, not a legal certification or a live check of bank
operations. The source note explicitly excludes eligibility, taxation and card-delivery
time from this approval. Only the selected registration section is indexed.

### DGI PDF extraction review

The registration sentence crosses a page boundary. Its continuation contains the
partner-bank qualifier, supporting documents, foreign-resident alternative, and
deposit deadline. Keeping only either page can misrepresent it. The new opt-in
`pdf_section` extraction removes only the exact reviewed leading header/folio,
joins consecutive page bodies with a newline, selects unique reviewed markers,
and keeps the section in one chunk. Layout drift, marker drift or oversize sections
fail ingestion. It does not use OCR, translate or rewrite the body.

Reviewed raw PDF SHA-256:
`7fa82a063e2f74001ecd03f4c2803e86295b35b514d4eef4f8adaee695bab9cd`.
The 30-day renewal window is an internal precaution, not a validity period stated
by the DGI. Confirm practical requirements with the official service before filing.

## Checks

Offline tests cover incomplete reviews, expiry boundaries, future checks, changed
content, source deduplication, verified/unverified mixtures, and prevention of any
unverified passage reaching a generator. They also cover cross-page extraction,
folio/header drift, missing/duplicate/reversed markers, oversized atomic sections,
index-version mismatch, and deposit-versus-card-delivery intent cues.
Live checks should confirm document questions cite only the DGI section during its
review window; unsupported fees, eligibility and card-delivery questions must abstain.
The 2026-10-03 run passed 105 offline software tests and 14 live API scenarios.
The live scenarios include four document-question formulations, a deposit-deadline
question, and unsupported fees, card delivery, eligibility, passport and university
questions. Source citations, page ranges and retained qualifiers are checked alongside
outcomes. Browser checks cover desktop/mobile layouts and stale-citation removal.
These are targeted regressions, not a representative retrieval/faithfulness benchmark.
Known-domain routing currently recognizes auto-entrepreneur, passport and university
queries only; it must not be described as general topic understanding.

Before enabling factual Darija generation, add claim-support evaluation and test
the actual Qwen runtime. Source approval and verbatim excerpts do not validate a
translation or synthesis. Lmaana 2.4 remains the ASR target; no ASR model is active yet.
