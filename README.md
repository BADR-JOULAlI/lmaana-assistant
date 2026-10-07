# Lmaana Assistant

A Moroccan Darija voice RAG project, starting with a runnable text-retrieval prototype.

**Primary ASR target: [Lmaana/lmaana-2.4](https://huggingface.co/Lmaana/lmaana-2.4).**
Whisper-large-v3-turbo is the planned comparison baseline. See the
[Lmaana 2.4 integration notes](docs/lmaana-2.4-integration.md).

## What works now

- Reviewed-source ingestion from HTML, PDF, or local UTF-8 text, with page/section provenance.
- Persistent Qdrant collections and atomic activation of versioned corpus manifests.
- Conservative Darija normalization, terminology aliases, and basic intent routing.
- FastAPI health endpoints and `POST /v1/answers` with citations and stage timings.
- An explicit lexical/excerpt profile that runs without downloading ML models.
- Query-focused, verbatim snippets instead of entire PDF pages, with conservative
  intent cues for documents, fees, deadlines, and eligibility. No matching cue means
  abstention; these heuristics are not a semantic answerability guarantee.
- A source-currency gate shared by all generators: incomplete, historical, changed,
  or expired reviews cannot support an affirmative answer.
- Qwen embedding and llama.cpp/Ollama generation adapters, with model/index compatibility checks,
  prompt token budgeting, citation validation, one retry, and timeout handling.
- A French/Darija Streamlit interface with separate reading directions, reflowed PDF
  lines, clickable citations, full original passages, and a visible corpus inventory.
- Language routing for French, standard Arabic, Darija in Arabic script, Latin Darija
  (Arabizi) and English, including explicit reply-language override and localized abstentions.

Ollama 0.35.1 has run real Qwen3-4B inference locally on the RTX 5070 Laptop GPU.
The Qwen embedding and standalone llama.cpp adapters still have only contract tests here.
Successful inference is not proof of language quality or factual correctness: initial
Arabic/Darija/Arabizi trials exposed mistranslations and incomplete answers.
Speech transcription, TTS, and model quality benchmarks are not implemented yet.
Excerpt mode returns `source_excerpts`, not an LLM-generated answer.
Each excerpt must be an exact substring of its cited passage. Display-only whitespace
reflow does not modify the source or API text. At most two complete passages are shown;
this selection can omit context, so consult the complete source before relying on it.

### Language behavior

The question language and the source language are separate. `POST /v1/answers`
accepts optional `response_language`: `auto` (default), `fr`, `ar`, `ary`, `ary-Latn`,
or `en`. `query.language` exposes the detected/selected language, a qualitative
confidence level, code-switching flag, decision method and policy version.

- **Detection:** lightweight, auditable script/word rules, not a learned language-ID
  model or proof of semantic understanding. Short or unfamiliar inputs can be ambiguous.
  Unknown language produces a language-choice clarification. The UI provides an override.
- **Neural profile:** the selected language is passed into the system prompt. Obvious
  script/language mismatches are rejected, with one retry. The guard is heuristic and
  cannot certify dialect fluency, translation fidelity or factual correctness.
- **Excerpt profile:** original source text is never translated or labelled as a
  same-language answer. `answer_language="source"` and the UI explicitly say so.
- **No evidence / unverified sources:** deterministic messages follow the reply language
  even without a language model. Names, source URLs and original citations stay intact.

`language-routing-v2` invalidates older UI answers, as does a generator-profile change.
The API's language metadata distinguishes recognition from successful generation.
`generation_failed=true` distinguishes a rejected model output from missing evidence.
The UI labels it "Réponse non validée"; it must not be mistaken for an expired source review.
Targeted guards reject stray writing systems, unsupported passport substitutions/numbers,
and omission of key conditions from document checklists in the initial French corpus.
These are conservative regression checks, not a general semantic verifier: valid
paraphrases may be withheld and other factual errors can still pass. Check the original source.
The current neural validation profile, `generation-validation-v2`, also checks
source-triggered email/online-application requirements and the submission deadline's
starting event. Its version is included in the generator identity to expire older cached outputs.

## Quick start on Windows

Run from the repository root. Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\lmaana.exe ingest configs/sources.example.json
.\.venv\Scripts\lmaana.exe serve
```

In a second terminal, from the same directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Open [the interface](http://127.0.0.1:8501) or [the API documentation](http://127.0.0.1:8000/docs).
Try: `شنو الوثائق لي خاصني باش ندير auto-entrepreneur؟`

The seed corpus includes the registration section of the official **DGI 2026 guide**,
reviewed on 2026-10-03, with an internal review deadline of 2026-11-02. During that
window, the example returns a French `source_excerpts` passage, not a generated Darija
answer. The excerpt retains the signed application, partner-bank restriction,
foreign-resident alternative, and deposit deadline across two PDF pages. This is a
simplified official guide, not a guarantee that practical requirements remain unchanged.
The historical Maroc PME brochure and undated CRI procedure remain withheld references.
After approval expiry, a relevant question returns `verification_required` until a
new review. Zero approved sources is a valid, ready service state.
Only manifests are committed; downloaded/extracted content and indexes stay local.
See the [source review policy and evidence log](docs/source-quality.md).

The default `lexical`/`excerpt` profile needs no model runtime. Lexical term vectors are a
development baseline, not semantic multilingual embeddings. Short Darija/French aliases help
with this initial example; this is not a retrieval-quality benchmark.

On Linux/macOS, use `.venv/bin/python` and `.venv/bin/lmaana` instead of the Windows paths.
With uv installed, `uv sync --locked --extra dev --extra ui` installs the locked environment.

## Commands and endpoints

```powershell
.\.venv\Scripts\lmaana.exe doctor
.\.venv\Scripts\lmaana.exe ask "Quels documents pour l'inscription auto-entrepreneur ?"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests apps
```

| Endpoint | Behavior |
| --- | --- |
| `GET /health/live` | Process liveness, without inference. |
| `GET /health/ready` | Corpus and configured generator readiness; 503 if unavailable. |
| `POST /v1/answers` | JSON `{"question": "..."}`; returns outcome, citations, versions, and timings. |

Questions must be nonblank and at most 1,500 characters. Outcomes are `source_excerpts`
(development mode), `answered`, `clarification_required`, `insufficient_evidence`, or
`verification_required`. The latter has no answer citations; `related_sources` lists
withheld references, separately from `citations`. Each reference includes its review status.
Busy inference returns 429, dependency outages 503, and model timeouts 504. A timeout marks
inference unavailable until the API is restarted, to avoid overlapping a possibly unfinished job.

## Adding documents

Copy `configs/sources.example.json`, add approved sources, and mark them reviewed after inspection.
Each source records title, publisher, URL, language, topic, reuse status, and freshness notes.
Optional `pages` are one-based PDF page numbers; `selector` selects HTML content.
Optional `section_heading` keeps only that exact extracted heading and fails closed
if it disappears. `reviewed: true` authorizes ingestion, not current-answer usage.
For a PDF section spanning pages, `pdf_section` specifies exact start/end markers,
the reviewed running header and the printed-page offset. Pages must be consecutive.
Only the expected leading header/folio is removed; body characters are retained,
with a newline joining page continuations. Missing/ambiguous markers or changed
headers abort ingestion. The section must fit one chunk; it is never split midway
through conditions. Locations use PDF page numbers, not the printed folios.
Sources default to `verification.status=unverified`. See the review policy before
creating a dated, content-hash-bound approval; a fetch date is not such an approval.
For local text/PDF/HTML, `local_path` is relative to the manifest and must stay inside its directory.
Remote URLs and each redirect must use HTTPS on an explicitly allowed host with public DNS addresses.

**Stop the API before ingesting.** Local Qdrant storage has one owner. Then run ingestion and
restart the API. A failed build leaves the previous active corpus intact. Old collections remain
available for recovery; this version does not automatically delete them.

Changing the embedding model or its revision requires re-ingestion. Query aliases, chunking,
model identifiers, dimensions, and distance conventions are versioned. Similarity thresholds are
uncalibrated starting values, not probabilities of correctness.
The current query rules are `darija-rules-v4`, and the lexical profile is
`lexical-sha256-4096-v3` (search-only normalization and multilingual question stop words).
Re-ingest after upgrading query or chunking rules; the API rejects incompatible indexes.
Chunking is `source-boundaries-v2-pdf-sections-size350-overlap50`. Card-delivery
duration questions require delivery and duration cues, not just a deposit deadline.
This is a targeted heuristic, not general semantic understanding.
The corpus schema is now 2, with source policy `reviewed-currency-v1`.
After upgrading Python modules, restart both the API and Streamlit; hot reload can
retain imported modules from the previous version. Old cached answers are discarded
when the source policy or corpus changes, or a cited review expires.

## Enabling the neural profile

Use `.env.example` as a reference. `.env` is optional and ignored by Git.

### Ollama (local generation)

Install [Ollama for Windows](https://docs.ollama.com/windows), then open a new terminal:

```powershell
ollama pull qwen3:4b
ollama list
```

Put these settings in the local `.env`, then restart **both** the API and Streamlit:

```dotenv
LMAANA_EMBEDDING_BACKEND=lexical
LMAANA_GENERATOR_BACKEND=ollama
LMAANA_OLLAMA_URL=http://127.0.0.1:11434
LMAANA_OLLAMA_MODEL=qwen3:4b
LMAANA_CONTEXT_TOKENS=8192
LMAANA_MAX_OUTPUT_TOKENS=1024
LMAANA_TIMEOUT_SECONDS=120
```

To pin the downloaded model manifest, copy its full `digest` from `GET /api/tags`
into `LMAANA_OLLAMA_MODEL_DIGEST`. A changed digest makes readiness fail rather
than silently using different weights. Generation does not need the `models`
Python extra and does not require re-ingesting the existing lexical corpus.

The adapter uses Ollama's native `/api/chat` with JSON schema, `think=false`,
`stream=false`, explicit context/output budgets and `truncate=false`/`shift=false`.
It checks `/api/tags` and `/api/show` for a local Qwen3 model, supported context
and optional digest. No automatic pull, remote model, redirect or excerpt fallback
is used. Metadata readiness does not prove a future GPU allocation will succeed.
The context preflight uses a conservative UTF-8 byte budget plus template reserve,
not an exact token count; entire lower-ranked passages are removed if necessary.
The native adapter was tested with Ollama 0.35.1; older runtimes may not support
all request controls. Invalid output is retried once, then withheld explicitly.

For local-only operation, disable Ollama's cloud features as described in its
[official FAQ](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features)
and restart Ollama. This machine has `disable_ollama_cloud=true` in Ollama's
per-user `server.json`, confirmed by the server log. The service listens on loopback.

The previously downloaded official GGUF can also be imported without downloading
weights again: `ollama create lmaana-qwen3:4b -f configs/Modelfile.qwen3`.
An imported GGUF and a registry tag are distinct artifacts; compare their actual
outputs and record their digests. Do not assume identical quality or templates.

Local validation on 2026-10-03:

- Active registry model: `qwen3:4b`, Q4_K_M, manifest digest
  `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- Ollama reported 100% GPU placement, about 3.9 GB loaded, context 8192.
- 244 offline tests pass. Earlier desktop/mobile browser checks covered explicit
  rejection of invalid Darija output. These are software checks, not a quality score.
- In the earlier 14-case live smoke run (before `generation-validation-v2`),
  10 matched the expected outcome. French and
  English document answers were generated with citations. Four Arabic/Darija/Arabizi
  answer attempts were withheld after validation failure, **not successful answers**.
  This includes a French question with a manual Darija override.
- The imported GGUF also had translation problems. Enabling thinking in a diagnostic
  trial did not resolve them; the active adapter keeps thinking disabled.

The earlier live run is saved locally in `outputs/Lmaana-Ollama-Checks.json` (not committed).
The latest `generation-validation-v2` smoke run matched 9 of 14 expected outcomes:
one generated French answer and eight expected abstentions/clarifications. Five
expected answers (Arabic, Darija, Arabizi, English and manual Darija override) were
withheld. All requests returned HTTP 200. The report and offline metrics are in
`outputs/Lmaana-Reliability-Final-Checks.json` and
`outputs/Lmaana-Reliability-Final-Metrics.json`. Missing human claim annotations
leave faithfulness unevaluated. These small stochastic runs are not a language
benchmark or a paired estimate of model improvement.
Fluent, faithful Darija generation remains unfinished; detection alone cannot provide it.

### Optional semantic embeddings and standalone llama.cpp

1. Install the `models` extra: `pip install -e ".[models]"` inside the virtual environment.
2. Set `LMAANA_EMBEDDING_BACKEND=qwen`, a full `LMAANA_EMBEDDING_REVISION` commit hash,
   and `LMAANA_EMBEDDING_DEVICE=cpu`. This loads Qwen3-Embedding-0.6B and may download weights.
3. Re-ingest the sources, then stop ingestion before serving.
4. Run a compatible llama.cpp server with Qwen3-4B Q4_K_M. Example using a separately downloaded
   GGUF file: `llama-server -m models/qwen3/Qwen3-4B-Q4_K_M.gguf -a lmaana-qwen3 -c 4096 -np 1 -ngl 0 --jinja`.
5. Set `LMAANA_GENERATOR_BACKEND=llamacpp`, then restart the API. The adapter checks the server's
   model alias and context size and uses `/apply-template`, `/tokenize`, and `/completion`.

Generation and embeddings are independent: keep `LMAANA_EMBEDDING_BACKEND=lexical`
to test a local Qwen generator without installing the `models` extra or downloading
Qwen embeddings. Use `--host 127.0.0.1 --port 8080 --reasoning off` with a compatible
llama.cpp build; the adapter also passes `enable_thinking=false` to the template.

Local preparation on 2026-10-03 selected official `Qwen/Qwen3-4B-GGUF` revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`, file `Qwen3-4B-Q4_K_M.gguf`, SHA-256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The official llama.cpp `b11146` Windows Vulkan archive was downloaded and its
GitHub-published SHA-256 verified. Windows Application Control blocked loading
that build's `ggml-vulkan.dll` with error 4551. It remains unvalidated; the policy
was not disabled or bypassed. The separately installed official Ollama runtime
successfully loaded Qwen3 on CUDA. Repository defaults remain excerpt mode;
the ignored local `.env` explicitly selects Ollama on this machine.

The standalone llama.cpp example starts generation on CPU. Profile GPU offload before sharing the 8 GB RTX 5070 Laptop
GPU with Lmaana 2.4. No cloud inference fallback or silent switch to another ASR model is configured.
Do not paste Hugging Face tokens into configuration committed to Git.

## Layout and architecture

```text
apps/streamlit_app.py             HTTP-only text interface
src/lmaana_assistant/
  api/                           FastAPI lifecycle and routes
  ingestion/                     source fetching, extraction, corpus activation
  retrieval/                     lexical/Qwen embeddings and Qdrant storage
  generation/                    excerpt/llama.cpp/Ollama adapters and validation
  evaluation/                    offline metrics and annotation validation
  asr/                           reserved for the Lmaana 2.4 milestone
  contracts.py                   typed shared records
  normalization.py               Darija rules and search aliases
  language.py                    detection, reply-language policy and localized messages
  pipeline.py                    retrieval-to-response orchestration
  config.py                      explicit environment-driven profiles
  cli.py                         ingest, serve, ask, evaluate, doctor
configs/                         reviewed-source manifest and synthetic metric examples
tests/                           offline unit and integration tests
docs/                            architecture and model integration notes
data/                            generated local corpus/index files (ignored)
```

![Target architecture](docs/diagrams/system.svg)

The [architecture design](docs/architecture.md) describes the full target and current differences.
Tests cover ingestion failures, storage locks, index compatibility, citation rejection, empty
corpora, concurrent requests, and model timeouts. They run without model weights or network access.
CI checks Python 3.11/3.12 on Linux and Windows.
These are software behavior checks, not a certification of administrative correctness.

Next: install and validate the native **Lmaana 2.4** fairseq2/OmniASR runtime, then connect the
validated provider to the existing transcription endpoint and add microphone input,
then compare against Whisper on held-out audio. Measure WER/CER, retrieval Recall@K/MRR,
claim support, and end-to-end latency before making quality claims.

## Offline evaluation

Evaluate saved outputs and annotations without loading any models or contacting a server:

```powershell
.\.venv\Scripts\lmaana.exe evaluate configs/evaluation.example.json --output outputs/evaluation-example.json
```

The included dataset is explicitly synthetic: its scores are metric examples, not
Lmaana, Whisper or Qwen benchmark results. The evaluator reports corpus WER/CER,
Recall/Precision/Hit/MRR at configured cutoffs, outcome agreement, answer coverage,
generation failures and latency percentiles. Human claim annotations are optional;
missing annotations leave faithfulness unevaluated. Reply-language labels do not
measure fluency. See the [evaluation protocol](docs/evaluation.md) for empty-reference
handling, negative queries, denominators and annotation requirements.
