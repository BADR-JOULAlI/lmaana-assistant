# Evaluation protocol

`lmaana evaluate` measures supplied outputs offline. It does not download models,
transcribe audio, run retrieval, or ask a language model to grade itself.

```powershell
.\.venv\Scripts\lmaana.exe evaluate configs/evaluation.example.json
.\.venv\Scripts\lmaana.exe evaluate configs/evaluation.example.json --output outputs/evaluation-example.json
```

The example contains fictional software scenarios and authored transcripts. Its
scores demonstrate arithmetic only. Do not report them as Lmaana/Whisper/Qwen
performance. Replace them with held-out outputs and annotations for a benchmark.

## Dataset contract

The JSON format has `schema_version: 1`, a nonblank `dataset_id`, optional `notes`,
`model_profile` and `corpus_release`, and at least one nonempty task:

- `asr`: case ID, reference transcript, hypothesis transcript.
- `retrieval`: case ID, all relevant passage IDs, ranked retrieved passage IDs.
- `responses`: case ID, expected/actual outcome, expected/actual reply-language
  label, optional `generation_failed`, `latency_ms` and human `claim_support` labels.

IDs must be unique within a task. Duplicate ranked passage IDs are rejected rather
than counted twice. Unknown fields and inconsistent annotations are rejected.
The command returns a nonzero exit code for invalid input. A valid report is not
automatically a quality pass: there are no universal acceptance thresholds.
The output path cannot be the input dataset or an existing link to that file.

Record the exact source corpus, embedding fingerprint, generator/model digest,
prompt/validation policy, audio split and decoding settings in experiment notes.
Compare systems against the same references and ground truth. Keep development,
validation and final test data separate, including speakers and duplicates.

## Transcription

WER uses whitespace-delimited words; CER uses Unicode code points. No automatic
spelling, punctuation or dialect normalization is applied. Set
`cer_remove_whitespace: true` for a separate whitespace-free CER protocol. This
does not change WER tokenization. Grapheme clusters and tokenizer-unit error rates
are different metrics and are not implemented by this command.

The minimum edit alignment reports substitutions, deletions and insertions.
Equivalent-cost alignments prefer fewer substitutions, then fewer deletions;
breakdowns can differ from another implementation while the total WER is equal.

Corpus WER/CER add error counts and reference lengths before division. Macro WER
averages defined per-clip rates. An empty reference has a null rate; its inserted
words still contribute to corpus totals. All-empty references leave the rate null.
WER can exceed 1 when insertions exceed the reference length.

These functions are ready for future Lmaana 2.4 and Whisper outputs. They are not
evidence that ASR inference is integrated or that either model has been benchmarked.

## Retrieval

Set positive unique cutoffs with `retrieval_k`, such as `[1, 3, 5, 10]`.

- Recall@K: relevant IDs recovered / all relevant IDs.
- Precision@K: relevant IDs recovered / K. Missing result positions are penalized.
- Hit@K: at least one relevant result in the first K positions.
- MRR@K: mean inverse first-relevant rank, zero if absent within K.

Aggregate these over queries with at least one relevant passage. Questions with
no relevant ground truth are reported separately, including the proportion with
nonempty retrieved results. Null recall for a negative query must not be treated
as zero or one. Ground-truth relevance should distinguish topical similarity from
usable evidence; keep the annotation convention fixed across experiments.

### Running passage retrieval instead of supplying rankings

`lmaana benchmark-retrieval` encodes a supplied, pre-segmented passage set, runs
each written question, and feeds the measured rankings into the same offline
metric functions. It never opens Qdrant, reads the active corpus, calls the API,
loads ASR, or generates answers. Graphify dependency tracing helped keep this
experiment separate from `AnswerPipeline` and ingestion.

```powershell
.\.venv\Scripts\lmaana.exe benchmark-retrieval configs/retrieval-benchmark.example.json --query-mode expanded --output outputs/retrieval-lexical-expanded-v1.json
.\.venv\Scripts\lmaana.exe benchmark-retrieval configs/retrieval-benchmark.example.json --query-mode raw --output outputs/retrieval-lexical-raw-v1.json
```

The dataset declares `passages` (`id`, `text`), `questions` (`id`, `question`,
author-assigned `language`, `relevant_ids`), `retrieval_k`, `dataset_id`, and
nonblank experiment `notes`. An empty `relevant_ids` list is an explicit negative.
Unknown references, duplicates and blank fields fail validation. The output
cannot overwrite the dataset, including an existing link to the input file.

The committed example has eight fictional **software** passages, all French,
and forty questions: seven positives and one negative for each of `fr`, `ar`,
`ary`, `ary-Latn`, and `en`. Translated/paraphrased questions share scenarios;
they are development probes, not independent held-out samples. No administrative
requirements or real-model outputs have been invented for this fixture.

Reports retain question text, expected IDs, query expansion, intent and detected
language, ranked IDs with cosine scores, and the results retained by `--min-score`.
The dataset SHA-256 is over canonical validated JSON, not the source file's layout.
Record the embedding fingerprint, query-rules version, query mode, cutoffs and
threshold when comparing runs. Exact score ties sort by passage ID. Zero vectors
are not arbitrarily ranked; malformed or nonfinite embeddings fail the run.
Cosine-vector validation is shared with the production store. Ingestion rejects
zero document vectors before creating a collection; this diagnostic counts and
omits them from ranking. Production searches return no results for a zero query
and report malformed query embeddings as dependency failures. Valid vectors are
normalized before Qdrant's float32 storage/search to handle extreme finite scales.

`retrieval` reports thresholded metrics; `unfiltered_retrieval` reports ranking
quality before the threshold. Both are also stratified by **annotated**, not
automatically detected, language. Negative queries have a separate denominator.
The test does not apply topic/currentness filters or answerability guards:
a nonempty negative ranking is a retrieval candidate, **not a wrong displayed
answer**. Those protections are tested separately in the pipeline tests.

Latency is a single sequential measurement, with document encoding and CLI model
loading reported separately. There is no warmup/repetition protocol; do not use
this small diagnostic as a production throughput or cold-start benchmark.

### Lexical development baseline

Measured with `lexical-sha256-4096-v3`, `darija-rules-v5`, cosine threshold `0.18`,
and the committed fixture hash
`f8f25f3c04cf1a3eb2d4a8281b752038c44c82090a17044b9720f4baf9f5374d`:

| Query input | Positive top-1 hits | Positive Recall@1 | Negative queries with a candidate |
|---|---:|---:|---:|
| Raw question | 15 / 35 | 42.9% | 0 / 5 |
| Expanded question | 17 / 35 | 48.6% | 3 / 5 |

For the expanded run, thresholded top-1 hits are French 7/7, Arabic 1/7, Darija
0/7, Arabizi 6/7, and English 3/7. This fixture intentionally exposes cross-language
vocabulary gaps. The result does **not** measure real Moroccan-service recall,
ASR accuracy, response correctness, or dialect fluency. Do not tune the final
threshold on these probes and then report the same probes as held-out performance.

### Comparing the optional Qwen profile

The Qwen adapter uses the documented instruction format from the
[official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Use a dedicated virtual environment **without** `--system-site-packages`, install
the optional models extra, and supply a full 40-character checkpoint commit.
Keep the device on CPU for an experiment that should not consume the ASR GPU:

```powershell
python -m venv work/qwen-benchmark/.venv
.\work\qwen-benchmark\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\work\qwen-benchmark\.venv\Scripts\python.exe -m pip install -e ".[models]"
# Verify native runtime imports before spending time downloading model weights.
.\work\qwen-benchmark\.venv\Scripts\python.exe -c "import torch, sentence_transformers; print(torch.__version__, sentence_transformers.__version__)"
# Replace FULL_40_HEX_COMMIT with the checkpoint revision being evaluated.
.\work\qwen-benchmark\.venv\Scripts\lmaana.exe benchmark-retrieval configs/retrieval-benchmark.example.json --backend qwen --revision FULL_40_HEX_COMMIT --device cpu --query-mode expanded --output outputs/retrieval-qwen-expanded-v1.json
```

This command is **local-cache-only by default**, using Sentence Transformers'
[`local_files_only` option](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
Add `--allow-download` only when intentionally downloading the pinned checkpoint.
Missing dependencies or a missing cached model fail explicitly; there is no
silent lexical fallback. The application has a separate opt-in
`LMAANA_EMBEDDING_LOCAL_FILES_ONLY=true` setting. Neither command changes `.env`
or activates a new serving index.

The Qwen benchmark uses explicit CLI arguments and profile defaults only:
neither the serving `LMAANA_*` environment nor `.env` selects its checkpoint or
ASR/generation settings. The normal application still reads its environment.
Reports now include `runtime` with Python/OS, installed library versions,
the model parameter's actual device and dtype, maximum sequence length and CPU
thread settings. Optional libraries are not imported merely to collect metadata;
absent versions and non-model fields remain null, not guessed from CLI arguments.
The dependency analysis kept this provenance collection outside the serving path.
Inherited optional training packages can import unrelated native libraries even
for an embedding-only task. On this machine, a system-site-packages trial imported
global `datasets`/PyArrow and hit Windows Application Control. The dedicated
inference environment excludes these optional packages, but this alone does not
guarantee that its other native dependencies are permitted. No system policy is
changed. A missing Sentence Transformers package and a failed dependency import
have distinct errors, with the original exception retained for diagnostics.
The local benchmark CLI also prints an explicitly chained import cause on stderr;
it does not silently substitute lexical scores or overwrite a prior report on a
model-initialization failure. Arbitrary non-import causes are not automatically
printed as extra diagnostic details.

### Local Qwen runtime finding (22 October 2026)

Preparation confirmed the official Qwen checkpoint revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and its
[published weight metadata](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B/revision/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3?blobs=true):
`model.safetensors`, 1,191,586,416 bytes, expected SHA-256
`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.
The completed local download matched both this byte count and SHA-256 and was
placed in the pinned Hugging Face snapshot alongside its configuration/tokenizer
files. No partial weight file was loaded. Artifact integrity is verified; model
inference is **not** verified.

The clean Windows environment uses Python 3.12.10, Torch `2.14.1+cpu`,
Transformers `4.57.6`, Sentence Transformers `5.7.0`, scikit-learn `1.9.1` and
NumPy `2.5.4`. `datasets` and PyArrow are absent. Torch imports and reports four CPU
threads; importing Sentence Transformers fails in scikit-learn's
`_openmp_helpers.cp312-win_amd64.pyd` with an Application Control denial. Windows
Code Integrity events 3033/3077 independently record the denied module. This is
not a missing-checkpoint or GPU-memory error. No Qwen weights were loaded and
neither query mode ran.

Stop at this security boundary: no policy changes, DLL relocation, or repeated
attempts to execute the blocked module. The machine owner/security administrator
must review the dependency and policy before this Windows runtime can be retried.
The active corpus, serving environment and Lmaana 2.4 ASR runtime remain unchanged.

No actual Qwen benchmark result is supplied with this baseline: adapter tests use
controlled vectors and loader stubs. Run both query modes on the same dataset
revision, then use a separately reviewed Moroccan-document development/held-out
set before selecting a model or recalibrating thresholds. A shared numeric
threshold is an experimental control, not proof that scores from different
embedders are calibrated. Changing the serving embedding profile still requires
re-ingestion and restart; never replace the active index with this software fixture.

## Responses and latency

Outcome agreement, generated-answer coverage, desired-answer outcome recall,
abstentions despite an expected answer and generation failures are separate
measures. `source_excerpts` does not count as a generated answer. A rejected
generation cannot count as an accepted answer.

Reply-language agreement compares metadata labels only. It does not measure
Darija fluency, translation quality, factual correctness or condition coverage.

Human claim labels are `supported`, `partial`, `unsupported` and `contradicted`.
The fully supported claim rate uses only labelled claims. Missing or empty labels
leave the answer unannotated and the metric null when no labels exist. This rate
does not detect omitted claims: assess checklist completeness separately.

Latency is in milliseconds. P50/P95 use linear interpolation at `(n-1)*p`, and the
sample count is always included. No samples produces null percentiles. Record
cold/warm model state, caching, retries and workload before interpreting timings.

## Current validation findings

The 3 October prompt diagnostics use fixed DGI evidence and a fixed sampling seed.
The previous diagnostic script had constructed extra language instructions but
sent the original messages. This has been corrected. Vocabulary, a fictional
language example and clause-by-clause rewriting still produced mistranslations or
omissions. One example-guided Darija output passed the older heuristic despite
missing email and online-registration requirements; it is now rejected.

The neural validation policy is `generation-validation-v2`, included in the
generator identity to invalidate cached answers from older policies. It adds
source-triggered email and online-application coverage and checks that the initial
submission deadline retains a submission action and an application-creation
starting event when the source specifies one. These
remain conservative lexical checks, not a general entailment or fluency model.

Live reports and prompt diagnostics are local experiment artifacts. They are
small smoke tests, not statistically representative benchmarks. The earlier
14-case report and its outputs are retained separately from new runs.

After the final changes, all 244 offline software tests passed. In the latest
14-case local smoke run, all requests returned HTTP 200 and nine matched the
expected outcome: one French generated answer, seven expected abstentions and
one expected clarification. Five expected answers were withheld after validation
failure: Arabic, Darija, Arabizi, English and a French question with Darija override.
The corresponding human claim-support score remains null because these outputs
have not been annotated. This is a small integration check, not a fluency or
administrative-correctness benchmark. Live sampling is stochastic; unlike the
prompt probes, these two smoke runs are not a fixed-seed paired comparison.

Local artifacts: `outputs/Lmaana-Reliability-Final-Checks.json`,
`outputs/Lmaana-Reliability-Final-Annotations.json` and
`outputs/Lmaana-Reliability-Final-Metrics.json`. Recompute the response metrics with:

```powershell
.\.venv\Scripts\lmaana.exe evaluate outputs/Lmaana-Reliability-Final-Annotations.json --output outputs/Lmaana-Reliability-Final-Metrics.json
```
