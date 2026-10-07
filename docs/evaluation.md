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
