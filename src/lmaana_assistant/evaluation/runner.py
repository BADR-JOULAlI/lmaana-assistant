"""Evaluate supplied transcripts, retrieval rankings and response annotations."""

from collections import Counter
from datetime import UTC, datetime

from lmaana_assistant.evaluation.dataset import EvaluationDataset
from lmaana_assistant.evaluation.metrics import (
    edit_counts,
    latency_summary,
    mean_defined,
    sum_counts,
)


def evaluate(dataset: EvaluationDataset) -> dict:
    report = {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.dataset_id,
        "notes": dataset.notes,
        "model_profile": dataset.model_profile,
        "corpus_release": dataset.corpus_release,
        "scope": (
            "Offline metrics from supplied outputs and annotations; not a fluency certificate."
        ),
    }
    if dataset.asr:
        rows, word_counts, character_counts = [], [], []
        for case in dataset.asr:
            words = edit_counts(case.reference.split(), case.hypothesis.split())
            reference, hypothesis = case.reference, case.hypothesis
            if dataset.cer_remove_whitespace:
                reference = "".join(reference.split())
                hypothesis = "".join(hypothesis.split())
            characters = edit_counts(reference, hypothesis)
            word_counts.append(words)
            character_counts.append(characters)
            rows.append({"id": case.id, "wer": words.to_dict(), "cer": characters.to_dict()})
        report["asr"] = {
            "case_count": len(rows),
            "word_tokenization": "whitespace-split",
            "character_units": "unicode-code-points",
            "normalization": "none",
            "cer_remove_whitespace": dataset.cer_remove_whitespace,
            "empty_reference_policy": "rate=null; insertions retained in corpus totals",
            "wer": sum_counts(word_counts).to_dict(),
            "cer": sum_counts(character_counts).to_dict(),
            "macro_wer_nonempty_references": mean_defined([x.error_rate for x in word_counts]),
            "macro_wer_sample_count": sum(x.reference_units > 0 for x in word_counts),
            "cases": rows,
        }
    if dataset.retrieval:
        rows = []
        for case in dataset.retrieval:
            relevant = set(case.relevant_ids)
            first = next(
                (
                    rank
                    for rank, identifier in enumerate(case.ranked_ids, 1)
                    if identifier in relevant
                ),
                None,
            )
            scores = {}
            for k in dataset.retrieval_k:
                found = len(relevant & set(case.ranked_ids[:k]))
                scores[str(k)] = {
                    "recall": found / len(relevant) if relevant else None,
                    "precision": found / k,
                    "hit": int(found > 0),
                    "reciprocal_rank": 1 / first if first is not None and first <= k else 0,
                }
            rows.append(
                {
                    "id": case.id,
                    "relevant_count": len(relevant),
                    "retrieved_count": len(case.ranked_ids),
                    "first_relevant_rank": first,
                    "at_k": scores,
                }
            )
        positive = [row for row in rows if row["relevant_count"]]
        negative = [row for row in rows if not row["relevant_count"]]
        report["retrieval"] = {
            "case_count": len(rows),
            "queries_with_relevant_passages": len(positive),
            "queries_without_relevant_passages": len(negative),
            "aggregation": "macro over queries with relevant passages; precision denominator is K",
            "at_k": {
                str(k): {
                    "mean_recall": mean_defined(
                        [row["at_k"][str(k)]["recall"] for row in positive]
                    ),
                    "mean_precision": mean_defined(
                        [row["at_k"][str(k)]["precision"] for row in positive]
                    ),
                    "hit_rate": mean_defined([row["at_k"][str(k)]["hit"] for row in positive]),
                    "mrr": mean_defined(
                        [row["at_k"][str(k)]["reciprocal_rank"] for row in positive]
                    ),
                }
                for k in dataset.retrieval_k
            },
            "negative_query_nonempty_result_rate": mean_defined(
                [float(row["retrieved_count"] > 0) for row in negative]
            ),
            "cases": rows,
        }
    if dataset.responses:
        cases = dataset.responses
        desired = [case for case in cases if case.expected_outcome == "answered"]
        answered = [case for case in cases if case.actual_outcome == "answered"]
        annotated = [case for case in answered if case.claim_support]
        labels = Counter(label for case in annotated for label in case.claim_support)
        report["responses"] = {
            "case_count": len(cases),
            "outcome_match_count": sum(
                case.expected_outcome == case.actual_outcome for case in cases
            ),
            "outcome_match_rate": mean_defined(
                [float(case.expected_outcome == case.actual_outcome) for case in cases]
            ),
            "reply_language_label_match_rate": mean_defined(
                [
                    float(case.expected_reply_language == case.actual_reply_language)
                    for case in cases
                ]
            ),
            "language_metric_scope": "Metadata labels only; dialect fluency is not measured.",
            "generated_answer_count": len(answered),
            "generated_answer_coverage": len(answered) / len(cases),
            "desired_answer_count": len(desired),
            "desired_answer_outcome_recall": mean_defined(
                [float(case.actual_outcome == "answered") for case in desired]
            ),
            "abstained_despite_expected_answer": sum(
                case.actual_outcome in {"insufficient_evidence", "verification_required"}
                for case in desired
            ),
            "generation_failure_count": sum(case.generation_failed for case in cases),
            "claim_support": {
                "annotated_answer_count": len(annotated),
                "unannotated_answer_count": len(answered) - len(annotated),
                "annotated_claim_count": sum(labels.values()),
                "labels": dict(labels),
                "fully_supported_claim_rate": labels["supported"] / sum(labels.values())
                if labels
                else None,
                "scope": "Human-supplied annotations only; missing labels remain unevaluated.",
            },
            "latency": latency_summary(
                [case.latency_ms for case in cases if case.latency_ms is not None]
            ),
            "cases": [case.model_dump() for case in cases],
        }
    return report
