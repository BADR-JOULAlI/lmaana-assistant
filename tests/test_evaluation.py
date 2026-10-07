import json
from math import inf, nan

import pytest
from pydantic import ValidationError

from lmaana_assistant.evaluation.dataset import (
    AsrCase,
    EvaluationDataset,
    ResponseCase,
    RetrievalCase,
)
from lmaana_assistant.evaluation.metrics import edit_counts, latency_summary
from lmaana_assistant.evaluation.runner import evaluate


@pytest.mark.parametrize(
    ("reference", "hypothesis", "counts"),
    [
        ("je veux une carte", "je veux carte maintenant", (0, 1, 1)),
        ("je veux une photo", "je veux deux photos demain", (2, 0, 1)),
        ("a", "a b c", (0, 0, 2)),
        ("a b", "", (0, 2, 0)),
        ("", "a b", (0, 0, 2)),
        ("", "", (0, 0, 0)),
        ("شنو خاصني", "شنو خاصني", (0, 0, 0)),
        ("a b", "b a", (0, 1, 1)),
    ],
)
def test_edit_alignment(reference, hypothesis, counts):
    result = edit_counts(reference.split(), hypothesis.split())
    assert (result.substitutions, result.deletions, result.insertions) == counts
    assert result.reference_units == len(reference.split())


def test_empty_reference_does_not_become_zero_error_or_disappear_from_totals():
    dataset = EvaluationDataset(
        dataset_id="empty-and-nonempty",
        asr=[
            AsrCase(id="silence", reference="", hypothesis="invented words"),
            AsrCase(id="speech", reference="one two", hypothesis="one two"),
        ],
    )
    report = evaluate(dataset)["asr"]
    assert report["cases"][0]["wer"]["error_rate"] is None
    assert report["wer"]["insertions"] == 2
    assert report["wer"]["error_rate"] == 1
    assert report["macro_wer_nonempty_references"] == 0
    assert report["macro_wer_sample_count"] == 1


def test_corpus_wer_is_weighted_by_reference_length():
    dataset = EvaluationDataset(
        dataset_id="different-durations",
        asr=[
            AsrCase(id="short", reference="a", hypothesis="b"),
            AsrCase(id="long", reference="a b c d e f g h i", hypothesis="a b c d e f g h i"),
        ],
    )
    report = evaluate(dataset)["asr"]
    assert report["wer"]["error_rate"] == 0.1
    assert report["macro_wer_nonempty_references"] == 0.5


def test_empty_references_and_negative_only_retrieval_have_no_fabricated_scores():
    dataset = EvaluationDataset(
        dataset_id="no-positive-reference",
        asr=[AsrCase(id="silence", reference="", hypothesis="noise")],
        retrieval=[RetrievalCase(id="unknown", relevant_ids=[], ranked_ids=[])],
    )
    report = evaluate(dataset)
    assert report["asr"]["wer"]["error_rate"] is None
    assert report["asr"]["wer"]["insertions"] == 1
    assert report["asr"]["macro_wer_nonempty_references"] is None
    assert report["retrieval"]["at_k"]["1"]["mrr"] is None
    assert report["retrieval"]["negative_query_nonempty_result_rate"] == 0


def test_character_whitespace_policy_is_explicit():
    dataset = EvaluationDataset(
        dataset_id="orthography",
        asr=[AsrCase(id="spaces", reference="شنو خاصني", hypothesis="شنوخاصني")],
    )
    assert evaluate(dataset)["asr"]["cer"]["deletions"] == 1
    dataset.cer_remove_whitespace = True
    assert evaluate(dataset)["asr"]["cer"]["errors"] == 0
    assert evaluate(dataset)["asr"]["word_tokenization"] == "whitespace-split"


def test_retrieval_cutoffs_and_negative_queries_have_separate_denominators():
    dataset = EvaluationDataset(
        dataset_id="retrieval",
        retrieval_k=[1, 3],
        retrieval=[
            RetrievalCase(id="two", relevant_ids=["A", "C"], ranked_ids=["B", "A", "C"]),
            RetrievalCase(id="miss", relevant_ids=["D"], ranked_ids=["E"]),
            RetrievalCase(id="no-evidence", relevant_ids=[], ranked_ids=["noise"]),
        ],
    )
    result = evaluate(dataset)["retrieval"]
    assert result["at_k"]["1"]["mrr"] == 0
    assert result["at_k"]["3"]["mrr"] == 0.25
    assert result["at_k"]["3"]["mean_recall"] == 0.5
    assert result["at_k"]["3"]["mean_precision"] == pytest.approx(1 / 3)
    assert result["at_k"]["3"]["hit_rate"] == 0.5
    assert result["queries_with_relevant_passages"] == 2
    assert result["negative_query_nonempty_result_rate"] == 1
    assert result["cases"][2]["at_k"]["3"]["recall"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"relevant_ids": ["a", "a"]},
        {"ranked_ids": ["a", "a"]},
        {"ranked_ids": [" "]},
    ],
)
def test_duplicate_or_blank_passages_cannot_inflate_metrics(overrides):
    with pytest.raises(ValidationError):
        RetrievalCase.model_validate(
            {"id": "test", "relevant_ids": ["a"], "ranked_ids": ["a"], **overrides}
        )


def response_case(identifier="case", **overrides):
    return ResponseCase(
        **{
            "id": identifier,
            "expected_outcome": "answered",
            "actual_outcome": "answered",
            "expected_reply_language": "ary",
            "actual_reply_language": "ary",
            **overrides,
        }
    )


def test_missing_claim_annotations_are_not_treated_as_perfect_faithfulness():
    dataset = EvaluationDataset(dataset_id="unannotated", responses=[response_case()])
    scores = evaluate(dataset)["responses"]
    assert scores["reply_language_label_match_rate"] == 1
    assert scores["claim_support"]["fully_supported_claim_rate"] is None
    assert scores["claim_support"]["unannotated_answer_count"] == 1
    assert scores["latency"]["sample_count"] == 0
    assert scores["latency"]["p95_ms"] is None


def test_rejected_generation_does_not_count_as_answer_success():
    dataset = EvaluationDataset(
        dataset_id="rejections",
        responses=[
            response_case("good", claim_support=["supported", "partial"]),
            response_case("failed", actual_outcome="insufficient_evidence", generation_failed=True),
            response_case(
                "expected-refusal",
                expected_outcome="insufficient_evidence",
                actual_outcome="insufficient_evidence",
            ),
        ],
    )
    scores = evaluate(dataset)["responses"]
    assert scores["outcome_match_count"] == 2
    assert scores["generated_answer_count"] == 1
    assert scores["generated_answer_coverage"] == pytest.approx(1 / 3)
    assert scores["desired_answer_outcome_recall"] == 0.5
    assert scores["abstained_despite_expected_answer"] == 1
    assert scores["generation_failure_count"] == 1
    assert scores["claim_support"]["fully_supported_claim_rate"] == 0.5


@pytest.mark.parametrize("problem", ["failed-answer", "claim-on-abstention", "infinite-latency"])
def test_inconsistent_response_annotations_are_rejected(problem):
    overrides = {
        "failed-answer": {"generation_failed": True},
        "claim-on-abstention": {
            "actual_outcome": "insufficient_evidence",
            "claim_support": ["supported"],
        },
        "infinite-latency": {"latency_ms": inf},
    }
    with pytest.raises(ValidationError):
        response_case(**overrides[problem])


@pytest.mark.parametrize("values", [[], [5], [100, 200, 1000]])
def test_latency_percentiles_publish_sample_count(values):
    result = latency_summary(values)
    assert result["sample_count"] == len(values)
    if not values:
        assert result["p50_ms"] is None
    elif len(values) == 1:
        assert result["p50_ms"] == result["p95_ms"] == 5
    else:
        assert result["p50_ms"] == 200
        assert result["p95_ms"] == pytest.approx(920)


@pytest.mark.parametrize("invalid", [-1, inf, nan])
def test_invalid_latencies_are_rejected(invalid):
    with pytest.raises(ValueError):
        latency_summary([invalid])


@pytest.mark.parametrize("cutoffs", [[], [0], [-1], [1, 1]])
def test_invalid_cutoffs_are_rejected(cutoffs):
    with pytest.raises(ValidationError):
        EvaluationDataset(
            dataset_id="test",
            asr=[AsrCase(id="clip", reference="a", hypothesis="a")],
            retrieval_k=cutoffs,
        )


def test_cli_evaluates_without_network_or_model_loading(tmp_path, monkeypatch, capsys):
    from lmaana_assistant.cli import main

    dataset = tmp_path / "annotations.json"
    dataset.write_text(
        json.dumps(
            {"dataset_id": "cli", "asr": [{"id": "clip", "reference": "a", "hypothesis": "a b c"}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["lmaana", "evaluate", str(dataset)])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: pytest.fail("Network used"))
    monkeypatch.setattr(
        "lmaana_assistant.cli.make_embedder", lambda *args: pytest.fail("Model used")
    )
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["asr"]["wer"]["error_rate"] == 2


def test_cli_rejects_invalid_dataset_and_saves_valid_report(tmp_path, monkeypatch, capsys):
    from lmaana_assistant.cli import main

    dataset, output = tmp_path / "dataset.json", tmp_path / "reports" / "metrics.json"
    monkeypatch.setattr("sys.argv", ["lmaana", "evaluate", str(dataset), "--output", str(output)])
    dataset.write_text('{"dataset_id":"empty"}', encoding="utf-8")
    assert main() == 1
    assert not output.exists()
    capsys.readouterr()
    dataset.write_text(
        json.dumps(
            {"dataset_id": "valid", "asr": [{"id": "clip", "reference": "a", "hypothesis": "a"}]}
        ),
        encoding="utf-8",
    )
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["dataset_id"] == "valid"


def test_cli_preserves_annotations_if_output_is_the_input_file(tmp_path, monkeypatch, capsys):
    from lmaana_assistant.cli import main

    dataset = tmp_path / "annotations.json"
    original = json.dumps(
        {"dataset_id": "preserved", "asr": [{"id": "clip", "reference": "a", "hypothesis": "a"}]}
    )
    dataset.write_text(original, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["lmaana", "evaluate", str(dataset), "--output", str(dataset)])
    assert main() == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert dataset.read_text(encoding="utf-8") == original
