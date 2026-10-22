import json
import sys
from math import inf, nan
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable
from lmaana_assistant.evaluation.retrieval import RetrievalBenchmark, benchmark_retrieval
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder, QwenEmbedder


def fixture_data():
    return {
        "dataset_id": "ranking-arithmetic",
        "notes": "Authored software fixture, not a model-quality benchmark.",
        "retrieval_k": [1, 3],
        "passages": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}],
        "questions": [
            {"id": "positive", "question": "alpha", "language": "en", "relevant_ids": ["a"]},
            {"id": "negative", "question": "gamma", "language": "ary", "relevant_ids": []},
        ],
    }


class ControlledEmbedder:
    dimension = 2
    fingerprint = "controlled-test-vectors"

    def encode_documents(self, texts):
        return [[2.0, 0.0], [0.0, 5.0]]

    def encode_query(self, text):
        return [3.0, 0.0] if text == "alpha" else [0.0, 0.0]


def test_measured_rankings_reuse_metrics_and_separate_negative_queries():
    dataset = RetrievalBenchmark.model_validate(fixture_data())
    report = benchmark_retrieval(dataset, ControlledEmbedder(), query_mode="raw", min_score=0.5)
    assert report["retrieval"]["at_k"]["1"]["mean_recall"] == 1
    assert report["retrieval"]["at_k"]["3"]["mean_precision"] == pytest.approx(1 / 3)
    assert report["retrieval"]["negative_query_nonempty_result_rate"] == 0
    assert report["by_language"]["ary"]["retrieval"]["at_k"]["1"]["mrr"] is None
    assert report["cases"][0]["ranked"] == [{"id": "a", "score": 1}, {"id": "b", "score": 0}]
    assert report["cases"][0]["accepted"] == [{"id": "a", "score": 1}]
    assert report["cases"][1]["ranked"] == []
    assert report["query_latency"]["sample_count"] == 2
    assert report["dimension"] == 2
    assert len(report["dataset_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_threshold_does_not_hide_the_original_ranking():
    class NearMatch(ControlledEmbedder):
        def encode_query(self, text):
            return [1.0, 1.0]

    report = benchmark_retrieval(
        RetrievalBenchmark.model_validate(fixture_data()), NearMatch(), min_score=0.9
    )
    assert report["retrieval"]["at_k"]["1"]["mrr"] == 0
    assert report["unfiltered_retrieval"]["at_k"]["1"]["mrr"] == 1
    assert [item["id"] for item in report["cases"][0]["ranked"]] == ["a", "b"]
    assert report["cases"][0]["accepted"] == []


def test_ties_are_independent_of_passage_order_and_dataset_identity_is_recorded():
    data = fixture_data()
    first = benchmark_retrieval(RetrievalBenchmark.model_validate(data), LexicalEmbedder())
    data["passages"].reverse()
    second = benchmark_retrieval(RetrievalBenchmark.model_validate(data), LexicalEmbedder())
    assert first["cases"][1]["ranked"] == second["cases"][1]["ranked"]
    assert first["dataset_sha256"] != second["dataset_sha256"]


def test_expansion_is_explicit_and_original_questions_are_preserved():
    data = fixture_data()
    data["questions"][0]["question"] = "شــنو الوثــائق؟"
    dataset = RetrievalBenchmark.model_validate(data)
    expanded = benchmark_retrieval(dataset, LexicalEmbedder())
    raw = benchmark_retrieval(dataset, LexicalEmbedder(), query_mode="raw")
    assert raw["cases"][0]["query_text"] == "شــنو الوثــائق؟"
    assert expanded["cases"][0]["question"] == "شــنو الوثــائق؟"
    assert "documents pièces" in expanded["cases"][0]["query_text"]
    assert expanded["dataset_sha256"] == raw["dataset_sha256"]
    assert expanded["query_mode"] != raw["query_mode"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("passages", []),
        ("questions", []),
        ("notes", " "),
        ("dataset_id", " "),
        ("retrieval_k", []),
        ("retrieval_k", [0]),
        ("retrieval_k", [1, 1]),
    ],
)
def test_invalid_dataset_headers_are_rejected(field, value):
    with pytest.raises(ValidationError):
        RetrievalBenchmark.model_validate({**fixture_data(), field: value})


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate-passage",
        "duplicate-question",
        "missing-id",
        "blank-text",
        "duplicate-reference",
        "blank-reference",
        "blank-question",
        "unknown-field",
    ],
)
def test_invalid_or_ambiguous_annotations_are_rejected(problem):
    data = fixture_data()
    if problem == "duplicate-passage":
        data["passages"].append(data["passages"][0])
    elif problem == "duplicate-question":
        data["questions"].append(data["questions"][0])
    elif problem == "missing-id":
        data["questions"][0]["relevant_ids"] = ["missing"]
    elif problem == "blank-text":
        data["passages"][0]["text"] = " "
    elif problem == "duplicate-reference":
        data["questions"][0]["relevant_ids"] = ["a", "a"]
    elif problem == "blank-reference":
        data["questions"][0]["relevant_ids"] = [" "]
    elif problem == "blank-question":
        data["questions"][0]["question"] = " "
    else:
        data["questions"][0]["invented_score"] = 1
    with pytest.raises(ValidationError):
        RetrievalBenchmark.model_validate(data)


@pytest.mark.parametrize("threshold", [-1, 1.01, inf, nan])
def test_invalid_threshold_is_rejected_before_encoding(threshold):
    class MustNotEncode(ControlledEmbedder):
        def encode_documents(self, texts):
            pytest.fail("Invalid settings must not invoke the model")

    with pytest.raises(ValueError, match="threshold"):
        benchmark_retrieval(
            RetrievalBenchmark.model_validate(fixture_data()), MustNotEncode(), min_score=threshold
        )


@pytest.mark.parametrize(
    "problem", ["count", "document-shape", "query-shape", "document-nan", "query-infinity"]
)
def test_malformed_embeddings_fail_instead_of_producing_a_quality_score(problem):
    class BadEmbedder(ControlledEmbedder):
        def encode_documents(self, texts):
            if problem == "count":
                return [[1.0, 0.0]]
            if problem == "document-shape":
                return [[1.0], [0.0, 1.0]]
            if problem == "document-nan":
                return [[nan, 0.0], [0.0, 1.0]]
            return super().encode_documents(texts)

        def encode_query(self, text):
            return [1.0] if problem == "query-shape" else [inf, 0.0]

    with pytest.raises(ValueError, match="Embedding"):
        benchmark_retrieval(RetrievalBenchmark.model_validate(fixture_data()), BadEmbedder())


def test_zero_document_vectors_are_not_arbitrarily_ranked():
    class ZeroDocument(ControlledEmbedder):
        def encode_documents(self, texts):
            return [[0.0, 0.0], [1.0, 0.0]]

    result = benchmark_retrieval(RetrievalBenchmark.model_validate(fixture_data()), ZeroDocument())
    assert result["zero_document_vector_count"] == 1
    assert result["cases"][0]["ranked"] == [{"id": "b", "score": 1}]


def write_dataset(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(fixture_data()), encoding="utf-8")
    return path


def test_cli_lexical_run_is_independent_of_serving_config_and_network(
    tmp_path, monkeypatch, capsys
):
    from lmaana_assistant.cli import main

    path = write_dataset(tmp_path)
    monkeypatch.setenv("LMAANA_ASR_MODEL", "invalid-serving-profile")
    monkeypatch.setattr("sys.argv", ["lmaana", "benchmark-retrieval", str(path)])
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("Network accessed"))
    monkeypatch.setattr(
        "lmaana_assistant.cli.make_embedder", lambda *a: pytest.fail("Model loaded")
    )
    monkeypatch.setattr(
        "lmaana_assistant.retrieval.store.VectorStore",
        lambda *a: pytest.fail("Active index opened"),
    )
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["embedding"] == LexicalEmbedder.fingerprint
    assert result["model_download_allowed"] is False
    assert result["model_load_ms"] >= 0


def test_cli_writes_report_and_cannot_overwrite_ground_truth(tmp_path, monkeypatch, capsys):
    from lmaana_assistant.cli import main

    path = write_dataset(tmp_path)
    original = path.read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["lmaana", "benchmark-retrieval", str(path), "--output", str(path)]
    )
    assert main() == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert path.read_bytes() == original
    report = tmp_path / "reports" / "retrieval.json"
    monkeypatch.setattr(
        "sys.argv", ["lmaana", "benchmark-retrieval", str(path), "--output", str(report)]
    )
    assert main() == 0
    assert json.loads(report.read_text(encoding="utf-8"))["dataset_id"] == "ranking-arithmetic"


@pytest.mark.parametrize("invalid", ["nan", "-0.1", "1.2"])
def test_invalid_cli_threshold_cannot_trigger_an_explicit_download(
    tmp_path, monkeypatch, capsys, invalid
):
    from lmaana_assistant.cli import main

    path = write_dataset(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lmaana",
            "benchmark-retrieval",
            str(path),
            "--backend",
            "qwen",
            "--revision",
            "a" * 40,
            "--allow-download",
            "--min-score",
            invalid,
        ],
    )
    monkeypatch.setattr(
        "lmaana_assistant.cli.make_embedder", lambda *a: pytest.fail("Model loaded")
    )
    assert main() == 1
    assert "threshold" in capsys.readouterr().err


def test_invalid_annotations_cannot_load_a_model(tmp_path, monkeypatch, capsys):
    from lmaana_assistant.cli import main

    path = write_dataset(tmp_path)
    data = fixture_data()
    data["questions"][0]["relevant_ids"] = ["not-in-corpus"]
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["lmaana", "benchmark-retrieval", str(path), "--backend", "qwen"]
    )
    monkeypatch.setattr(
        "lmaana_assistant.cli.make_embedder", lambda *a: pytest.fail("Model loaded")
    )
    assert main() == 1
    assert "Unknown relevant passage" in capsys.readouterr().err


@pytest.mark.parametrize("allow_download", [False, True])
def test_qwen_cli_explicitly_controls_model_downloads(
    tmp_path, monkeypatch, capsys, allow_download
):
    from lmaana_assistant.cli import main

    path = write_dataset(tmp_path)
    revision = "a" * 40
    arguments = [
        "lmaana",
        "benchmark-retrieval",
        str(path),
        "--backend",
        "qwen",
        "--revision",
        revision,
    ]
    if allow_download:
        arguments.append("--allow-download")
    monkeypatch.setattr("sys.argv", arguments)
    # A serving-only environment must not select another checkpoint or prevent
    # the isolated Qwen experiment from starting.
    monkeypatch.setenv("LMAANA_ASR_MODEL", "invalid-serving-profile")
    monkeypatch.setenv("LMAANA_EMBEDDING_MODEL", "unrelated-serving-model")
    monkeypatch.setenv("LMAANA_CONTEXT_TOKENS", "invalid-serving-budget")

    def fake_factory(settings):
        assert settings.embedding_revision == revision
        assert settings.embedding_device == "cpu"
        assert settings.embedding_local_files_only is not allow_download
        assert settings.asr_backend == "unavailable"
        assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
        return ControlledEmbedder()

    monkeypatch.setattr("lmaana_assistant.cli.make_embedder", fake_factory)
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["model_download_allowed"] is allow_download


@pytest.mark.parametrize("local_only", [False, True])
def test_qwen_adapter_passes_pin_and_local_cache_policy(monkeypatch, local_only):
    calls = []

    def fake_loader(model, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=fake_loader)
    )
    settings = Settings(
        _env_file=None, embedding_revision="b" * 40, embedding_local_files_only=local_only
    )
    embedder = QwenEmbedder(settings)
    assert calls == [
        (
            settings.embedding_model,
            {
                "revision": "b" * 40,
                "device": "cpu",
                "trust_remote_code": False,
                "local_files_only": local_only,
            },
        )
    ]
    assert "instruction-v1" in embedder.fingerprint


def test_missing_cached_qwen_fails_without_lexical_fallback(monkeypatch):
    def missing(*args, **kwargs):
        assert kwargs["local_files_only"]
        raise OSError("Missing cached revision")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing)
    )
    with pytest.raises(DependencyUnavailable, match="cached locally"):
        QwenEmbedder(
            Settings(_env_file=None, embedding_revision="b" * 40, embedding_local_files_only=True)
        )


def test_qwen_keeps_instruction_on_queries_only_and_requests_normalized_vectors(monkeypatch):
    calls = []

    class Array(list):
        def tolist(self):
            return list(self)

    class Model:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return Array([Array([1.0] + [0.0] * 1023) for _ in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: Model()),
    )
    embedder = QwenEmbedder(Settings(_env_file=None, embedding_revision="c" * 40))
    assert len(embedder.encode_documents(["Un passage."])[0]) == 1024
    assert len(embedder.encode_query("شنو الوثائق؟")) == 1024
    assert calls == [
        (["Un passage."], {"batch_size": 8, "normalize_embeddings": True}),
        (
            [f"Instruct: {embedder.instruction}\nQuery: شنو الوثائق؟"],
            {"normalize_embeddings": True},
        ),
    ]


@pytest.mark.parametrize("revision", ["", "main", "a" * 39, "g" * 40])
def test_qwen_rejects_unpinned_or_invalid_revisions_before_loading(revision):
    with pytest.raises(DependencyUnavailable, match="full EMBEDDING_REVISION"):
        QwenEmbedder(Settings(_env_file=None, embedding_revision=revision))


def test_committed_development_fixture_has_five_languages_and_explicit_negatives():
    path = Path(__file__).resolve().parents[1] / "configs/retrieval-benchmark.example.json"
    dataset = RetrievalBenchmark.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(dataset.passages) == 8
    assert len(dataset.questions) == 40
    assert {question.language for question in dataset.questions} == {
        "fr",
        "ar",
        "ary",
        "ary-Latn",
        "en",
    }
    assert sum(not question.relevant_ids for question in dataset.questions) == 5
