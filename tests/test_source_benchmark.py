import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lmaana_assistant.contracts import SourceManifest
from lmaana_assistant.evaluation.retrieval import RetrievalBenchmark, dataset_sha256
from lmaana_assistant.evaluation.sources import SourceBenchmarkSpec, prepare_source_benchmark
from lmaana_assistant.ingestion.sources import extract_sections


@pytest.fixture
def source_benchmark(source_manifest):
    raw = (source_manifest.parent / "guide.html").read_bytes()
    spec = {
        "dataset_id": "source-probe-test",
        "notes": "Authored software fixture; no administrative requirements.",
        "annotation_status": "draft",
        "source_id": "project-guide",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "passages": [
            {"id": "registration", "start_marker": "Pour inscrire"},
            {"id": "display", "section_index": 1, "start_marker": "Le projet"},
        ],
        "questions": [
            {
                "id": "q-fr",
                "family": "docs",
                "question": "Quels documents ?",
                "language": "fr",
                "relevant_ids": ["registration"],
            },
            {
                "id": "q-ar",
                "family": "docs",
                "question": "ما الوثائق؟",
                "language": "ar",
                "relevant_ids": ["registration"],
            },
        ],
    }
    manifest = SourceManifest.model_validate_json(source_manifest.read_text(encoding="utf-8"))
    return spec, manifest, raw


def test_prepared_passages_are_exact_source_slices_with_hashes_and_family_provenance(
    source_benchmark,
):
    spec, manifest, raw = source_benchmark
    dataset, provenance = prepare_source_benchmark(SourceBenchmarkSpec(**spec), manifest, raw)
    sections = extract_sections(manifest.sources[0], raw)
    assert provenance["dataset_sha256"] == dataset_sha256(dataset)
    assert provenance["question_families"] == {"q-fr": "docs", "q-ar": "docs"}
    assert provenance["annotation_status"] == "draft"
    for passage, reference in zip(dataset.passages, provenance["passages"], strict=True):
        text = sections[reference["section_index"]][1]
        assert passage.text == text[reference["start_character"] : reference["end_character"]]
        assert reference["text_sha256"] == hashlib.sha256(passage.text.encode()).hexdigest()


def test_changed_source_fails_before_extraction(source_benchmark, monkeypatch):
    spec, manifest, raw = source_benchmark
    monkeypatch.setattr(
        "lmaana_assistant.evaluation.sources.extract_sections",
        lambda *a: pytest.fail("Changed snapshot reached extraction"),
    )
    with pytest.raises(ValueError, match="snapshot hash"):
        prepare_source_benchmark(SourceBenchmarkSpec(**spec), manifest, raw + b"changed")


def test_source_review_record_and_spec_cannot_disagree_on_snapshot(source_benchmark):
    spec, manifest, raw = source_benchmark
    manifest.sources[0].verification.content_sha256 = "0" * 64
    with pytest.raises(ValueError, match="manifest and benchmark snapshot"):
        prepare_source_benchmark(SourceBenchmarkSpec(**spec), manifest, raw)


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_source_reference_must_resolve_uniquely(source_benchmark, problem):
    spec, manifest, raw = source_benchmark
    if problem == "missing":
        spec["source_id"] = "unknown-source"
    else:
        manifest.sources.append(manifest.sources[0])
    with pytest.raises(ValueError, match="exactly one source"):
        prepare_source_benchmark(SourceBenchmarkSpec(**spec), manifest, raw)


@pytest.mark.parametrize("problem", ["missing", "ambiguous", "overlap", "reversed", "section"])
def test_source_layout_and_selection_errors_fail_closed(source_benchmark, problem):
    spec, manifest, raw = source_benchmark
    if problem == "missing":
        spec["passages"][0]["start_marker"] = "not in source"
    elif problem == "ambiguous":
        spec["passages"][0]["start_marker"] = "de"
    elif problem == "overlap":
        spec["passages"][1] = {"id": "display", "start_marker": "Pour inscrire"}
    elif problem == "reversed":
        spec["passages"][0] |= {"start_marker": "documentation", "end_marker": "Pour inscrire"}
    else:
        spec["passages"][0]["section_index"] = 99
    with pytest.raises(ValueError):
        prepare_source_benchmark(SourceBenchmarkSpec(**spec), manifest, raw)


def test_translation_family_cannot_silently_change_its_relevance_labels(source_benchmark):
    spec, _, _ = source_benchmark
    spec["questions"][1]["relevant_ids"] = []
    with pytest.raises(ValidationError, match="family must share"):
        SourceBenchmarkSpec(**spec)


def test_cli_prepares_without_network_models_index_or_serving_configuration(
    source_benchmark, source_manifest, tmp_path, monkeypatch
):
    from lmaana_assistant.cli import main

    spec, _, _ = source_benchmark
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "prepared.json"
    monkeypatch.setenv("LMAANA_ASR_MODEL", "invalid-serving-model")
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("Network accessed"))
    monkeypatch.setattr(
        "lmaana_assistant.cli.make_embedder", lambda *a: pytest.fail("Model loaded")
    )
    monkeypatch.setattr(
        "lmaana_assistant.retrieval.store.VectorStore", lambda *a: pytest.fail("Index opened")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "lmaana",
            "prepare-retrieval",
            str(specification),
            "--manifest",
            str(source_manifest),
            "--source",
            str(tmp_path / "guide.html"),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    dataset = RetrievalBenchmark.model_validate_json(output.read_text(encoding="utf-8"))
    proof = json.loads((tmp_path / "prepared.provenance.json").read_text())
    assert proof["dataset_sha256"] == dataset_sha256(dataset)


@pytest.mark.parametrize("original", ["spec.json", "sources.json", "guide.html"])
def test_cli_outputs_cannot_overwrite_any_input(
    source_benchmark, source_manifest, tmp_path, monkeypatch, original
):
    from lmaana_assistant.cli import main

    spec, _, _ = source_benchmark
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps(spec), encoding="utf-8")
    protected = tmp_path / original
    previous = protected.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "lmaana",
            "prepare-retrieval",
            str(specification),
            "--manifest",
            str(source_manifest),
            "--source",
            str(tmp_path / "guide.html"),
            "--output",
            str(protected),
        ],
    )
    assert main() == 1
    assert protected.read_bytes() == previous


def test_committed_snapshot_spec_has_balanced_languages_families_and_explicit_negatives():
    path = Path(__file__).resolve().parents[1] / "configs/retrieval-dgi-snapshot.dev.json"
    spec = SourceBenchmarkSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.annotation_status == "draft"
    assert len(spec.questions) == 50
    assert sum(not q.relevant_ids for q in spec.questions) == 15
    assert len({q.family for q in spec.questions}) == 10
    for family in {q.family for q in spec.questions}:
        assert {q.language for q in spec.questions if q.family == family} == {
            "fr",
            "ar",
            "ary",
            "ary-Latn",
            "en",
        }
