import builtins
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from lmaana_assistant.config import BenchmarkSettings
from lmaana_assistant.errors import DependencyUnavailable
from lmaana_assistant.evaluation.runtime import runtime_metadata
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder, QwenEmbedder


def test_benchmark_configuration_ignores_serving_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LMAANA_EMBEDDING_MODEL=not-the-benchmark-model\n")
    monkeypatch.setenv("LMAANA_ASR_MODEL", "invalid-serving-model")
    monkeypatch.setenv("LMAANA_CONTEXT_TOKENS", "invalid-budget")
    settings = BenchmarkSettings(embedding_revision="a" * 40)
    assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.embedding_revision == "a" * 40
    assert settings.asr_model == "Lmaana/lmaana-2.4"


def test_absent_optional_runtime_versions_stay_unknown_without_importing_torch(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("lmaana_assistant.evaluation.runtime.version", missing)
    monkeypatch.setitem(sys.modules, "torch", None)
    report = runtime_metadata(LexicalEmbedder())
    assert all(value is None for value in report["package_versions"].values())
    assert report["model_device"] is report["model_dtype"] is report["torch_threads"] is None


def test_runtime_metadata_reports_actual_parameter_device_dtype_and_thread_settings(monkeypatch):
    class Model:
        max_seq_length = 32768

        def parameters(self):
            yield SimpleNamespace(device="cpu", dtype="torch.float32")

    monkeypatch.setattr("lmaana_assistant.evaluation.runtime.version", lambda name: "test-version")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 4))
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    report = runtime_metadata(SimpleNamespace(model=Model()))
    assert report["model_device"] == "cpu"
    assert report["model_dtype"] == "torch.float32"
    assert report["model_max_seq_length"] == 32768
    assert report["torch_threads"] == 4
    assert report["thread_environment"]["OMP_NUM_THREADS"] == "4"
    assert report["package_versions"]["sentence-transformers"] == "test-version"


@pytest.mark.parametrize(
    "error, message",
    [
        (ModuleNotFoundError(name="sentence_transformers"), "Install the models extra"),
        (ModuleNotFoundError(name="pyarrow"), "runtime dependency could not import"),
        (ImportError("Application Control policy blocked a binary"), "application-control"),
    ],
)
def test_qwen_distinguishes_missing_package_from_broken_dependency(monkeypatch, error, message):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(DependencyUnavailable, match=message) as caught:
        QwenEmbedder(BenchmarkSettings(embedding_revision="a" * 40))
    assert caught.value.__cause__ is error
