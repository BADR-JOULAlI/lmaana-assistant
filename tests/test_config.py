import pytest
from pydantic import ValidationError

from lmaana_assistant.config import Settings


def test_environment_selects_explicit_profile(monkeypatch):
    monkeypatch.setenv("LMAANA_GENERATOR_BACKEND", "llamacpp")
    config = Settings(_env_file=None)
    assert config.generator_backend == "llamacpp"
    assert config.embedding_backend == "lexical"


def test_invalid_context_budget_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, context_tokens=1024, max_output_tokens=1024)


def test_stale_asr_environment_cannot_select_an_older_release(monkeypatch):
    monkeypatch.setenv("LMAANA_ASR_MODEL", "sailu4/lmaana-2.1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("field", ["asr_worker_url", "ollama_url"])
@pytest.mark.parametrize("port", ["0", "-1", "65536", "invalid"])
def test_invalid_local_ports_are_rejected_during_configuration(field, port):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            generator_backend="ollama",
            **{field: f"http://127.0.0.1:{port}"},
        )


@pytest.mark.parametrize("url", ["http://localhost", "http://127.0.0.1:8001/", "http://[::1]:8001"])
def test_valid_local_addresses_support_explicit_and_default_ports(url):
    config = Settings(
        _env_file=None, generator_backend="ollama", asr_worker_url=url, ollama_url=url
    )
    assert config.asr_worker_url == config.ollama_url == url
