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
