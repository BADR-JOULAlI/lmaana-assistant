import json

import httpx
import pytest
from pydantic import ValidationError

from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout, InvalidGeneration
from lmaana_assistant.generation.adapters import make_generator
from lmaana_assistant.generation.ollama import OllamaGenerator
from lmaana_assistant.language import INSTRUCTIONS
from lmaana_assistant.normalization import process_query


def response_body(text="You need the project documents.", **overrides):
    return {
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "outcome": "answered",
                    "statements": [{"text": text, "citation_ids": ["chunk-0"]}],
                }
            ),
        },
        **overrides,
    }


@pytest.mark.parametrize(
    ("question", "language", "text"),
    [
        ("Quels documents pour le projet ?", "fr", "Vous devez fournir les documents du projet."),
        ("ما هي وثائق المشروع؟", "ar", "يجب تقديم وثائق المشروع."),
        ("شنو خاصني ديال المشروع؟", "ary", "خاصك الوثائق ديال المشروع."),
        ("chno khassni dyal projet?", "ary-Latn", "Khassk lwra9 dyal projet."),
        ("What documents for the project?", "en", "You need the project documents."),
    ],
)
def test_ollama_language_schema_and_budgets(settings, evidence, question, language, text):
    def handler(request):
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == settings.ollama_model
        assert INSTRUCTIONS[language] in payload["messages"][0]["content"]
        assert json.loads(payload["messages"][1]["content"])["question"] == question
        assert payload["format"]["additionalProperties"] is False
        assert payload["format"]["$defs"]["Statement"]["properties"]["citation_ids"]["items"][
            "enum"
        ] == ["chunk-0", "chunk-1"]
        assert "source_excerpts" not in payload["format"]["properties"]["outcome"]["enum"]
        assert all(payload[key] is False for key in ("stream", "think", "truncate", "shift"))
        assert payload["options"]["num_ctx"] == settings.context_tokens
        assert payload["options"]["num_predict"] == settings.max_output_tokens
        return httpx.Response(200, json=response_body(text))

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        assert (
            OllamaGenerator(settings, client)
            .generate(process_query(question), evidence)
            .statements[0]
            .text
            == text
        )


@pytest.mark.parametrize(
    "problem",
    ["none", "missing", "digest", "remote", "family", "context", "history", "malformed", "offline"],
)
def test_ollama_readiness(settings, problem):
    settings.ollama_model_digest = "a" * 64

    def handler(request):
        if problem == "offline":
            return httpx.Response(503)
        if problem == "malformed":
            return httpx.Response(200, json=[])
        if request.url.path == "/api/tags":
            model = {"name": settings.ollama_model, "digest": "a" * 64}
            if problem == "digest":
                model["digest"] = "b" * 64
            return httpx.Response(200, json={"models": [] if problem == "missing" else [model]})
        assert request.url.path == "/api/show"
        info = {
            "details": {"family": "qwen3"},
            "capabilities": ["completion", "thinking"],
            "model_info": {"qwen3.context_length": 32768},
        }
        if problem == "remote":
            info["remote_host"] = "https://example.org"
        elif problem == "family":
            info["details"]["family"] = "other"
        elif problem == "context":
            info["model_info"]["qwen3.context_length"] = 2048
        elif problem == "history":
            info["messages"] = [{"role": "user", "content": "hidden history"}]
        return httpx.Response(200, json=info)

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        assert OllamaGenerator(settings, client).ready() is (problem == "none")


@pytest.mark.parametrize(
    "problem",
    [
        "truncated",
        "unfinished",
        "language",
        "citations",
        "reserved",
        "json",
        "message",
        "reasoning",
        "tools",
    ],
)
def test_ollama_rejects_invalid_generation(settings, evidence, problem):
    body = response_body()
    if problem == "truncated":
        body["done_reason"] = "length"
    elif problem == "unfinished":
        body["done"] = False
    elif problem == "language":
        body = response_body("يجب تقديم وثائق المشروع.")
    elif problem in {"citations", "reserved"}:
        answer = json.loads(body["message"]["content"])
        if problem == "citations":
            answer["statements"][0]["citation_ids"] = ["invented"]
        else:
            answer["outcome"] = "verification_required"
        body["message"]["content"] = json.dumps(answer)
    elif problem == "json":
        body["message"]["content"] = "not JSON"
    elif problem == "message":
        body["message"] = []
    elif problem == "reasoning":
        body["message"]["thinking"] = "private reasoning"
    elif problem == "tools":
        body["message"]["tool_calls"] = [{"function": "web_search"}]
    with httpx.Client(
        base_url="http://local",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    ) as client:
        with pytest.raises(InvalidGeneration):
            OllamaGenerator(settings, client).generate(process_query("What documents?"), evidence)


@pytest.mark.parametrize("problem", ["timeout", "offline", "malformed", "remote", "redirect"])
def test_ollama_operational_errors(settings, evidence, problem):
    def handler(request):
        if problem == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if problem == "offline":
            return httpx.Response(503)
        if problem == "redirect":
            return httpx.Response(307, headers={"Location": "https://example.org"})
        if problem == "remote":
            return httpx.Response(200, json=response_body(remote_host="https://example.org"))
        return httpx.Response(200, json=[])

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InferenceTimeout if problem == "timeout" else DependencyUnavailable):
            OllamaGenerator(settings, client).generate(process_query("What documents?"), evidence)


def test_ollama_trims_whole_passages_before_request(settings, evidence):
    evidence[1].chunk.text = "x" * 20000

    def handler(request):
        passages = json.loads(json.loads(request.content)["messages"][1]["content"])[
            "reference_passages"
        ]
        assert [item["id"] for item in passages] == ["chunk-0"]
        return httpx.Response(200, json=response_body())

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        generator = OllamaGenerator(settings, client)
        assert generator.generate(process_query("What documents?"), evidence).outcome == "answered"
        evidence[0].chunk.text = "x" * 20000
        with pytest.raises(InvalidGeneration, match="context"):
            generator.generate(process_query("What documents?"), evidence)


@pytest.mark.parametrize(
    "url",
    [
        "https://ollama.com",
        "http://192.168.1.1:11434",
        "http://user:secret@localhost:11434",
        "http://localhost/api",
    ],
)
def test_ollama_requires_local_url(url):
    with pytest.raises(ValidationError, match="loopback"):
        Settings(_env_file=None, generator_backend="ollama", ollama_url=url)


def test_ollama_profile_factory_and_no_cloud():
    config = Settings(_env_file=None, generator_backend="ollama")
    generator = make_generator(config)
    assert isinstance(generator, OllamaGenerator)
    generator.close()
    with pytest.raises(ValidationError, match="local Ollama"):
        Settings(_env_file=None, generator_backend="ollama", ollama_model="qwen3:cloud")
