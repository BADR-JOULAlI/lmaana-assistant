import json

import httpx
import pytest

from lmaana_assistant.contracts import GeneratedAnswer, Statement
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout, InvalidGeneration
from lmaana_assistant.generation.adapters import LlamaCppGenerator, validate_citations
from lmaana_assistant.normalization import process_query


def test_generator_trims_context_and_sends_counted_tokens(settings, evidence):
    templates = []
    completions = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            data = json.loads(body["messages"][1]["content"])
            templates.append(data["reference_passages"])
            return httpx.Response(200, json={"prompt": str(len(templates[-1]))})
        if request.url.path == "/tokenize":
            return httpx.Response(
                200, json={"tokens": [42] * (4000 if body["content"] == "2" else 100)}
            )
        if request.url.path == "/completion":
            completions.append(body)
            return httpx.Response(
                200,
                json={
                    "content": json.dumps(
                        {
                            "outcome": "answered",
                            "statements": [{"text": "ها المعلومات.", "citation_ids": ["chunk-0"]}],
                        }
                    ),
                    "stop_type": "eos",
                },
            )
        raise AssertionError(request.url)

    client = httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler))
    generator = LlamaCppGenerator(settings, client)
    try:
        result = generator.generate(process_query("المشروع"), evidence)
        assert result.outcome == "answered"
        assert [len(items) for items in templates] == [2, 1]
        assert completions[0]["prompt"] == [42] * 100
        assert completions[0]["n_predict"] == settings.max_output_tokens
        assert completions[0]["json_schema"]["additionalProperties"] is False
    finally:
        generator.close()


@pytest.mark.parametrize("ids", [[], ["invented"], ["chunk-0", "invented"]])
def test_citation_validator_rejects_missing_or_unknown_references(ids, evidence):
    answer = GeneratedAnswer(
        outcome="answered", statements=[Statement(text="Claim", citation_ids=ids)]
    )
    with pytest.raises(InvalidGeneration):
        validate_citations(answer, evidence)


def test_readiness_checks_model_and_context(settings):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.llm_model}]})
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 2048}})

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        assert LlamaCppGenerator(settings, client).ready() is False


@pytest.mark.parametrize("kind", ["timeout", "offline", "malformed"])
def test_generator_distinguishes_operational_errors(settings, evidence, kind):
    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if kind == "offline":
            return httpx.Response(503)
        return httpx.Response(200, json={"wrong": "shape"})

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        generator = LlamaCppGenerator(settings, client)
        error = InferenceTimeout if kind == "timeout" else DependencyUnavailable
        with pytest.raises(error):
            generator.generate(process_query("المشروع"), evidence)


@pytest.mark.parametrize(
    ("question", "code", "answer_text"),
    [
        ("Quels documents pour le projet ?", "fr", "Vous devez fournir les documents du projet."),
        ("ما هي وثائق المشروع؟", "ar", "يجب تقديم وثائق المشروع."),
        ("شنو خاصني ديال المشروع؟", "ary", "خاصك الوثائق ديال المشروع."),
        ("chno khassni dyal projet?", "ary-Latn", "Khassk lwra9 dyal projet."),
        ("What documents for the project?", "en", "You need the project documents."),
    ],
)
def test_generator_receives_language_instruction_and_disabled_thinking(
    settings, evidence, question, code, answer_text
):
    from lmaana_assistant.language import INSTRUCTIONS

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            assert INSTRUCTIONS[code] in body["messages"][0]["content"]
            assert body["chat_template_kwargs"]["enable_thinking"] is False
            assert json.loads(body["messages"][1]["content"])["question"] == question
            return httpx.Response(200, json={"prompt": "test"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2]})
        assert "source_excerpts" not in body["json_schema"]["properties"]["outcome"]["enum"]
        return httpx.Response(
            200,
            json={
                "content": json.dumps(
                    {
                        "outcome": "answered",
                        "statements": [{"text": answer_text, "citation_ids": ["chunk-0"]}],
                    }
                ),
                "stop_type": "eos",
            },
        )

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        result = LlamaCppGenerator(settings, client).generate(process_query(question), evidence)
        assert result.statements[0].text == answer_text


def test_generator_rejects_answer_in_wrong_language(settings, evidence):
    def handler(request):
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "test"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={
                "content": json.dumps(
                    {
                        "outcome": "answered",
                        "statements": [
                            {"text": "يجب تقديم وثائق المشروع.", "citation_ids": ["chunk-0"]}
                        ],
                    }
                )
            },
        )

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InvalidGeneration, match="requested language"):
            LlamaCppGenerator(settings, client).generate(
                process_query("Quels documents pour le projet ?"), evidence
            )
