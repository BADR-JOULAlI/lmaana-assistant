import builtins
import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from lmaana_assistant.diagnostics import inspect_environment, main


def installed_project(monkeypatch, requirements, *, python_requirement=">=3.11"):
    project = SimpleNamespace(
        version="0.2.0",
        metadata={"Requires-Python": python_requirement},
        requires=requirements,
    )
    monkeypatch.setattr("lmaana_assistant.diagnostics.metadata.distribution", lambda name: project)


def test_core_diagnostic_checks_metadata_without_importing_application_or_models(monkeypatch):
    installed_project(
        monkeypatch,
        [
            "pydantic>=2,<3",
            "streamlit>=1; extra == 'ui'",
            "sentence-transformers>=3; extra == 'models'",
            "old-package; python_version < '3.0'",
        ],
    )
    monkeypatch.setattr("lmaana_assistant.diagnostics.metadata.version", lambda name: "2.9.0")
    monkeypatch.setenv("LMAANA_CONTEXT_TOKENS", "invalid-serving-setting")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "sentence_transformers", "streamlit"}
        assert not name.startswith("lmaana_assistant.config")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report = inspect_environment()
    assert report["requirements_satisfied"] is True
    assert [row["distribution"] for row in report["requirements"]] == ["pydantic"]


def test_requested_extras_report_missing_and_incompatible_packages(monkeypatch):
    installed_project(
        monkeypatch,
        ["pydantic>=2,<3", "streamlit>=1; extra == 'ui'", "pytest>=8; extra == 'dev'"],
    )

    def installed_version(name):
        if name == "streamlit":
            raise metadata.PackageNotFoundError(name)
        return {"pydantic": "1.10.0", "pytest": "9.0.0"}[name]

    monkeypatch.setattr("lmaana_assistant.diagnostics.metadata.version", installed_version)
    report = inspect_environment(["ui", "dev", "ui"])
    assert report["requirements_satisfied"] is False
    assert report["profiles"] == ["dev", "ui"]
    assert {row["distribution"]: row["status"] for row in report["requirements"]} == {
        "pydantic": "version_mismatch",
        "streamlit": "missing",
        "pytest": "satisfied",
    }


@pytest.mark.parametrize(
    "requirement, installed, expected",
    [
        ("pydantic>=2", "broken-version", "invalid_version"),
        ("pydantic @ https://example.org/package.whl", "2.9.0", "source_not_verified"),
    ],
)
def test_unverifiable_metadata_cannot_pass(monkeypatch, requirement, installed, expected):
    installed_project(monkeypatch, [requirement])
    monkeypatch.setattr("lmaana_assistant.diagnostics.metadata.version", lambda name: installed)
    report = inspect_environment()
    assert report["requirements_satisfied"] is False
    assert report["requirements"][0]["status"] == expected


def test_unsupported_python_cannot_pass_with_all_packages_present(monkeypatch):
    installed_project(monkeypatch, [], python_requirement=">=99")
    report = inspect_environment()
    assert report["python_satisfies_requirement"] is False
    assert report["requirements_satisfied"] is False


def test_uninstalled_project_returns_a_failed_json_report(monkeypatch, capsys):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("lmaana_assistant.diagnostics.metadata.distribution", missing)
    monkeypatch.setattr("sys.argv", ["lmaana-env"])
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["requirements_satisfied"] is False
    assert "Install lmaana-assistant" in report["errors"][0]


def test_missing_checker_dependency_is_reported_without_a_traceback(monkeypatch):
    original_import = builtins.__import__

    def missing_packaging(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ModuleNotFoundError(name="packaging")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_packaging)
    report = inspect_environment()
    assert report["requirements_satisfied"] is False
    assert "packaging" in report["errors"][0]
