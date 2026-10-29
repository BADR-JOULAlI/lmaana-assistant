"""Inspect installed dependency metadata without importing application/model runtimes."""

import argparse
import json
import platform
import sys
from importlib import metadata

PROJECT = "lmaana-assistant"
PROFILES = ("core", "ui", "models", "dev")


def inspect_environment(profiles: list[str] | None = None) -> dict:
    selected = sorted(set(profiles or ["core"]))
    if set(selected) - set(PROFILES):
        raise ValueError("Unknown dependency profile.")
    report = {
        "schema_version": 1,
        "project": PROJECT,
        "project_version": None,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_requirement": None,
        "python_satisfies_requirement": None,
        "profiles": selected,
        "requirements_satisfied": False,
        "requirements": [],
        "errors": [],
        "scope": (
            "Installed metadata for direct dependencies and selected extras only. "
            "Does not verify native imports, model files, CUDA, services, or transitive "
            "dependencies. Does not read .env or contact services."
        ),
    }
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        report["errors"].append("Install the packaging dependency to inspect version requirements.")
        return report
    try:
        project = metadata.distribution(PROJECT)
    except metadata.PackageNotFoundError:
        report["errors"].append("Install lmaana-assistant in this interpreter first.")
        return report

    report["project_version"] = project.version
    report["python_requirement"] = project.metadata.get("Requires-Python")
    report["python_satisfies_requirement"] = SpecifierSet(
        report["python_requirement"] or ""
    ).contains(report["python_version"], prereleases=True)
    extras = ["", *(profile for profile in selected if profile != "core")]
    for raw in project.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in extras
        ):
            continue
        installed = None
        try:
            installed = metadata.version(requirement.name)
            Version(installed)  # Invalid metadata is not a successful dependency check.
            status = (
                "satisfied"
                if requirement.specifier.contains(installed, prereleases=True)
                else "version_mismatch"
            )
            if requirement.url:
                status = "source_not_verified"
        except metadata.PackageNotFoundError:
            status = "missing"
        except InvalidVersion:
            status = "invalid_version"
        report["requirements"].append(
            {
                "distribution": requirement.name,
                "specifier": str(requirement.specifier),
                "installed_version": installed,
                "status": status,
            }
        )
    report["requirements_satisfied"] = report["python_satisfies_requirement"] and all(
        row["status"] == "satisfied" for row in report["requirements"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=PROFILES, action="append", help="Repeat to include multiple extras."
    )
    args = parser.parse_args()
    try:
        report = inspect_environment(args.profile)
    except (OSError, ValueError) as exc:
        print(f"Invalid dependency metadata: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if report["requirements_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
