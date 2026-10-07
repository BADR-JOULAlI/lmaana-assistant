"""Authored software fixtures, not a certification of administrative rules."""

from types import SimpleNamespace

import pytest

from lmaana_assistant.contracts import Source
from lmaana_assistant.ingestion import sources
from lmaana_assistant.ingestion.sources import extract_pdf_section, extract_sections, load_chunks
from lmaana_assistant.retrieval.embeddings import LexicalEmbedder


def section_source(**changes):
    values = dict(
        id="test-section",
        title="Software test guide",
        publisher="Test fixture",
        url="https://example.org/guide.pdf",
        kind="pdf",
        pages=[2, 3],
        reviewed=True,
        pdf_section=dict(
            start_marker="Section: project documents",
            end_marker="Section: unrelated",
            running_header="TEST GUIDE",
            printed_page_offset=-1,
        ),
    )
    return Source.model_validate(values | changes)


def page_texts():
    return [
        (
            "page 2",
            "TEST GUIDE\n\n1\nPrevious topic.\nSection: project\ndocuments\n"
            "Documents: send the request to the\n",
        ),
        (
            "page 3",
            "TEST GUIDE\n\n2\nproject owner, with the project reference,\n"
            "or the test ID for a demo, within 3 days.\nSection: unrelated\nOther text.",
        ),
    ]


def test_cross_page_section_keeps_conditions_and_provenance():
    location, text = extract_pdf_section(section_source(), page_texts())
    assert location.startswith("pages 2–3")
    assert "to the\nproject owner" in text
    assert "or the test ID for a demo, within 3 days." in text
    assert "TEST GUIDE" not in text
    assert "Previous topic" not in text and "Other text" not in text
    assert "Section: project\ndocuments" in text  # no rewritten body characters


@pytest.mark.parametrize("change", ["header", "folio", "start", "end", "duplicate", "order"])
def test_pdf_layout_or_boundary_drift_fails_closed(change):
    pages = page_texts()
    replacements = {
        "header": (0, "TEST GUIDE", "NEW HEADER"),
        "folio": (1, "\n2\n", "\n9\n"),
        "start": (0, "Section: project", "Changed project"),
        "end": (1, "Section: unrelated", "Changed end"),
        "duplicate": (1, "Other text.", "Section: unrelated"),
        "order": (0, "Previous topic.", "Section: unrelated"),
    }
    index, before, after = replacements[change]
    pages[index] = (pages[index][0], pages[index][1].replace(before, after))
    if change == "order":
        pages[1] = (pages[1][0], pages[1][1].replace("Section: unrelated", "Other heading"))
    with pytest.raises(ValueError):
        extract_pdf_section(section_source(), pages)


def test_identical_numbers_in_body_are_not_deleted():
    pages = page_texts()
    pages[1] = (pages[1][0], pages[1][1].replace("within 3 days.", "within\n2\ndays."))
    _, text = extract_pdf_section(section_source(), pages)
    assert "within\n2\ndays." in text


@pytest.mark.parametrize("pages", [[3, 2], [2, 4], [2, 2], None, []])
def test_pdf_sections_require_consecutive_explicit_pages(pages):
    with pytest.raises(ValueError):
        section_source(pages=pages)


@pytest.mark.parametrize(
    "changes", [{"kind": "html"}, {"selector": "main"}, {"section_heading": "Other section"}]
)
def test_pdf_sections_cannot_mix_extraction_modes(changes):
    with pytest.raises(ValueError):
        section_source(**changes)


def mock_pdf(monkeypatch, pages):
    reader = SimpleNamespace(
        pages=[
            SimpleNamespace(extract_text=lambda: "Cover"),
            *[SimpleNamespace(extract_text=lambda text=text: text) for _, text in pages],
        ]
    )
    monkeypatch.setattr(sources, "PdfReader", lambda _: reader)


def test_pdf_section_dispatch(monkeypatch):
    mock_pdf(monkeypatch, page_texts())
    sections = extract_sections(section_source(), b"fixture")
    assert len(sections) == 1
    assert "to the\nproject owner" in sections[0][1]


def test_long_reviewed_section_is_not_split_mid_condition(monkeypatch, tmp_path):
    import json

    pages = page_texts()
    pages[0] = (pages[0][0], pages[0][1] + "software " * 400)
    mock_pdf(monkeypatch, pages)
    monkeypatch.setattr(sources, "load_bytes", lambda *args: b"fixture")
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "allowed_hosts": ["example.org"],
                "sources": [section_source().model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds one chunk"):
        load_chunks(manifest, LexicalEmbedder())
