"""Bounded retrieval and extraction with page/section provenance."""

import hashlib
import io
import ipaddress
import re
import socket
import ssl
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from uuid import NAMESPACE_URL, uuid5

import httpx
import truststore
from bs4 import BeautifulSoup
from pypdf import PdfReader

from lmaana_assistant.contracts import Chunk, Embedder, Source, SourceManifest

MAX_BYTES = 10 * 1024 * 1024
CHUNKER_VERSION = "source-boundaries-v2-pdf-sections-size350-overlap50"


def validate_url(url: str, allowed_hosts: list[str]) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Source URL must use HTTPS on an explicitly allowed host.")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Source host must resolve only to public IP addresses.")


def fetch_bytes(url: str, allowed_hosts: list[str]) -> bytes:
    with httpx.Client(
        timeout=30,
        follow_redirects=False,
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        headers={"User-Agent": "LmaanaAssistant/0.2 (research ingestion)"},
    ) as client:
        for _ in range(6):
            validate_url(url, allowed_hosts)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Source redirect has no destination.")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > MAX_BYTES:
                        raise ValueError("Source exceeds the 10 MiB ingestion limit.")
                return bytes(data)
    raise ValueError("Too many source redirects.")


def load_bytes(source: Source, manifest_dir: Path, allowed_hosts: list[str]) -> bytes:
    if source.local_path:
        candidate = (manifest_dir / source.local_path).resolve()
        if not candidate.is_relative_to(manifest_dir.resolve()):
            raise ValueError("Local source paths must stay inside the manifest directory.")
        if candidate.stat().st_size > MAX_BYTES:
            raise ValueError("Local source exceeds the 10 MiB ingestion limit.")
        return candidate.read_bytes()
    return fetch_bytes(str(source.url), allowed_hosts)


def extract_pdf_section(source: Source, pages: list[tuple[str, str]]) -> tuple[str, str]:
    """Join a reviewed cross-page section, failing closed on layout/marker drift."""
    spec = source.pdf_section
    assert spec is not None and source.pages is not None
    bodies = []
    for number, (_, text) in zip(source.pages, pages, strict=True):
        # Only discard the two reviewed leading nonblank lines, never all matching
        # numbers or headers in the body (which may carry real requirements).
        nonblank = list(re.finditer(r"[^\r\n]+", text))
        nonblank = [line for line in nonblank if line.group().strip()]
        expected_folio = str(number + spec.printed_page_offset)
        if (
            len(nonblank) < 3
            or " ".join(nonblank[0].group().split()) != " ".join(spec.running_header.split())
            or nonblank[1].group().strip() != expected_folio
        ):
            raise ValueError(f"Source {source.id} PDF header/folio needs extraction review.")
        bodies.append(text[nonblank[1].end() :].strip())
    joined = "\n".join(bodies)

    def unique_marker(marker: str):
        pattern = r"\s+".join(re.escape(word) for word in marker.split())
        matches = list(re.finditer(pattern, joined))
        if len(matches) != 1:
            raise ValueError(f"Source {source.id} PDF section marker is missing or ambiguous.")
        return matches[0]

    start, end = unique_marker(spec.start_marker), unique_marker(spec.end_marker)
    if end.start() <= start.end():
        raise ValueError(f"Source {source.id} PDF section markers are out of order.")
    location = f"pages {source.pages[0]}–{source.pages[-1]} · {spec.start_marker}"
    return location, joined[start.start() : end.start()].strip()


def extract_sections(source: Source, data: bytes) -> list[tuple[str, str]]:
    if source.kind == "pdf":
        reader = PdfReader(io.BytesIO(data))
        selected = source.pages or list(range(1, len(reader.pages) + 1))
        if any(number > len(reader.pages) for number in selected):
            raise ValueError(f"Source {source.id} references a nonexistent PDF page.")
        sections = [
            (f"page {number}", reader.pages[number - 1].extract_text() or "") for number in selected
        ]
        if source.pdf_section is not None:
            sections = [extract_pdf_section(source, sections)]
    elif source.kind == "html":
        soup = BeautifulSoup(data, "html.parser")
        for node in soup.select("script, style, nav, header, footer, noscript"):
            node.decompose()
        root = (
            soup.select_one(source.selector)
            if source.selector
            else soup.find("main") or soup.find("article") or soup.body
        )
        if root is None:
            raise ValueError(f"No content matched the selector for {source.id}.")
        sections = []
        heading = source.title
        lines: list[str] = []
        for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
            if node.name.startswith("h"):
                if lines:
                    sections.append((heading, "\n".join(lines)))
                    lines = []
                heading = node.get_text(" ", strip=True)
            elif not node.find(["p", "li", "tr"]):
                lines.append(node.get_text(" ", strip=True))
        if lines:
            sections.append((heading, "\n".join(lines)))
        if not sections:
            sections = [(source.title, root.get_text("\n", strip=True))]
    else:
        sections = [(source.title, data.decode("utf-8-sig"))]
    if source.section_heading is not None:
        sections = [
            (location, text) for location, text in sections if location == source.section_heading
        ]
    # Reject rather than silently indexing an empty or clearly damaged extraction.
    sections = [(location, text.strip()) for location, text in sections if text.strip()]
    if not sections or any("\ufffd" in text or "\x00" in text for _, text in sections):
        raise ValueError(f"Source {source.id} needs extraction review (empty or damaged text).")
    return sections


def load_chunks(manifest_path: Path, embedder: Embedder) -> tuple[SourceManifest, list[Chunk]]:
    manifest = SourceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if len({source.id for source in manifest.sources}) != len(manifest.sources):
        raise ValueError("Source IDs must be unique.")
    if any(not source.reviewed for source in manifest.sources):
        raise ValueError("Review every source and set reviewed=true before ingestion.")
    chunks: list[Chunk] = []
    for source in manifest.sources:
        data = load_bytes(source, manifest_path.parent, manifest.allowed_hosts)
        digest = hashlib.sha256(data).hexdigest()
        if (
            source.verification.status == "verified"
            and source.verification.content_sha256 != digest
        ):
            raise ValueError(f"Source {source.id} changed since verification. Review it again.")
        fetched = datetime.now(UTC)
        for section_number, (location, text) in enumerate(extract_sections(source, data)):
            parts = embedder.split(text, 350, 50)
            if source.pdf_section is not None and len(parts) != 1:
                raise ValueError(
                    f"Source {source.id} PDF section exceeds one chunk; review a smaller scope."
                )
            for index, part in enumerate(parts):
                identity = (
                    f"{source.id}:{source.url}:{digest}:{CHUNKER_VERSION}:"
                    f"{embedder.fingerprint}:{location}:{section_number}:{index}"
                )
                chunks.append(
                    Chunk(
                        id=str(uuid5(NAMESPACE_URL, identity)),
                        source=source,
                        text=part,
                        location=location,
                        content_hash=digest,
                        fetched_at=fetched,
                    )
                )
    if not chunks:
        raise ValueError("No searchable chunks were extracted.")
    return manifest, chunks
