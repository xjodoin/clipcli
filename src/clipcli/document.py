"""Ground planning in production documents (run of show, briefs) and mine their assets.

A .docx is a zip archive: paragraph text lives in word/document.xml and any
embedded visuals (logos, banners) in word/media/ — both are extracted with the
standard library, no new dependencies.
"""

from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DOCUMENT_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")


@dataclass
class DocumentContext:
    source: Path
    text: str
    images: list[Path] = field(default_factory=list)

    def as_prompt_block(self, *, max_chars: int = 12_000) -> str:
        body = self.text[:max_chars]
        lines = [
            f"Production document ({self.source.name}) — authoritative for names, "
            "titles, partners, program order, and messaging. Prefer it over what "
            "you hear when they disagree:",
            body,
        ]
        if len(self.text) > max_chars:
            lines.append("[... document truncated ...]")
        return "\n".join(lines)


def load_document(path: Path, *, media_dir: Path | None = None) -> DocumentContext:
    """Load a production document; .docx also yields its embedded images."""
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _load_docx(path, media_dir)
    if suffix in {".txt", ".md"}:
        return DocumentContext(source=path, text=path.read_text())
    raise ValueError(f"Unsupported document type: {path.suffix}. Use .docx, .md, or .txt.")


def _load_docx(path: Path, media_dir: Path | None) -> DocumentContext:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        paragraphs = []
        for chunk in xml.split("</w:p>"):
            text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", chunk))
            if text.strip():
                paragraphs.append(html.unescape(text.strip()))
        images: list[Path] = []
        if media_dir is not None:
            media_dir.mkdir(parents=True, exist_ok=True)
            for name in sorted(archive.namelist()):
                if name.startswith("word/media/") and name.lower().endswith(DOCUMENT_IMAGE_SUFFIXES):
                    target = media_dir / Path(name).name
                    target.write_bytes(archive.read(name))
                    images.append(target)
    return DocumentContext(source=path, text="\n".join(paragraphs), images=images)
