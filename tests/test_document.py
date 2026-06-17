import zipfile
from pathlib import Path

import pytest

from clipcli import promo
from clipcli.document import DocumentContext, load_document
from clipcli.gemini import _build_promo_prompt
from clipcli.models import PromoAsset, PromoScene
from clipcli.promo import _resolve_asset


def _make_docx(path: Path, *, with_image: bool = True) -> Path:
    document_xml = (
        '<?xml version="1.0"?><w:document xmlns:w="http://x">'
        "<w:body>"
        "<w:p><w:r><w:t>RUN OF SHOW</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>16h05 | Pierre B&amp;eacute;... </w:t></w:r>"
        "<w:r><w:t>(Greybox)</w:t></w:r></w:p>"
        "<w:p><w:r><w:t xml:space=\"preserve\">  </w:t></w:r></w:p>"  # whitespace-only: dropped
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        if with_image:
            archive.writestr("word/media/image1.png", b"PNG-BYTES")
            archive.writestr("word/media/diagram.emf", b"EMF")  # unsupported format: skipped
    return path


def test_load_docx_extracts_text_and_images(tmp_path: Path) -> None:
    docx = _make_docx(tmp_path / "ros.docx")

    doc = load_document(docx, media_dir=tmp_path / "media")

    assert "RUN OF SHOW" in doc.text
    # Runs of the same paragraph join without separators; paragraphs join with newlines.
    assert "(Greybox)" in doc.text
    assert len(doc.text.splitlines()) == 2
    assert [image.name for image in doc.images] == ["image1.png"]
    assert doc.images[0].read_bytes() == b"PNG-BYTES"


def test_load_docx_without_media_dir_skips_images(tmp_path: Path) -> None:
    docx = _make_docx(tmp_path / "ros.docx")
    doc = load_document(docx)
    assert doc.images == []


def test_load_document_rejects_unknown_types(tmp_path: Path) -> None:
    pdf = tmp_path / "brief.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Unsupported document"):
        load_document(pdf)


def test_load_document_reads_plain_text(tmp_path: Path) -> None:
    brief = tmp_path / "brief.md"
    brief.write_text("# Brief\nLancement HSC")
    doc = load_document(brief)
    assert "Lancement HSC" in doc.text


def test_prompt_block_truncates_long_documents(tmp_path: Path) -> None:
    doc = DocumentContext(source=tmp_path / "ros.docx", text="x" * 20_000)
    block = doc.as_prompt_block(max_chars=100)
    assert "truncated" in block
    assert "authoritative" in block


def test_promo_prompt_includes_document_and_file_assets(tmp_path: Path) -> None:
    image = tmp_path / "media" / "image1.png"
    doc = DocumentContext(source=tmp_path / "ros.docx", text="16h32 COUPURE DE RUBAN", images=[image])

    prompt = _build_promo_prompt(
        None, duration=60.0, scenes=8, language="fr-CA", with_video=False, document=doc
    )

    assert "COUPURE DE RUBAN" in prompt
    assert 'asset kind "file"' in prompt
    assert str(image) in prompt


def test_resolve_asset_copies_local_files(tmp_path: Path) -> None:
    source = tmp_path / "logo.png"
    source.write_bytes(b"PNG")
    asset = PromoAsset(kind="file", value=str(source))

    output = _resolve_asset(asset, 2, tmp_path / "work")

    assert output == tmp_path / "work" / "assets" / "02.png"
    assert output.read_bytes() == b"PNG"


def test_resolve_asset_missing_file_raises(tmp_path: Path) -> None:
    asset = PromoAsset(kind="file", value=str(tmp_path / "absent.png"))
    with pytest.raises(FileNotFoundError):
        _resolve_asset(asset, 1, tmp_path / "work")


def test_resolve_asset_normalizes_gif_to_png_frame(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "logo.gif"
    source.write_bytes(b"GIF89a")
    captured = {}

    def fake_extract_frame(src, output, at, **kwargs):
        captured["src"] = src
        captured["at"] = at
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"PNG")
        return output

    def forbid_copy(*args, **kwargs):
        raise AssertionError("animated formats must be normalized, not copied verbatim")

    monkeypatch.setattr(promo.ffmpeg, "extract_frame", fake_extract_frame)
    monkeypatch.setattr(promo.shutil, "copyfile", forbid_copy)

    asset = PromoAsset(kind="file", value=str(source))
    output = _resolve_asset(asset, 7, tmp_path / "work")

    # A logo GIF is decoded to a still PNG frame, not copied with its .gif suffix.
    assert output == tmp_path / "work" / "assets" / "07.png"
    assert captured["src"] == source.resolve()
    assert captured["at"] == 0.0


def test_promo_scene_accepts_fit_and_card_color() -> None:
    scene = PromoScene(
        start=0.0,
        end=5.0,
        asset=PromoAsset(kind="file", value="/x/logo.png", fit="card", card_color="0xFFFFFF"),
    )
    assert scene.asset.fit == "card"
    assert scene.asset.card_color == "0xFFFFFF"
