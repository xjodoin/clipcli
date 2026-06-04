import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipcli import assets, ffmpeg, promo
from clipcli.gemini import parse_promo_json
from clipcli.models import PromoAsset, PromoScene
from clipcli.promo import PromoOptions, _render_scene_segment, _resolve_asset, _TimedScene


def _fake_response(content: bytes, content_type: str = "image/png", status_code: int = 200):
    return SimpleNamespace(
        status_code=status_code,
        content=content,
        headers={"content-type": content_type},
    )


def test_fetch_logo_builds_clearbit_url(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return _fake_response(b"PNG")

    monkeypatch.setattr(assets.httpx, "get", fake_get)
    output = assets.fetch_logo(" NIMIntelliance.CA ", tmp_path / "logo.png")

    assert captured["url"] == "https://logo.clearbit.com/nimintelliance.ca?size=512&format=png"
    assert output.read_bytes() == b"PNG"


def test_fetch_image_url_rejects_non_images(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        assets.httpx, "get", lambda url, **kwargs: _fake_response(b"<html>", "text/html")
    )
    with pytest.raises(assets.AssetError, match="Not an image"):
        assets.fetch_image_url("https://example.com/x", tmp_path / "x.png")


def test_fetch_image_url_rejects_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        assets.httpx, "get", lambda url, **kwargs: _fake_response(b"", status_code=404)
    )
    with pytest.raises(assets.AssetError, match="404"):
        assets.fetch_image_url("https://example.com/missing.png", tmp_path / "x.png")


def test_generate_image_writes_inline_data(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            part = SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png", data=b"IMG"))
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
            )

    from google import genai

    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(models=FakeModels()))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    output = assets.generate_image("isometric diagram of connected care", tmp_path / "gen.png")

    assert captured["model"] == assets.DEFAULT_IMAGE_MODEL
    assert "isometric diagram" in captured["contents"]
    assert output.read_bytes() == b"IMG"


def test_resolve_asset_caches_and_routes(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_logo(domain, output):
        calls.append(("logo", domain))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"PNG")
        return output

    monkeypatch.setattr(promo.assets, "fetch_logo", fake_logo)
    asset = PromoAsset(kind="logo", value="greybox.ca")

    first = _resolve_asset(asset, 3, tmp_path)
    second = _resolve_asset(asset, 3, tmp_path)  # cached: no second fetch

    assert first == second == tmp_path / "assets" / "03.png"
    assert calls == [("logo", "greybox.ca")]


def test_render_scene_segment_uses_image_for_asset_scenes(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_image_segment(image, output, *, duration, mode, fit, card_color="0x0E1320"):
        captured["image"] = image
        captured["fit"] = fit
        captured["card_color"] = card_color
        captured["duration"] = duration
        return output

    def forbid_montage(*args, **kwargs):
        raise AssertionError("asset scenes must not cut source footage")

    monkeypatch.setattr(ffmpeg, "render_image_segment", fake_image_segment)
    monkeypatch.setattr(ffmpeg, "render_montage_segment", forbid_montage)
    monkeypatch.setattr(
        promo, "_resolve_asset", lambda asset, index, work_dir: tmp_path / "assets" / "01.png"
    )

    scene = PromoScene(start=0.0, end=5.0, vo="x", asset=PromoAsset(kind="logo", value="greybox.ca"))
    item = _TimedScene(scene=scene, start=0.0, duration=5.0)
    options = PromoOptions(source=tmp_path / "in.mp4", output_dir=tmp_path)

    _render_scene_segment(item, 1, tmp_path / "in.mp4", tmp_path / "work", options)

    assert captured["fit"] == "card"
    assert captured["duration"] == 5.0


def test_parse_promo_json_keeps_asset_scenes_out_of_range(tmp_path: Path) -> None:
    raw = {
        "title": "X",
        "scenes": [
            {"start": 10.0, "end": 14.0, "vo": "footage"},
            # Nominal range beyond the video: fine, it doesn't read the source.
            {"start": 0.0, "end": 5.0, "vo": "card", "asset": {"kind": "logo", "value": "greybox.ca"}},
        ],
    }
    plan = parse_promo_json(json.dumps(raw), video_duration=365.0)
    assert len(plan.scenes) == 2
    assert plan.scenes[1].asset is not None
    assert plan.scenes[1].asset.kind == "logo"


def test_render_image_segment_builds_card_and_cover_commands(monkeypatch, tmp_path: Path) -> None:
    commands = []
    monkeypatch.setattr(ffmpeg, "run", lambda command, timeout=None: commands.append(command))

    ffmpeg.render_image_segment(tmp_path / "logo.png", tmp_path / "card.mp4", duration=5.0, fit="card")
    ffmpeg.render_image_segment(tmp_path / "art.png", tmp_path / "cover.mp4", duration=5.0, fit="cover")

    card, cover = commands
    assert any("overlay=(W-w)/2:(H-h)/2" in part for part in card)
    assert card[card.index("-frames:v") + 1] == "150"
    assert any("zoompan=" in part for part in cover)
    assert any("force_original_aspect_ratio=increase" in part for part in cover)
