from pathlib import Path

import pytest

from clipcli import sound
from clipcli.models import ClipPlan


def test_sound_query_for_health_clip_prefers_subtle_corporate_music() -> None:
    clip = ClipPlan(
        title="Health",
        start=0,
        end=30,
        summary="Connected health access for patients at home.",
        hook="Transformation de la sante connectee",
        caption="Care at home.",
    )

    assert sound.sound_query_for_clip(clip) == "subtle corporate technology ambient music"


def test_search_freesound_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)
    monkeypatch.delenv("FREESOUND_TOKEN", raising=False)

    with pytest.raises(sound.SoundSearchError, match="FREESOUND_API_KEY"):
        sound.search_freesound("ambient", clip_duration=30)


def test_search_freesound_filters_noncommercial_results(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "id": 1,
                        "name": "blocked",
                        "username": "a",
                        "license": "https://creativecommons.org/licenses/by-nc/4.0/",
                        "duration": 20,
                        "tags": ["music"],
                        "url": "https://freesound.org/s/1/",
                        "previews": {"preview-hq-mp3": "https://example.com/1.mp3"},
                    },
                    {
                        "id": 2,
                        "name": "safe",
                        "username": "b",
                        "license": "Creative Commons 0",
                        "duration": 26,
                        "tags": ["ambient"],
                        "url": "https://freesound.org/s/2/",
                        "previews": {"preview-hq-mp3": "https://example.com/2.mp3"},
                    },
                ]
            }

    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return Response()

    monkeypatch.setattr(sound.httpx, "get", fake_get)

    results = sound.search_freesound("ambient", clip_duration=30, api_key="token")

    assert len(results) == 1
    assert results[0].title == "safe"
    assert captured["headers"]["Authorization"] == "Token token"
    assert "fields=" in captured["url"]


def test_search_freesound_filters_unsuitable_sound_beds(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "id": 1,
                        "name": "white noise",
                        "username": "a",
                        "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                        "duration": 40,
                        "tags": ["noise", "ambient"],
                        "url": "https://freesound.org/s/1/",
                        "previews": {"preview-hq-mp3": "https://example.com/1.mp3"},
                    },
                    {
                        "id": 2,
                        "name": "calm background music",
                        "username": "b",
                        "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                        "duration": 42,
                        "tags": ["background", "music", "calm"],
                        "url": "https://freesound.org/s/2/",
                        "previews": {"preview-hq-mp3": "https://example.com/2.mp3"},
                    },
                ]
            }

    monkeypatch.setattr(sound.httpx, "get", lambda *args, **kwargs: Response())

    results = sound.search_freesound("ambient", clip_duration=40, api_key="token")

    assert [candidate.title for candidate in results] == ["calm background music"]


def test_find_sound_bed_writes_metadata(monkeypatch, tmp_path: Path) -> None:
    clip = ClipPlan(
        title="x",
        start=0,
        end=20,
        summary="x",
        hook="x",
        caption="x",
    )
    candidate = sound.SoundCandidate(
        source="freesound",
        id="123",
        title="bed",
        author="artist",
        page_url="https://freesound.org/s/123/",
        preview_url="https://example.com/bed.mp3",
        license="Creative Commons 0",
        duration=20,
        tags=["music"],
        score=5,
    )
    monkeypatch.setattr(sound, "search_freesound", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(sound, "download_sound_preview", lambda candidate, output: output.write_bytes(b"mp3") or output)

    asset = sound.find_sound_bed(clip, tmp_path, query="ambient")

    assert asset.path.read_bytes() == b"mp3"
    metadata = asset.metadata_path.read_text()
    assert '"query": "ambient"' in metadata
    assert "Creative Commons 0" in metadata


def test_find_music_relaxes_query_and_duration(monkeypatch, tmp_path):
    from clipcli import sound as sound_module

    attempts = []

    def fake_search(query, *, clip_duration, max_candidates=12, min_duration=None, max_duration=None, **kwargs):
        attempts.append((query, min_duration))
        if query == "inspiring background music" and min_duration == 8.0:
            return [
                sound_module.SoundCandidate(
                    source="freesound",
                    id="42",
                    title="Uplift",
                    author="composer",
                    page_url="https://freesound.org/s/42/",
                    preview_url="https://cdn.freesound.org/42.mp3",
                    license="Creative Commons 0",
                    duration=45.0,
                    tags=["music"],
                    score=2.0,
                )
            ]
        return []

    monkeypatch.setattr(sound_module, "search_freesound", fake_search)
    monkeypatch.setattr(sound_module, "download_sound_preview", lambda candidate, output: output.write_bytes(b"x"))

    asset = sound_module.find_music(
        "upbeat corporate technology innovation",
        tmp_path,
        duration=30.0,
    )

    assert asset.candidate.id == "42"
    assert attempts[0] == ("upbeat corporate technology innovation music", 30.0)
    assert ("upbeat corporate music", 30.0) in attempts
    assert attempts[-1] == ("inspiring background music", 8.0)


def test_score_music_prefers_upbeat_over_ambient_soundscape():
    from clipcli.sound import _score_music

    upbeat = {
        "name": "Upbeat Corporate Anthem",
        "tags": ["upbeat", "corporate", "drums", "music"],
        "avg_rating": 4.0,
        "num_downloads": 1000,
    }
    ambient = {
        "name": "Tranquil Nature Soundscape",
        "tags": ["ambient", "atmospheric", "calm", "ethereal", "gentle"],
        "avg_rating": 4.0,
        "num_downloads": 1000,
    }
    upbeat_score = _score_music(upbeat, duration=120.0, clip_duration=30.0, license_name="cc-by")
    ambient_score = _score_music(ambient, duration=120.0, clip_duration=30.0, license_name="cc-by")
    assert upbeat_score > ambient_score + 1.0


def test_is_promo_music_rejects_soundscapes():
    from clipcli.sound import SoundCandidate, _is_promo_music

    def candidate(title, tags):
        return SoundCandidate(
            source="freesound",
            id="1",
            title=title,
            author="a",
            page_url="https://freesound.org/s/1/",
            preview_url="https://cdn.freesound.org/1.mp3",
            license="cc0",
            duration=60.0,
            tags=tags,
        )

    assert not _is_promo_music(candidate("Forest Nature Soundscape", ["nature", "birds"]))
    assert _is_promo_music(candidate("Uplifting Corporate Track", ["upbeat", "music"]))
