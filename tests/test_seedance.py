from clipcli.models import BrollPrompt
from clipcli.seedance import SeedDanceClient, _video_from_response


def test_video_from_ark_content_object() -> None:
    video = _video_from_response(
        {
            "id": "cgt-1",
            "status": "succeeded",
            "content": {"video_url": "https://example.com/video.mp4"},
        }
    )

    assert video.id == "cgt-1"
    assert video.url == "https://example.com/video.mp4"


def test_video_from_content_list() -> None:
    video = _video_from_response(
        {
            "data": {
                "task_id": "task-1",
                "status": "succeeded",
                "content": [{"url": "https://example.com/list.mp4"}],
            }
        }
    )

    assert video.id == "task-1"
    assert video.url == "https://example.com/list.mp4"


def test_seedance_create_video_clamps_short_duration(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {
                "id": "task-1",
                "status": "succeeded",
                "content": {"video_url": "https://example.com/video.mp4"},
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("clipcli.seedance.httpx.post", fake_post)
    client = SeedDanceClient(api_key="key", base_url="https://ark.example/api/v3")

    video = client.create_video(BrollPrompt(at=0, duration=2, prompt="cinematic cutaway"))

    assert video.url == "https://example.com/video.mp4"
    assert captured["url"] == "https://ark.example/api/v3/contents/generations/tasks"
    assert captured["json"]["duration"] == 4.0
    assert captured["json"]["model"] == "doubao-seedance-2-0-260128"
    assert captured["json"]["content"] == [{"type": "text", "text": "cinematic cutaway"}]


def test_seedance_create_video_uses_fal_key(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {"video": {"url": "https://example.com/fal.mp4"}}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return Response()

    monkeypatch.setattr("clipcli.seedance.httpx.post", fake_post)
    client = SeedDanceClient(provider="fal", api_key="fal-key")

    video = client.create_video(BrollPrompt(at=0, duration=2, prompt="cinematic cutaway"))

    assert video.url == "https://example.com/fal.mp4"
    assert captured["url"] == "https://queue.fal.run/bytedance/seedance-2.0/text-to-video"
    assert captured["headers"]["Authorization"] == "Key fal-key"
    assert captured["json"]["duration"] == "4"
    assert captured["json"]["aspect_ratio"] == "9:16"
    assert captured["json"]["generate_audio"] is False


def test_seedance_fal_queue_poll_result(monkeypatch) -> None:
    calls = []

    class Response:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self) -> None:
            pass

        def json(self):
            return self._data

    def fake_post(url, headers, json, timeout):
        calls.append(("post", url))
        return Response(
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/model/requests/req-1/status",
                "response_url": "https://queue.fal.run/model/requests/req-1/response",
            }
        )

    def fake_get(url, headers, timeout):
        calls.append(("get", url))
        if url.endswith("/status"):
            return Response(
                {
                    "status": "COMPLETED",
                    "response_url": "https://queue.fal.run/model/requests/req-1/response",
                }
            )
        return Response({"video": {"url": "https://example.com/queued.mp4"}})

    monkeypatch.setattr("clipcli.seedance.httpx.post", fake_post)
    monkeypatch.setattr("clipcli.seedance.httpx.get", fake_get)
    client = SeedDanceClient(provider="fal", api_key="fal-key")

    video = client.create_video(BrollPrompt(at=0, duration=4, prompt="cinematic cutaway"))

    assert video.url == "https://example.com/queued.mp4"
    assert calls == [
        ("post", "https://queue.fal.run/bytedance/seedance-2.0/text-to-video"),
        ("get", "https://queue.fal.run/model/requests/req-1/status"),
        ("get", "https://queue.fal.run/model/requests/req-1/response"),
    ]


def test_seedance_fal_uses_image_to_video_when_image_url(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {"video": {"url": "https://example.com/image.mp4"}}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return Response()

    monkeypatch.setattr("clipcli.seedance.httpx.post", fake_post)
    client = SeedDanceClient(provider="fal", api_key="fal-key")

    video = client.create_video(
        BrollPrompt(
            at=0,
            duration=4,
            prompt="subtle camera move over product dashboard",
            image_url="data:image/jpeg;base64,abc",
        )
    )

    assert video.url == "https://example.com/image.mp4"
    assert captured["url"] == "https://queue.fal.run/bytedance/seedance-2.0/image-to-video"
    assert captured["json"]["image_url"] == "data:image/jpeg;base64,abc"
