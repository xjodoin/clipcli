from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from .models import BrollPrompt, SeedDanceVideo


class SeedDanceClient:
    """Minimal provider-agnostic SeedDance 2.0 HTTP client.

    SeedDance deployments vary by gateway. Configure the base URL and API key through
    SEEDDANCE_BASE_URL and SEEDDANCE_API_KEY, or pass them explicitly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        max_wait: float = 900.0,
    ) -> None:
        self.provider = (provider or os.environ.get("SEEDDANCE_PROVIDER") or "ark").strip().lower()
        if self.provider == "fal":
            self.api_key = api_key or os.environ.get("FAL_KEY") or os.environ.get("SEEDDANCE_API_KEY")
            self.base_url = (base_url or os.environ.get("FAL_BASE_URL") or "https://queue.fal.run").rstrip("/")
            self.model = (
                model
                or os.environ.get("SEEDDANCE_MODEL")
                or "bytedance/seedance-2.0/text-to-video"
            )
        else:
            self.api_key = api_key or os.environ.get("SEEDDANCE_API_KEY") or os.environ.get("ARK_API_KEY")
            self.base_url = (
                base_url
                or os.environ.get("SEEDDANCE_BASE_URL")
                or "https://ark.cn-beijing.volces.com/api/v3"
            ).rstrip("/")
            self.model = model or os.environ.get("SEEDDANCE_MODEL") or "doubao-seedance-2-0-260128"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        if not self.api_key:
            if self.provider == "fal":
                raise RuntimeError("Set FAL_KEY to generate SeedDance b-roll with fal.ai.")
            raise RuntimeError("Set SEEDDANCE_API_KEY or ARK_API_KEY to generate SeedDance b-roll.")

    def create_video(
        self,
        prompt: BrollPrompt,
        *,
        ratio: str = "9:16",
        resolution: str = "720p",
        wait: bool = True,
    ) -> SeedDanceVideo:
        if self.provider == "fal":
            return self._create_video_fal(
                prompt,
                ratio=ratio,
                resolution=resolution,
            )
        return self._create_video_ark(
            prompt,
            ratio=ratio,
            resolution=resolution,
            wait=wait,
        )

    def _create_video_ark(
        self,
        prompt: BrollPrompt,
        *,
        ratio: str,
        resolution: str,
        wait: bool,
    ) -> SeedDanceVideo:
        generation_duration = max(4.0, min(prompt.duration, 15.0))
        payload = {
            "model": self.model,
            "content": [{"type": "text", "text": prompt.prompt}],
            "duration": generation_duration,
            "ratio": ratio,
            "resolution": resolution,
            "watermark": False,
        }
        if prompt.negative_prompt:
            payload["negative_prompt"] = prompt.negative_prompt
        response = httpx.post(
            f"{self.base_url}/contents/generations/tasks",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={key: value for key, value in payload.items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        video = _video_from_response(response.json())
        if wait and video.id and not video.url:
            return self.poll(video.id)
        return video

    def _create_video_fal(
        self,
        prompt: BrollPrompt,
        *,
        ratio: str,
        resolution: str,
    ) -> SeedDanceVideo:
        generation_duration = str(int(max(4.0, min(prompt.duration, 15.0))))
        payload = {
            "prompt": prompt.prompt,
            "resolution": resolution,
            "duration": generation_duration,
            "aspect_ratio": ratio,
            "generate_audio": False,
        }
        model = self.model
        if prompt.image_url:
            payload["image_url"] = prompt.image_url
            if "text-to-video" in model:
                model = model.replace("text-to-video", "image-to-video")
        if prompt.negative_prompt:
            payload["negative_prompt"] = prompt.negative_prompt
        response = httpx.post(
            f"{self.base_url}/{model}",
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        submitted = response.json()
        response_url = submitted.get("response_url")
        status_url = submitted.get("status_url")
        if not response_url or not status_url:
            return _video_from_response(submitted)
        result = self._poll_fal_result(status_url, response_url)
        return _video_from_response(result)

    def _poll_fal_result(self, status_url: str, response_url: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_wait
        headers = {"Authorization": f"Key {self.api_key}"}
        while True:
            status_response = httpx.get(
                status_url,
                headers=headers,
                timeout=self.timeout,
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")
            if status == "COMPLETED":
                if status_data.get("error"):
                    raise RuntimeError(f"fal SeedDance generation failed: {status_data['error']}")
                result_response = httpx.get(
                    status_data.get("response_url") or response_url,
                    headers=headers,
                    timeout=self.timeout,
                )
                result_response.raise_for_status()
                return result_response.json()
            if status in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"fal SeedDance generation ended with status {status}: {status_data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"fal SeedDance generation did not finish within {self.max_wait}s.")
            time.sleep(self.poll_interval)

    def poll(self, task_id: str) -> SeedDanceVideo:
        deadline = time.monotonic() + self.max_wait
        while True:
            response = httpx.get(
                f"{self.base_url}/contents/generations/tasks/{task_id}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            video = _video_from_response(response.json())
            if video.status in {"succeeded", "failed", "cancelled"}:
                return video
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SeedDance task {task_id} did not finish within {self.max_wait}s.")
            time.sleep(self.poll_interval)

    def download(self, video: SeedDanceVideo, output: Path) -> Path:
        if not video.url:
            raise RuntimeError("SeedDance response did not include a video URL to download.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", video.url, timeout=self.timeout) as response:
            response.raise_for_status()
            with output.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        video.local_path = output
        return output


def _video_from_response(data: dict[str, Any]) -> SeedDanceVideo:
    nested = data.get("data") if isinstance(data.get("data"), dict) else data
    content = nested.get("content")
    video = nested.get("video")
    url = nested.get("url") or nested.get("video_url") or nested.get("output_url")
    if isinstance(video, dict):
        url = url or video.get("url") or video.get("video_url")
    if isinstance(content, dict):
        url = url or content.get("video_url") or content.get("url")
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                url = url or item.get("video_url") or item.get("url")
                if url:
                    break
    return SeedDanceVideo(
        id=nested.get("id") or nested.get("task_id"),
        status=nested.get("status"),
        url=url,
        raw=data,
    )
