"""Visual assets for montages: fetch real branding online, generate the rest.

Real organizations get their real logos (fetched by website domain); imagined
visuals — diagrams, mood shots, branding cards — are generated with Nano Banana
(the Gemini image models).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"  # Nano Banana 2

LOGO_ENDPOINT = "https://logo.clearbit.com/{domain}?size=512&format=png"
USER_AGENT = "clipcli/0.1 (+https://github.com/xjodoin/clipcli)"


class AssetError(RuntimeError):
    pass


def fetch_logo(domain: str, output: Path) -> Path:
    """Fetch an organization's logo by website domain."""
    return fetch_image_url(LOGO_ENDPOINT.format(domain=domain.strip().lower()), output)


def fetch_image_url(url: str, output: Path) -> Path:
    """Download an image asset; refuses non-image responses."""
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code != 200 or not response.content:
        raise AssetError(f"Asset fetch failed ({response.status_code}): {url}")
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise AssetError(f"Not an image ({content_type or 'unknown content type'}): {url}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return output


def generate_image(
    prompt: str,
    output: Path,
    *,
    model: str = DEFAULT_IMAGE_MODEL,
    aspect_ratio: str = "16:9",
    api_key: str | None = None,
) -> Path:
    """Generate an image with Nano Banana (Gemini image models)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise AssetError("google-genai is not installed.") from exc

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise AssetError("Set GEMINI_API_KEY or GOOGLE_API_KEY to generate assets.")

    client = genai.Client(api_key=api_key)
    config_kwargs: dict = {"response_modalities": ["TEXT", "IMAGE"]}
    try:
        config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
    except (TypeError, AttributeError):
        pass
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    data = _inline_image(response)
    if not data:
        raise AssetError(f"Nano Banana returned no image for prompt: {prompt[:60]!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output


def _inline_image(response: object) -> bytes | None:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            mime = getattr(inline, "mime_type", "") or ""
            data = getattr(inline, "data", None)
            if data and mime.startswith("image/"):
                return data
    return None
