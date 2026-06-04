from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import ffmpeg

DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_TTS_VOICE = "Charon"

VoiceoverProvider = Literal["chatterbox", "gemini", "say"]


class VoiceoverError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceoverLine:
    text: str
    path: Path
    duration: float


def synthesize_voiceover_lines(
    lines: list[str],
    output_dir: Path,
    *,
    provider: VoiceoverProvider = "gemini",
    voice: str | None = None,
    style: str | None = None,
    language: str | None = None,
    exaggeration: float = 0.5,
    model: str = DEFAULT_TTS_MODEL,
    api_key: str | None = None,
) -> list[VoiceoverLine]:
    """Synthesize one WAV file per voiceover line and return measured durations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chatterbox = _load_chatterbox(language) if provider == "chatterbox" else None
    results: list[VoiceoverLine] = []
    for index, text in enumerate(lines, start=1):
        output = output_dir / f"vo-{index:02d}.wav"
        cleaned = " ".join(text.split())
        if not cleaned:
            raise VoiceoverError(f"Voiceover line {index} is empty.")
        if provider == "chatterbox":
            _chatterbox_tts(
                chatterbox,
                cleaned,
                output,
                language=language,
                reference_audio=voice,
                exaggeration=exaggeration,
            )
        elif provider == "gemini":
            _gemini_tts(cleaned, output, voice=voice or DEFAULT_TTS_VOICE, style=style, model=model, api_key=api_key)
        elif provider == "say":
            _say_tts(cleaned, output, voice=voice)
        else:
            raise VoiceoverError(f"Unsupported voiceover provider: {provider}")
        results.append(VoiceoverLine(text=cleaned, path=output, duration=ffmpeg.probe_duration(output)))
    return results


def _load_chatterbox(language: str | None):
    """Load Chatterbox (Resemble AI, MIT) from Hugging Face.

    The multilingual checkpoint covers French and 20+ other languages and exposes
    emotion exaggeration control, which suits marketing voiceover.
    """
    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        raise VoiceoverError(
            "Chatterbox is not installed. Install it with: pip install -e '.[tts]' "
            "(or pip install chatterbox-tts)."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    try:
        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception:
        if device == "cpu":
            raise
        return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


def _chatterbox_tts(
    model,
    text: str,
    output: Path,
    *,
    language: str | None,
    reference_audio: str | None,
    exaggeration: float,
    attempts: int = 3,
) -> Path:
    import torchaudio

    language_id = (language or "en").split("-")[0].lower()
    kwargs: dict = {
        "language_id": language_id,
        "exaggeration": min(2.0, max(0.0, exaggeration)),
    }
    if reference_audio:
        reference = Path(reference_audio).expanduser()
        if not reference.exists():
            raise VoiceoverError(
                f"Chatterbox voice must be a reference audio file; not found: {reference}"
            )
        kwargs["audio_prompt_path"] = str(reference)
    elif language_id != "en":
        # The built-in voice is an English speaker; per the Chatterbox docs, drop CFG
        # for cross-language generation so the output doesn't keep an English accent.
        kwargs["cfg_weight"] = 0.0

    # The sampler occasionally forces EOS mid-word; keep the take whose ending is
    # quietest and retry the ones that are still speaking at the very edge.
    best_wav = None
    best_tail = None
    for _attempt in range(max(1, attempts)):
        wav = model.generate(text, **kwargs)
        tail = _tail_level(wav, model.sr)
        if best_tail is None or tail < best_tail:
            best_wav, best_tail = wav, tail
        if tail < 0.02:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), best_wav.cpu(), model.sr)
    return output


def _tail_level(wav, sample_rate: int, *, window_seconds: float = 0.25) -> float:
    """RMS of the last fraction of a take; loud tails mean the speech was cut off."""
    samples = wav.reshape(-1)
    window = max(1, int(sample_rate * window_seconds))
    tail = samples[-window:].float()
    return float((tail * tail).mean().sqrt())


def _gemini_tts(
    text: str,
    output: Path,
    *,
    voice: str,
    style: str | None,
    model: str,
    api_key: str | None,
    attempts: int = 3,
) -> Path:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise VoiceoverError("google-genai is not installed.") from exc

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise VoiceoverError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini voiceover.")

    client = genai.Client(api_key=api_key)
    contents = f"{style.strip().rstrip(':.')}: {text}" if style else text
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            audio = _inline_audio(response)
            if audio:
                _write_pcm_wav(audio, output)
                return output
            last_error = VoiceoverError("Gemini TTS returned no audio data.")
        except Exception as exc:  # provider/network errors are retryable
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise VoiceoverError(f"Gemini TTS failed for line: {text[:60]!r}") from last_error


def _inline_audio(response: object) -> bytes | None:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                return data
    return None


def _write_pcm_wav(data: bytes, output: Path, *, sample_rate: int = 24000) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)


def _say_tts(text: str, output: Path, *, voice: str | None) -> Path:
    aiff = output.with_suffix(".aiff")
    command = ["say", "-o", str(aiff)]
    if voice:
        command.extend(["-v", voice])
    command.append(text)
    ffmpeg.run(command, timeout=300)
    ffmpeg.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output)],
        timeout=300,
    )
    aiff.unlink(missing_ok=True)
    return output
