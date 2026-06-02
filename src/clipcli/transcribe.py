from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Transcript, TranscriptSegment, Word


SUPPORTED_WHISPERX_DEVICES = {"cpu", "cuda"}


def resolve_whisperx_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "mps":
        raise RuntimeError(
            "WhisperX does not support device='mps' here because faster-whisper/"
            "CTranslate2 rejects MPS. Use --device cpu on Apple Silicon, or --device cuda "
            "on an NVIDIA machine."
        )
    if normalized not in SUPPORTED_WHISPERX_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_WHISPERX_DEVICES | {"auto"}))
        raise RuntimeError(f"Unsupported WhisperX device '{device}'. Use one of: {supported}.")
    return normalized


def load_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text())
    if "segments" in data:
        return _from_whisperx_result(data, source=str(path))
    if isinstance(data, list):
        return Transcript(segments=[_segment_from_mapping(item) for item in data], source=str(path))
    raise ValueError(f"Unsupported transcript shape in {path}")


def save_transcript(transcript: Transcript, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.model_dump_json(indent=2))
    return path


def transcribe_with_whisperx(
    audio_path: Path,
    *,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 8,
    language: str | None = None,
    hf_token: str | None = None,
    diarize: bool = False,
) -> Transcript:
    device = resolve_whisperx_device(device)
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed. Install with `pip install 'clipcli[transcribe]'`."
        ) from exc

    audio = whisperx.load_audio(str(audio_path))
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )
    result = model.transcribe(audio, batch_size=batch_size)
    detected_language = result.get("language") or language

    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if diarize:
        if not hf_token:
            raise RuntimeError("WhisperX diarization requires --hf-token or HUGGING_FACE_TOKEN.")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    result["language"] = detected_language
    return _from_whisperx_result(result, source=str(audio_path))


def _from_whisperx_result(data: dict[str, Any], source: str | None = None) -> Transcript:
    segments = [_segment_from_mapping(segment) for segment in data.get("segments", [])]
    return Transcript(
        source=source,
        language=data.get("language"),
        segments=segments,
    )


def _segment_from_mapping(segment: dict[str, Any]) -> TranscriptSegment:
    words = [
        Word(
            text=str(word.get("word") or word.get("text") or "").strip(),
            start=float(word.get("start", segment.get("start", 0))),
            end=float(word.get("end", segment.get("end", 0))),
            speaker=word.get("speaker"),
        )
        for word in segment.get("words", [])
        if word.get("start") is not None and word.get("end") is not None
    ]
    return TranscriptSegment(
        text=str(segment.get("text", "")).strip(),
        start=float(segment["start"]),
        end=float(segment["end"]),
        speaker=segment.get("speaker"),
        words=words,
    )
