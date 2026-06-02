from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .audio import enhance_with_deepfilternet
from .captions import write_ass_captions
from . import ffmpeg
from .fallback import fallback_plan_from_transcript
from .gemini import DEFAULT_GEMINI_MODEL, parse_plan_json, plan_clips_with_gemini
from .models import ClipPlan, ClipPlanResult, OutputMode, RenderStyle, SoundIntensity, SoundSource, Transcript
from .seedance import SeedDanceClient
from .sound import find_sound_bed
from .transcribe import load_transcript, save_transcript, transcribe_with_whisperx


@dataclass
class GenerateOptions:
    source: Path
    output_dir: Path
    clips: int = 3
    min_seconds: int = 15
    max_seconds: int = 60
    mode: OutputMode = "vertical"
    crop_x: float | None = None
    render_style: RenderStyle = "viral"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    transcript_path: Path | None = None
    plan_path: Path | None = None
    whisper_model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 8
    language: str | None = None
    hf_token: str | None = None
    diarize: bool = False
    generate_broll: bool = False
    broll_reference: str = "source-frame"
    enhance_audio: bool = False
    sound_search: bool = False
    sound_query: str | None = None
    sound_source: SoundSource = "freesound"
    sound_intensity: SoundIntensity = "low"
    render: bool = True
    captions: bool = True


@dataclass
class GenerateResult:
    transcript: Path
    plan: Path
    clips: list[Path]
    broll: list[Path]
    sounds: list[Path]


def generate(options: GenerateOptions) -> GenerateResult:
    ffmpeg.ensure_ffmpeg()
    source = options.source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    work_dir = options.output_dir / "work"
    clips_dir = options.output_dir / "clips"
    broll_dir = options.output_dir / "broll"
    options.output_dir.mkdir(parents=True, exist_ok=True)

    transcript = _load_or_transcribe(options, work_dir)
    transcript_path = options.output_dir / "transcript.json"
    save_transcript(transcript, transcript_path)

    plan = _load_or_plan(options, transcript)
    plan_path = options.output_dir / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2))

    broll_outputs: list[Path] = []
    broll_by_clip: dict[int, list[ffmpeg.BrollOverlay]] = {}
    if options.generate_broll:
        broll_by_clip = _generate_broll(
            plan,
            broll_dir,
            source=source,
            work_dir=work_dir,
            limit=options.clips,
            reference_mode=options.broll_reference,
        )
        broll_outputs = [
            overlay.path
            for overlays in broll_by_clip.values()
            for overlay in overlays
        ]

    rendered: list[Path] = []
    sound_outputs: list[Path] = []
    if options.render:
        for index, clip in enumerate(plan.clips[: options.clips], start=1):
            filename = f"{index:02d}-{_slug(clip.title)}.mp4"
            slug = _slug(clip.title)
            captions_path = None
            if options.captions:
                captions_path = write_ass_captions(
                    transcript,
                    clip,
                    work_dir / "captions" / f"{index:02d}-{_slug(clip.title)}.ass",
                    style=options.render_style,
                )
            output_path = clips_dir / filename
            render_output = (
                work_dir / "renders" / filename
                if options.enhance_audio or options.sound_search
                else output_path
            )
            rendered_clip = ffmpeg.render_clip(
                source,
                clip,
                render_output,
                mode=options.mode,
                crop_x=options.crop_x,
                render_style=options.render_style,
                captions=captions_path,
                broll=broll_by_clip.get(index, []),
            )
            if options.enhance_audio:
                enhanced_output = (
                    work_dir / "renders" / f"{index:02d}-{slug}-enhanced.mp4"
                    if options.sound_search
                    else output_path
                )
                rendered_clip = _replace_with_enhanced_audio(
                    source,
                    clip,
                    rendered_clip,
                    enhanced_output,
                    work_dir=work_dir,
                    index=index,
                    slug=slug,
                )
            if options.sound_search:
                sound_asset = find_sound_bed(
                    clip,
                    work_dir / "sounds" / f"{index:02d}-{slug}",
                    query=options.sound_query,
                    source=options.sound_source,
                )
                rendered_clip = ffmpeg.mix_sound_bed(
                    rendered_clip,
                    sound_asset.path,
                    output_path,
                    duration=clip.duration,
                    volume=_sound_volume(options.sound_intensity),
                )
                sound_outputs.append(sound_asset.metadata_path)
            rendered.append(rendered_clip)

    return GenerateResult(
        transcript=transcript_path,
        plan=plan_path,
        clips=rendered,
        broll=broll_outputs,
        sounds=sound_outputs,
    )


def _replace_with_enhanced_audio(
    source: Path,
    clip: ClipPlan,
    rendered_clip: Path,
    output: Path,
    *,
    work_dir: Path,
    index: int,
    slug: str,
) -> Path:
    audio_dir = work_dir / "audio"
    stem = f"{index:02d}-{slug}"
    raw_audio = ffmpeg.extract_clip_audio(
        source,
        audio_dir / f"{stem}-raw-48k.wav",
        start=clip.start,
        duration=clip.duration,
    )
    denoised_audio = enhance_with_deepfilternet(
        raw_audio,
        audio_dir / f"{stem}-deepfilter.wav",
        work_dir=audio_dir,
    )
    mastered_audio = ffmpeg.master_speech_audio(
        denoised_audio,
        audio_dir / f"{stem}-mastered.wav",
    )
    return ffmpeg.replace_audio(rendered_clip, mastered_audio, output)


def _load_or_transcribe(options: GenerateOptions, work_dir: Path) -> Transcript:
    if options.transcript_path:
        return load_transcript(options.transcript_path)
    if options.enhance_audio:
        raw_audio_path = ffmpeg.extract_audio(
            options.source,
            work_dir / "transcription-audio-raw-48k.wav",
            sample_rate=48000,
        )
        denoised_audio_path = enhance_with_deepfilternet(
            raw_audio_path,
            work_dir / "transcription-audio-deepfilter-48k.wav",
            work_dir=work_dir / "transcription-audio",
        )
        audio_path = ffmpeg.extract_audio(
            denoised_audio_path,
            work_dir / "transcription-audio-enhanced-16k.wav",
        )
    else:
        audio_path = ffmpeg.extract_audio(options.source, work_dir / "audio.wav")
    return transcribe_with_whisperx(
        audio_path,
        model_name=options.whisper_model,
        device=options.device,
        compute_type=options.compute_type,
        batch_size=options.batch_size,
        language=options.language,
        hf_token=options.hf_token,
        diarize=options.diarize,
    )


def _load_or_plan(options: GenerateOptions, transcript: Transcript) -> ClipPlanResult:
    if options.plan_path:
        plan = parse_plan_json(options.plan_path.read_text())
    else:
        plan = plan_clips_with_gemini(
            transcript,
            clips=options.clips,
            min_seconds=options.min_seconds,
            max_seconds=options.max_seconds,
            model=options.gemini_model,
        )
    if plan.clips:
        return plan
    return fallback_plan_from_transcript(
        transcript,
        clips=options.clips,
        min_seconds=options.min_seconds,
        max_seconds=options.max_seconds,
        summary=plan.summary,
        topics=plan.topics,
    )


def _generate_broll(
    plan: ClipPlanResult,
    broll_dir: Path,
    *,
    source: Path,
    work_dir: Path,
    limit: int,
    reference_mode: str,
) -> dict[int, list[ffmpeg.BrollOverlay]]:
    client = SeedDanceClient()
    outputs: dict[int, list[ffmpeg.BrollOverlay]] = {}
    for clip_index, clip in enumerate(plan.clips[:limit], start=1):
        for prompt_index, prompt in enumerate(clip.broll, start=1):
            prompt_for_generation = prompt
            if reference_mode == "source-frame" and not prompt.image_url:
                frame_path = ffmpeg.extract_frame(
                    source,
                    work_dir / "broll_refs" / f"{clip_index:02d}-{prompt_index:02d}.jpg",
                    prompt.at,
                )
                prompt_for_generation = prompt.model_copy(
                    update={"image_url": ffmpeg.image_data_uri(frame_path)}
                )
            elif reference_mode.startswith("source-frame-") and not prompt.image_url:
                crop_name = reference_mode.removeprefix("source-frame-")
                crop_x = {"left": 0.0, "center": 0.5, "right": 1.0}.get(crop_name)
                if crop_x is None:
                    raise ValueError(
                        "Unsupported b-roll reference mode. Use source-frame, "
                        "source-frame-left, source-frame-center, source-frame-right, or none."
                    )
                frame_path = ffmpeg.extract_frame(
                    source,
                    work_dir / "broll_refs" / f"{clip_index:02d}-{prompt_index:02d}.jpg",
                    prompt.at,
                    crop_x=crop_x,
                )
                prompt_for_generation = prompt.model_copy(
                    update={"image_url": ffmpeg.image_data_uri(frame_path)}
                )
            elif reference_mode.startswith("source-frame:") and not prompt.image_url:
                crop_x = float(reference_mode.split(":", 1)[1])
                frame_path = ffmpeg.extract_frame(
                    source,
                    work_dir / "broll_refs" / f"{clip_index:02d}-{prompt_index:02d}.jpg",
                    prompt.at,
                    crop_x=crop_x,
                )
                prompt_for_generation = prompt.model_copy(
                    update={"image_url": ffmpeg.image_data_uri(frame_path)}
                )
            video = client.create_video(prompt_for_generation)
            output = broll_dir / f"{clip_index:02d}-{prompt_index:02d}.mp4"
            downloaded = client.download(video, output)
            outputs.setdefault(clip_index, []).append(
                ffmpeg.BrollOverlay(
                    path=downloaded,
                    start=max(0.0, prompt.at - clip.start),
                    duration=min(prompt.duration, clip.duration),
                )
            )
    return outputs


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "clip"


def _sound_volume(intensity: SoundIntensity) -> float:
    return {
        "low": 0.055,
        "medium": 0.085,
        "high": 0.12,
    }[intensity]
