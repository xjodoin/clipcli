from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import assets, ffmpeg
from .captions import write_promo_overlays
from .document import load_document
from .gemini import DEFAULT_GEMINI_MODEL, parse_promo_json, plan_promo_with_gemini
from .gemma import DEFAULT_GEMMA_MODEL, plan_promo_with_gemma
from .models import OutputMode, PlannerBackend, PromoPlan, PromoScene, RenderStyle, Transcript
from .sound import find_music
from .transcribe import load_transcript
from .tts import DEFAULT_TTS_MODEL, VoiceoverLine, VoiceoverProvider, synthesize_voiceover_lines

DEFAULT_VO_STYLE = "Read in a confident, warm, energetic advertising tone, at a brisk pace"


@dataclass
class PromoOptions:
    source: Path
    output_dir: Path
    duration: float = 30.0
    scenes: int = 6
    mode: OutputMode = "original"
    crop_x: float | None = None
    render_style: RenderStyle = "viral"
    language: str | None = None
    planner: PlannerBackend = "gemini"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemma_model: str = DEFAULT_GEMMA_MODEL
    transcript_path: Path | None = None
    plan_path: Path | None = None
    document_path: Path | None = None
    video_planning: bool = True
    vo_provider: VoiceoverProvider = "gemini"
    vo_voice: str | None = None
    vo_exaggeration: float = 0.5
    tts_model: str = DEFAULT_TTS_MODEL
    music: bool = True
    music_query: str | None = None
    music_file: Path | None = None
    music_volume: float = 0.45
    transition_seconds: float = 0.35
    end_card: bool = True
    end_card_seconds: float = 3.0
    min_scene_seconds: float = 2.2
    max_scene_seconds: float = 8.0


@dataclass
class PromoResult:
    plan: Path
    video: Path
    voiceover: list[Path] = field(default_factory=list)
    music: Path | None = None
    duration: float = 0.0


@dataclass
class _TimedScene:
    scene: PromoScene
    start: float
    duration: float
    vo: VoiceoverLine | None = None
    montage_start: float = 0.0


def generate_promo(options: PromoOptions) -> PromoResult:
    ffmpeg.ensure_ffmpeg()
    source = options.source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    work_dir = options.output_dir / "work"
    options.output_dir.mkdir(parents=True, exist_ok=True)
    video_duration = ffmpeg.probe_duration(source)

    plan = _load_or_plan(options, source, work_dir, video_duration)
    plan_path = options.output_dir / "promo-plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2))

    timed = _synthesize_and_time(options, plan, work_dir, video_duration)

    segments = [
        _render_scene_segment(item, index, source, work_dir, options)
        for index, item in enumerate(timed, start=1)
    ]
    if options.end_card:
        segments.append(
            ffmpeg.make_end_card(
                work_dir / "segments" / "end-card.mp4",
                duration=options.end_card_seconds,
                mode=options.mode,
            )
        )

    total, end_card_start = _montage_timing(timed, options)
    captions = _write_overlays(plan, timed, options, total, end_card_start)

    silent = ffmpeg.concat_montage(
        segments,
        work_dir / "montage-silent.mp4",
        transition_duration=options.transition_seconds,
        render_style=options.render_style,
        captions=captions,
    )

    music_path: Path | None = None
    if options.music_file is not None:
        music_path = options.music_file.expanduser().resolve()
    elif options.music:
        asset = find_music(
            options.music_query or plan.music_query,
            work_dir / "music",
            duration=total,
        )
        music_path = asset.path

    voiceover = [
        (item.vo.path, item.montage_start + _vo_lead(index, options))
        for index, item in enumerate(timed)
        if item.vo is not None
    ]
    output = options.output_dir / "promo.mp4"
    ffmpeg.mix_voiceover_music(
        silent,
        voiceover,
        output,
        music=music_path,
        duration=total,
        music_volume=options.music_volume,
    )

    return PromoResult(
        plan=plan_path,
        video=output,
        voiceover=[path for path, _offset in voiceover],
        music=music_path,
        duration=total,
    )


def _load_or_plan(
    options: PromoOptions,
    source: Path,
    work_dir: Path,
    video_duration: float,
) -> PromoPlan:
    if options.plan_path:
        return parse_promo_json(options.plan_path.read_text(), video_duration=video_duration)
    transcript: Transcript | None = None
    if options.transcript_path:
        transcript = load_transcript(options.transcript_path)
    document = None
    if options.document_path:
        # Run-of-show / brief: authoritative names and messaging, plus any
        # embedded visuals (logos, banners) extracted as usable scene assets.
        document = load_document(options.document_path, media_dir=work_dir / "doc-media")
    if options.planner == "gemma":
        # Local multimodal planning: Gemma 4 on MLX watches keyframes and listens
        # to the soundtrack on-device; no proxy upload needed. WhisperX text and
        # Gemma's own listening notes complement each other, so transcribe when
        # no transcript was supplied.
        if transcript is None:
            transcript = _transcribe_for_gemma(source, work_dir)
        return plan_promo_with_gemma(
            transcript,
            video=source if options.video_planning else None,
            duration=options.duration,
            scenes=options.scenes,
            language=options.language,
            video_duration=video_duration,
            model=options.gemma_model,
            work_dir=work_dir,
            document=document,
            raw_path=work_dir / "promo-plan-raw.json",
        )
    proxy: Path | None = None
    if options.video_planning:
        proxy = work_dir / "proxy.mp4"
        if not proxy.exists():
            ffmpeg.make_proxy(source, proxy)
    return plan_promo_with_gemini(
        transcript,
        video=proxy,
        duration=options.duration,
        scenes=options.scenes,
        language=options.language,
        video_duration=video_duration,
        model=options.gemini_model,
        document=document,
        raw_path=work_dir / "promo-plan-raw.json",
    )


def _render_scene_segment(
    item: _TimedScene,
    index: int,
    source: Path,
    work_dir: Path,
    options: PromoOptions,
) -> Path:
    output = work_dir / "segments" / f"{index:02d}.mp4"
    asset = item.scene.asset
    if asset is None:
        return ffmpeg.render_montage_segment(
            source,
            output,
            start=item.start,
            duration=item.duration,
            mode=options.mode,
            crop_x=options.crop_x,
        )
    # Real logos stay crisp on a card; generated visuals fill the frame.
    fit = asset.fit or ("cover" if asset.kind == "generate" else "card")
    # Document-extracted logos usually sit on white; fetched ones on transparency.
    card_color = asset.card_color or ("0xFFFFFF" if asset.kind == "file" else "0x0E1320")
    return ffmpeg.render_image_segment(
        _resolve_asset(asset, index, work_dir),
        output,
        duration=item.duration,
        mode=options.mode,
        fit=fit,
        card_color=card_color,
    )


def _resolve_asset(asset, index: int, work_dir: Path) -> Path:
    """Fetch, copy, or generate a scene asset; cached so --plan re-renders stay cheap."""
    suffix = Path(asset.value).suffix.lower() if asset.kind == "file" else ".png"
    output = work_dir / "assets" / f"{index:02d}{suffix or '.png'}"
    if output.exists():
        return output
    if asset.kind == "generate":
        return assets.generate_image(asset.value, output)
    if asset.kind == "logo":
        return assets.fetch_logo(asset.value, output)
    if asset.kind == "file":
        source = Path(asset.value).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Asset file not found: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        return output
    return assets.fetch_image_url(asset.value, output)


def _transcribe_for_gemma(source: Path, work_dir: Path) -> Transcript | None:
    """Best-effort WhisperX transcript for local planning; cached in the work dir."""
    cached = work_dir / "transcript.json"
    if cached.exists():
        return load_transcript(cached)
    try:
        from .transcribe import save_transcript, transcribe_with_whisperx

        audio = ffmpeg.extract_audio(source, work_dir / "transcribe-audio-16k.wav")
        transcript = transcribe_with_whisperx(audio)
        save_transcript(transcript, cached)
        return transcript
    except Exception:
        # Local planning still grounds on frames and Gemma's listening notes.
        return None


def _synthesize_and_time(
    options: PromoOptions,
    plan: PromoPlan,
    work_dir: Path,
    video_duration: float,
) -> list[_TimedScene]:
    vo_scenes = [scene for scene in plan.scenes if scene.vo.strip()]
    vo_lines: list[VoiceoverLine] = []
    if vo_scenes:
        vo_lines = synthesize_voiceover_lines(
            [scene.vo for scene in vo_scenes],
            work_dir / "voiceover",
            provider=options.vo_provider,
            voice=_resolve_vo_voice(options, work_dir),
            style=plan.vo_style or DEFAULT_VO_STYLE,
            language=options.language or plan.language,
            exaggeration=options.vo_exaggeration,
            model=options.tts_model,
        )
    by_scene = dict(zip((id(scene) for scene in vo_scenes), vo_lines))

    durations: list[float] = []
    for scene in plan.scenes:
        vo = by_scene.get(id(scene))
        if vo is not None:
            duration = max(options.min_scene_seconds, vo.duration + 0.7)
        else:
            duration = scene.duration_hint or scene.duration
            duration = min(options.max_scene_seconds, max(options.min_scene_seconds, duration))
        durations.append(duration)

    # A fast narrator can leave the cut well short of the target; let the shots
    # breathe a little instead of rushing, without stretching any scene too far.
    transitions = options.transition_seconds * (len(durations) - 1 + (1 if options.end_card else 0))
    projected = sum(durations) - transitions + (options.end_card_seconds if options.end_card else 0.0)
    deficit = options.duration - projected
    if deficit > 1.0:
        bonus = min(1.5, deficit / len(durations))
        durations = [duration + bonus for duration in durations]

    timed: list[_TimedScene] = []
    cursor: float | None = None
    for scene, duration in zip(plan.scenes, durations):
        vo = by_scene.get(id(scene))
        start = scene.start
        if scene.asset is None:
            # Scene durations stretch to fit narration, so consecutive shots can end up
            # replaying the same footage; nudge small overlaps forward past the last shot.
            if cursor is not None and start < cursor and cursor - start < 10.0:
                start = cursor
            if start + duration > video_duration:
                start = max(0.0, video_duration - duration)
                duration = min(duration, video_duration - start)
            cursor = start + duration
        timed.append(_TimedScene(scene=scene, start=start, duration=duration, vo=vo))

    montage_start = 0.0
    for index, item in enumerate(timed):
        item.montage_start = montage_start
        montage_start += item.duration
        if index < len(timed) - 1 or options.end_card:
            montage_start -= min(
                options.transition_seconds, item.duration / 2, _next_duration(timed, index, options) / 2
            )
    return timed


def _resolve_vo_voice(options: PromoOptions, work_dir: Path) -> str | None:
    """Resolve `source:START[-END]` voices into a (denoised) reference clip.

    Cloning a native speaker straight from the footage keeps the narrator's accent
    matched to the source language. Make sure you have the speaker's permission
    before publishing a cloned voice.
    """
    voice = options.vo_voice
    if not voice or not voice.startswith("source:"):
        return voice
    spec = voice.removeprefix("source:")
    start_text, _, end_text = spec.partition("-")
    try:
        start = float(start_text)
        end = float(end_text) if end_text else start + 15.0
    except ValueError as exc:
        raise ValueError(f"Invalid source voice reference: {voice}. Use source:START[-END].") from exc
    duration = min(max(3.0, end - start), 20.0)
    reference = ffmpeg.extract_clip_audio(
        options.source.expanduser().resolve(),
        work_dir / "voiceover" / "reference.wav",
        start=start,
        duration=duration,
    )
    try:
        from .audio import enhance_with_deepfilternet

        reference = enhance_with_deepfilternet(
            reference,
            work_dir / "voiceover" / "reference-clean.wav",
            work_dir=work_dir / "voiceover",
        )
    except Exception:
        pass
    return str(reference)


def _next_duration(timed: list[_TimedScene], index: int, options: PromoOptions) -> float:
    if index < len(timed) - 1:
        return timed[index + 1].duration
    return options.end_card_seconds if options.end_card else timed[index].duration


def _montage_timing(timed: list[_TimedScene], options: PromoOptions) -> tuple[float, float]:
    last = timed[-1]
    end_of_scenes = last.montage_start + last.duration
    if not options.end_card:
        return end_of_scenes, end_of_scenes
    transition = min(options.transition_seconds, last.duration / 2, options.end_card_seconds / 2)
    end_card_start = end_of_scenes - transition
    return end_card_start + options.end_card_seconds, end_card_start


def _write_overlays(
    plan: PromoPlan,
    timed: list[_TimedScene],
    options: PromoOptions,
    total: float,
    end_card_start: float,
) -> Path:
    width, height = ffmpeg.montage_dimensions(options.mode)
    key_messages = [
        (
            item.montage_start + 0.30,
            item.montage_start + item.duration - 0.25,
            item.scene.key_message,
        )
        for item in timed
        if item.scene.key_message.strip()
    ]
    return write_promo_overlays(
        key_messages,
        options.output_dir / "work" / "promo-overlays.ass",
        width=width,
        height=height,
        title=plan.title,
        tagline=plan.tagline,
        end_start=end_card_start + 0.2 if options.end_card else None,
        end_end=total if options.end_card else None,
    )


def _vo_lead(index: int, options: PromoOptions) -> float:
    return 0.45 if index == 0 else options.transition_seconds + 0.05
