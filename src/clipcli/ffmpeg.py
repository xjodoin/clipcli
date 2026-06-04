from __future__ import annotations

import json
import shutil
import subprocess
import base64
from dataclasses import dataclass
from pathlib import Path

from .autocrop import detect_speaker_crop_x
from .models import ClipPlan, OutputMode, RenderStyle


class FfmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrollOverlay:
    path: Path
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def ensure_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FfmpegError(f"Missing required executable(s): {', '.join(missing)}")


def run(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        joined = " ".join(command)
        raise FfmpegError(f"Command failed: {joined}\n{result.stderr.strip()}")
    return result


def probe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio(source: Path, output: Path, *, sample_rate: int = 16000) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout=1800,
    )
    return output


def extract_clip_audio(
    source: Path,
    output: Path,
    *,
    start: float,
    duration: float,
    sample_rate: int = 48000,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout=1800,
    )
    return output


def master_speech_audio(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-af",
            (
                "highpass=f=70,"
                "lowpass=f=14000,"
                "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,"
                "loudnorm=I=-16:TP=-1.5:LRA=11,"
                "alimiter=limit=0.95"
            ),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout=1800,
    )
    return output


def replace_audio(video: Path, audio: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout=1800,
    )
    return output


def mix_sound_bed(
    video: Path,
    sound: Path,
    output: Path,
    *,
    duration: float,
    volume: float = 0.08,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fade_out_start = max(0.0, duration - 0.75)
    sound_volume = min(1.0, max(0.0, volume))
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-stream_loop",
            "-1",
            "-i",
            str(sound),
            "-filter_complex",
            (
                "[0:a]aresample=48000[voice];"
                f"[1:a]aresample=48000,atrim=0:{duration:.3f},"
                "asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={min(0.5, duration / 4):.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={min(0.75, duration / 4):.3f},"
                f"volume={sound_volume:.4f}[bed];"
                "[bed][voice]sidechaincompress=threshold=0.025:ratio=8:"
                "attack=20:release=300[ducked];"
                "[voice][ducked]amix=inputs=2:duration=first:normalize=0,"
                "alimiter=limit=0.95[aout]"
            ),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout=1800,
    )
    return output


def make_proxy(
    source: Path,
    output: Path,
    *,
    height: int = 480,
    fps: int = 12,
) -> Path:
    """Render a small upload proxy for multimodal planning."""
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale=-2:{height},fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout=3600,
    )
    return output


def render_montage_segment(
    source: Path,
    output: Path,
    *,
    start: float,
    duration: float,
    mode: OutputMode = "original",
    crop_x: float | None = None,
    fps: int = 30,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Render one silent, uniformly formatted montage shot."""
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-vf",
            f"{_montage_frame_filter(mode, crop_x=crop_x)},fps={fps},format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            str(output),
        ],
        timeout=3600,
    )
    return output


def render_image_segment(
    image: Path,
    output: Path,
    *,
    duration: float,
    mode: OutputMode = "original",
    fit: str = "cover",
    fps: int = 30,
    crf: int = 18,
    preset: str = "medium",
    card_color: str = "0x0E1320",
) -> Path:
    """Render a silent montage shot from a still asset.

    fit="cover" fills the frame with a slow push-in (generated visuals);
    fit="card" centers the asset on a brand-dark card (logos stay crisp).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = montage_dimensions(mode)
    frames = max(1, int(round(duration * fps)))
    if fit == "card":
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={card_color}:s={width}x{height}:d={duration:.3f}:r={fps}",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image),
            "-filter_complex",
            (
                f"[1:v]scale=w='min(iw*4,{int(width * 0.55)})':h='min(ih*4,{int(height * 0.38)})':"
                "force_original_aspect_ratio=decrease:flags=lanczos[asset];"
                "[0:v][asset]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
            ),
            "-map",
            "[vout]",
            "-frames:v",
            str(frames),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image),
            "-vf",
            (
                f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
                f"crop={width * 2}:{height * 2},"
                f"zoompan=z='1+0.08*on/{frames}':d={frames}"
                f":x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2':s={width}x{height}:fps={fps},"
                "format=yuv420p"
            ),
            "-frames:v",
            str(frames),
        ]
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            str(output),
        ]
    )
    run(command, timeout=600)
    return output


def make_end_card(
    output: Path,
    *,
    duration: float = 3.0,
    mode: OutputMode = "original",
    fps: int = 30,
    color: str = "0x0E1320",
) -> Path:
    """Render a silent solid-color end-card segment; text is burned in via captions."""
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = montage_dimensions(mode)
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={width}x{height}:d={duration:.3f}:r={fps}",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            str(output),
        ],
        timeout=600,
    )
    return output


def concat_montage(
    segments: list[Path],
    output: Path,
    *,
    transition_duration: float = 0.35,
    render_style: RenderStyle = "viral",
    captions: Path | None = None,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Crossfade montage segments together and burn in styled overlays."""
    if not segments:
        raise FfmpegError("No montage segments to concatenate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [probe_duration(segment) for segment in segments]

    command = ["ffmpeg", "-y"]
    for segment in segments:
        command.extend(["-i", str(segment)])

    filters: list[str] = []
    current = "0:v"
    offset = 0.0
    for index in range(1, len(segments)):
        transition = min(transition_duration, durations[index - 1] / 2, durations[index] / 2)
        offset += durations[index - 1] - transition
        label = f"x{index}"
        filters.append(
            f"[{current}][{index}:v]xfade=transition=fade:"
            f"duration={transition:.3f}:offset={offset:.3f}[{label}]"
        )
        current = label

    post = list(_style_filters(render_style))
    if captions:
        post.append(f"subtitles=filename='{_escape_filter_path(captions)}'")
    if post:
        filters.append(f"[{current}]{','.join(post)}[vout]")
        current = "vout"
    elif len(segments) == 1:
        filters.append(f"[{current}]null[vout]")
        current = "vout"

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{current}]" if not current[0].isdigit() else current,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command, timeout=3600)
    return output


def mix_voiceover_music(
    video: Path,
    voiceover: list[tuple[Path, float]],
    output: Path,
    *,
    music: Path | None = None,
    duration: float,
    music_volume: float = 0.30,
) -> Path:
    """Mix offset voiceover lines and a ducked music bed under a silent montage."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-i", str(video)]
    for path, _offset in voiceover:
        command.extend(["-i", str(path)])
    if music is not None:
        command.extend(["-stream_loop", "-1", "-i", str(music)])

    filters: list[str] = []
    vo_labels: list[str] = []
    for index, (_path, offset) in enumerate(voiceover, start=1):
        delay_ms = max(0, int(round(offset * 1000)))
        label = f"vo{index}"
        filters.append(
            f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        vo_labels.append(f"[{label}]")
    if vo_labels:
        joined = "".join(vo_labels)
        mix = f"amix=inputs={len(vo_labels)}:duration=longest:normalize=0," if len(vo_labels) > 1 else ""
        filters.append(
            f"{joined}{mix}apad=whole_dur={duration:.3f},atrim=0:{duration:.3f},"
            "loudnorm=I=-15:TP=-1.5:LRA=11[vomix]"
        )

    if music is not None:
        music_index = len(voiceover) + 1
        bed_volume = min(1.0, max(0.0, music_volume))
        fade_out = min(2.0, duration / 4)
        filters.append(
            f"[{music_index}:a]aresample=48000,aformat=channel_layouts=stereo,"
            "silenceremove=start_periods=1:start_threshold=-45dB,"
            f"atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
            "loudnorm=I=-16:TP=-2.0:LRA=11,aresample=48000,"
            f"afade=t=in:st=0:d=0.8,"
            f"afade=t=out:st={max(0.0, duration - fade_out):.3f}:d={fade_out:.3f},"
            f"volume={bed_volume:.4f}[bed]"
        )

    if vo_labels and music is not None:
        filters.append("[vomix]asplit=2[vokey][vomain]")
        filters.append(
            "[bed][vokey]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=500[ducked]"
        )
        filters.append("[vomain][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]")
    elif vo_labels:
        filters.append("[vomix]alimiter=limit=0.95[aout]")
    elif music is not None:
        filters.append("[bed]alimiter=limit=0.95[aout]")
    else:
        raise FfmpegError("Nothing to mix: provide voiceover lines, music, or both.")

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command, timeout=3600)
    return output


def montage_dimensions(mode: OutputMode) -> tuple[int, int]:
    if mode == "original":
        return 1920, 1080
    return 1080, 1920


def _montage_frame_filter(mode: OutputMode, *, crop_x: float | None = None) -> str:
    if mode == "original":
        return (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,setsar=1"
        )
    return _video_filter(mode, crop_x=crop_x) or "null"


def extract_keyframes(
    source: Path,
    output_dir: Path,
    *,
    count: int = 16,
    height: int = 448,
    duration: float | None = None,
) -> list[tuple[Path, float]]:
    """Sample evenly spaced, timestamped frames for local multimodal planning."""
    if count < 1:
        raise ValueError("keyframe count must be >= 1")
    total = duration if duration is not None else probe_duration(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[Path, float]] = []
    for index in range(count):
        at = (index + 0.5) * total / count
        output = output_dir / f"frame-{index:02d}.jpg"
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{at:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale=-2:{height}",
                "-q:v",
                "3",
                str(output),
            ],
            timeout=300,
        )
        frames.append((output, at))
    return frames


def extract_frame(
    source: Path,
    output: Path,
    at: float,
    *,
    crop_x: float | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, at):.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
    ]
    if crop_x is not None:
        anchor = min(1.0, max(0.0, crop_x))
        command.extend(
            [
                "-vf",
                f"crop=ih*9/16:ih:(iw-ih*9/16)*{anchor:.4f}:0,scale=720:1280",
            ]
        )
    command.extend(["-q:v", "3", str(output)])
    run(command, timeout=300)
    return output


def image_data_uri(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def render_clip(
    source: Path,
    plan: ClipPlan,
    output: Path,
    mode: OutputMode = "vertical",
    crop_x: float | None = None,
    render_style: RenderStyle = "viral",
    crf: int = 18,
    preset: str = "medium",
    captions: Path | None = None,
    broll: list[BrollOverlay] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, plan.end - plan.start)
    resolved_crop_x = _resolve_crop_x(source, plan, mode, crop_x)
    overlays = [overlay for overlay in broll or [] if overlay.path.exists() and overlay.duration > 0]
    if overlays:
        return _render_clip_with_broll(
            source,
            plan,
            output,
            duration=duration,
            mode=mode,
            crop_x=resolved_crop_x,
            render_style=render_style,
            crf=crf,
            preset=preset,
            captions=captions,
            broll=overlays,
        )

    vf = _combined_filter(mode, captions, crop_x=resolved_crop_x, render_style=render_style)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{plan.start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
    ]
    if vf:
        command.extend(["-vf", vf])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command, timeout=3600)
    return output


def _render_clip_with_broll(
    source: Path,
    plan: ClipPlan,
    output: Path,
    *,
    duration: float,
    mode: OutputMode,
    crop_x: float | None,
    render_style: RenderStyle,
    crf: int,
    preset: str,
    captions: Path | None,
    broll: list[BrollOverlay],
) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{plan.start:.3f}",
        "-i",
        str(source),
    ]
    for overlay in broll:
        command.extend(["-i", str(overlay.path)])

    filter_complex, video_label = _broll_filter_complex(
        mode=mode,
        crop_x=crop_x,
        render_style=render_style,
        duration=duration,
        broll=broll,
        captions=captions,
    )
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            video_label,
            "-map",
            "0:a?",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output),
        ]
    )
    run(command, timeout=3600)
    return output


def _video_filter(mode: OutputMode, crop_x: float | None = None) -> str | None:
    if mode == "original":
        return None
    if mode == "vertical_auto" and crop_x is None:
        crop_x = 0.5
    if crop_x is not None:
        anchor = min(1.0, max(0.0, crop_x))
        return (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920:(iw-1080)*{anchor:.4f}:(ih-1920)/2,"
            "setsar=1"
        )
    if mode == "vertical":
        return (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "setsar=1"
        )
    if mode == "vertical_left":
        return (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920:0:(ih-1920)/2,"
            "setsar=1"
        )
    if mode == "vertical_right":
        return (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920:iw-1080:(ih-1920)/2,"
            "setsar=1"
        )
    raise ValueError(f"Unsupported output mode: {mode}")


def _resolve_crop_x(
    source: Path,
    plan: ClipPlan,
    mode: OutputMode,
    crop_x: float | None,
) -> float | None:
    if crop_x is not None or mode != "vertical_auto":
        return crop_x
    anchor = detect_speaker_crop_x(source, start=plan.start, duration=plan.duration)
    return anchor.x if anchor else 0.5


def _combined_filter(
    mode: OutputMode,
    captions: Path | None,
    *,
    crop_x: float | None = None,
    render_style: RenderStyle = "viral",
) -> str | None:
    filters = []
    video_filter = _video_filter(mode, crop_x=crop_x)
    if video_filter:
        filters.append(video_filter)
    filters.extend(_style_filters(render_style))
    if captions:
        filters.append(f"subtitles=filename='{_escape_filter_path(captions)}'")
    return ",".join(filters) if filters else None


def _broll_filter_complex(
    *,
    mode: OutputMode,
    crop_x: float | None,
    render_style: RenderStyle,
    duration: float,
    broll: list[BrollOverlay],
    captions: Path | None,
) -> tuple[str, str]:
    filters = [
        f"[0:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,{_video_filter(mode, crop_x=crop_x) or 'null'}[v0]"
    ]
    current = "v0"
    for index, overlay in enumerate(broll, start=1):
        clipped_start = max(0.0, min(overlay.start, duration))
        clipped_duration = max(0.1, min(overlay.duration, duration - clipped_start))
        end = clipped_start + clipped_duration
        prepared = f"b{index}"
        output = f"v{index}"
        filters.append(
            f"[{index}:v]trim=duration={clipped_duration:.3f},"
            f"setpts=PTS-STARTPTS,{_video_filter(mode, crop_x=crop_x) or 'null'},"
            f"format=yuva420p,fade=t=in:st=0:d={min(0.45, clipped_duration / 3):.3f}:alpha=1,"
            f"fade=t=out:st={max(0.0, clipped_duration - min(0.45, clipped_duration / 3)):.3f}:"
            f"d={min(0.45, clipped_duration / 3):.3f}:alpha=1[{prepared}]"
        )
        filters.append(
            f"[{current}][{prepared}]overlay=0:0:"
            f"enable='between(t,{clipped_start:.3f},{end:.3f})'[{output}]"
        )
        current = output

    if captions:
        styled = "styled"
        filters.append(f"[{current}]{','.join(_style_filters(render_style)) or 'null'}[{styled}]")
        current = styled
        filters.append(f"[{current}]subtitles=filename='{_escape_filter_path(captions)}'[vout]")
        current = "vout"
    else:
        styled = "styled"
        filters.append(f"[{current}]{','.join(_style_filters(render_style)) or 'null'}[{styled}]")
        current = styled

    return ";".join(filters), f"[{current}]"


def _style_filters(render_style: RenderStyle) -> list[str]:
    if render_style == "clean":
        return []
    return [
        "eq=contrast=1.08:saturation=1.16:brightness=0.015",
        "unsharp=5:5:0.55:3:3:0.25",
        "vignette=angle=PI/8:eval=init",
    ]


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", r"\'").replace(":", r"\:")
