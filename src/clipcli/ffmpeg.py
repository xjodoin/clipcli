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
