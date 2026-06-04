from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .gemini import DEFAULT_GEMINI_MODEL, parse_plan_json, plan_clips_with_gemini
from .gemma import DEFAULT_GEMMA_MODEL, plan_clips_with_gemma
from .models import OutputMode, PlannerBackend, RenderStyle, SoundIntensity, SoundSource
from .pipeline import GenerateOptions, generate
from .promo import PromoOptions, generate_promo
from .transcribe import load_transcript, save_transcript
from .tts import DEFAULT_TTS_MODEL, VoiceoverProvider

app = typer.Typer(
    help="CLI-first AI video clipping with ffmpeg, WhisperX, Gemini or local Gemma 4 (MLX), and SeedDance.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit.", is_eager=True),
    ] = False,
    env_file: Annotated[Path | None, typer.Option("--env-file", help="Load environment variables from a file.")] = None,
) -> None:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv(".env.local")
        load_dotenv(".env", override=False)
    if version:
        console.print(f"clipcli {__version__}")
        raise typer.Exit()


@app.command("generate")
def generate_clips(
    source: Annotated[Path, typer.Argument(help="Input video file.")],
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("outputs"),
    clips: Annotated[int, typer.Option("--clips", "-n", min=1, max=20, help="Number of clips to produce.")] = 3,
    min_seconds: Annotated[int, typer.Option("--min-seconds", min=5, help="Minimum clip duration.")] = 15,
    max_seconds: Annotated[int, typer.Option("--max-seconds", min=10, help="Maximum clip duration.")] = 60,
    mode: Annotated[OutputMode, typer.Option("--mode", help="Render mode: vertical, vertical_auto, vertical_left, vertical_right, or original.")] = "vertical_auto",
    crop_x: Annotated[float | None, typer.Option("--crop-x", min=0.0, max=1.0, help="Custom horizontal crop anchor for vertical modes: 0=left, 0.5=center, 1=right.")] = None,
    render_style: Annotated[RenderStyle, typer.Option("--render-style", help="Visual style: viral or clean.")] = "viral",
    transcript: Annotated[Path | None, typer.Option("--transcript", help="Reuse an existing transcript JSON.")] = None,
    plan: Annotated[Path | None, typer.Option("--plan", help="Reuse an existing clip plan JSON.")] = None,
    planner: Annotated[PlannerBackend, typer.Option("--planner", help="Planning backend: gemini (cloud) or gemma (local multimodal on MLX).")] = "gemini",
    gemini_model: Annotated[str, typer.Option("--gemini-model", help="Gemini model for clip planning.")] = DEFAULT_GEMINI_MODEL,
    gemma_model: Annotated[str, typer.Option("--gemma-model", help="Local MLX model for --planner gemma.")] = DEFAULT_GEMMA_MODEL,
    whisper_model: Annotated[str, typer.Option("--whisper-model", help="WhisperX model name.")] = "large-v3",
    device: Annotated[str, typer.Option("--device", help="WhisperX device: cpu, cuda, or auto. MPS is not supported by WhisperX/faster-whisper.")] = "cpu",
    compute_type: Annotated[str, typer.Option("--compute-type", help="WhisperX compute type.")] = "int8",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="WhisperX batch size.")] = 8,
    language: Annotated[str | None, typer.Option("--language", help="Optional language hint.")] = None,
    diarize: Annotated[bool, typer.Option("--diarize", help="Enable WhisperX diarization.")] = False,
    hf_token: Annotated[str | None, typer.Option("--hf-token", help="Hugging Face token for diarization.")] = None,
    broll: Annotated[bool, typer.Option("--broll", help="Generate SeedDance 2.0 b-roll assets from plan prompts.")] = False,
    broll_reference: Annotated[str, typer.Option("--broll-reference", help="B-roll reference mode: source-frame, source-frame-left/center/right, source-frame:0.12, or none.")] = "source-frame",
    enhance_audio: Annotated[bool, typer.Option("--enhance-audio", help="Denoise with DeepFilterNet, then normalize/compress/limit speech audio.")] = False,
    sound_search: Annotated[bool, typer.Option("--sound-search", help="Search and mix a licensed low-volume sound bed under the speech.")] = False,
    sound_query: Annotated[str | None, typer.Option("--sound-query", help="Override the generated sound search query.")] = None,
    sound_source: Annotated[SoundSource, typer.Option("--sound-source", help="Sound search source. Currently supports: freesound.")] = "freesound",
    sound_intensity: Annotated[SoundIntensity, typer.Option("--sound-intensity", help="Background sound level: low, medium, or high.")] = "low",
    no_captions: Annotated[bool, typer.Option("--no-captions", help="Disable burned-in ASS captions.")] = False,
    no_render: Annotated[bool, typer.Option("--no-render", help="Write transcript/plan only.")] = False,
) -> None:
    """Run the full generate pipeline."""
    result = generate(
        GenerateOptions(
            source=source,
            output_dir=output_dir,
            clips=clips,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            mode=mode,
            crop_x=crop_x,
            render_style=render_style,
            planner=planner,
            gemini_model=gemini_model,
            gemma_model=gemma_model,
            transcript_path=transcript,
            plan_path=plan,
            whisper_model=whisper_model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            hf_token=hf_token or os.environ.get("HUGGING_FACE_TOKEN"),
            diarize=diarize,
            generate_broll=broll,
            broll_reference=broll_reference,
            enhance_audio=enhance_audio,
            sound_search=sound_search,
            sound_query=sound_query,
            sound_source=sound_source,
            sound_intensity=sound_intensity,
            render=not no_render,
            captions=not no_captions,
        )
    )
    _print_result(result.transcript, result.plan, result.clips, result.broll, result.sounds)


@app.command("promo")
def promo(
    source: Annotated[Path, typer.Argument(help="Input video file.")],
    output_dir: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("outputs/promo"),
    duration: Annotated[float, typer.Option("--duration", "-d", min=10, max=120, help="Target montage duration in seconds.")] = 30.0,
    scenes: Annotated[int, typer.Option("--scenes", min=3, max=12, help="Approximate number of montage scenes.")] = 6,
    mode: Annotated[OutputMode, typer.Option("--mode", help="Frame: original (16:9) or a vertical mode.")] = "original",
    crop_x: Annotated[float | None, typer.Option("--crop-x", min=0.0, max=1.0, help="Horizontal crop anchor for vertical modes.")] = None,
    render_style: Annotated[RenderStyle, typer.Option("--render-style", help="Visual style: viral or clean.")] = "viral",
    language: Annotated[str | None, typer.Option("--language", help="Voiceover/key-message language (defaults to the source language).")] = None,
    transcript: Annotated[Path | None, typer.Option("--transcript", help="Optional transcript JSON to ground the plan.")] = None,
    plan: Annotated[Path | None, typer.Option("--plan", help="Reuse an existing promo plan JSON.")] = None,
    doc: Annotated[Path | None, typer.Option("--doc", help="Production document (.docx/.md/.txt, e.g. a run of show) to ground names, partners, and messaging; embedded images become scene assets.")] = None,
    planner: Annotated[PlannerBackend, typer.Option("--planner", help="Planning backend: gemini (cloud) or gemma (local multimodal on MLX).")] = "gemini",
    gemini_model: Annotated[str, typer.Option("--gemini-model", help="Gemini model for promo planning.")] = DEFAULT_GEMINI_MODEL,
    gemma_model: Annotated[str, typer.Option("--gemma-model", help="Local MLX model for --planner gemma.")] = DEFAULT_GEMMA_MODEL,
    no_video_planning: Annotated[bool, typer.Option("--no-video-planning", help="Plan from the transcript only; skip the video proxy upload.")] = False,
    vo_provider: Annotated[VoiceoverProvider, typer.Option("--vo-provider", help="Voiceover provider: chatterbox (local Hugging Face), gemini, or say (macOS).")] = "gemini",
    vo_voice: Annotated[str | None, typer.Option("--vo-voice", help="Voice name (gemini/say); for chatterbox: a reference audio file to clone, or source:START[-END] to clone a speaker from the input video.")] = None,
    vo_exaggeration: Annotated[float, typer.Option("--vo-exaggeration", min=0.0, max=2.0, help="Chatterbox emotion intensity: 0.3 calm, 0.5 neutral, 0.7+ energetic.")] = 0.5,
    tts_model: Annotated[str, typer.Option("--tts-model", help="Gemini TTS model.")] = DEFAULT_TTS_MODEL,
    no_music: Annotated[bool, typer.Option("--no-music", help="Skip the music bed.")] = False,
    music_query: Annotated[str | None, typer.Option("--music-query", help="Override the planned music search query.")] = None,
    music_file: Annotated[Path | None, typer.Option("--music-file", help="Use a local music file instead of searching.")] = None,
    music_volume: Annotated[float, typer.Option("--music-volume", min=0.0, max=1.0, help="Music bed volume before ducking (applied after loudness normalization).")] = 0.45,
    no_end_card: Annotated[bool, typer.Option("--no-end-card", help="Skip the title/tagline end card.")] = False,
) -> None:
    """Generate a short marketing montage: multi-shot edit, voiceover, key messages, music."""
    result = generate_promo(
        PromoOptions(
            source=source,
            output_dir=output_dir,
            duration=duration,
            scenes=scenes,
            mode=mode,
            crop_x=crop_x,
            render_style=render_style,
            language=language,
            planner=planner,
            gemini_model=gemini_model,
            gemma_model=gemma_model,
            transcript_path=transcript,
            plan_path=plan,
            document_path=doc,
            video_planning=not no_video_planning,
            vo_provider=vo_provider,
            vo_voice=vo_voice,
            vo_exaggeration=vo_exaggeration,
            tts_model=tts_model,
            music=not no_music,
            music_query=music_query,
            music_file=music_file,
            music_volume=music_volume,
            end_card=not no_end_card,
        )
    )
    table = Table(title=f"Promo ({result.duration:.1f}s)")
    table.add_column("Type")
    table.add_column("Path")
    table.add_row("plan", str(result.plan))
    table.add_row("video", str(result.video))
    for vo_path in result.voiceover:
        table.add_row("voiceover", str(vo_path))
    if result.music:
        table.add_row("music", str(result.music))
    console.print(table)


@app.command()
def plan(
    transcript: Annotated[Path, typer.Argument(help="Transcript JSON from WhisperX or clipcli.")],
    output: Annotated[Path, typer.Option("--out", "-o", help="Plan output path.")] = Path("plan.json"),
    clips: Annotated[int, typer.Option("--clips", "-n", min=1, max=20)] = 3,
    min_seconds: Annotated[int, typer.Option("--min-seconds", min=5)] = 15,
    max_seconds: Annotated[int, typer.Option("--max-seconds", min=10)] = 60,
    planner: Annotated[PlannerBackend, typer.Option("--planner", help="Planning backend: gemini (cloud) or gemma (local multimodal on MLX).")] = "gemini",
    gemini_model: Annotated[str, typer.Option("--gemini-model")] = DEFAULT_GEMINI_MODEL,
    gemma_model: Annotated[str, typer.Option("--gemma-model")] = DEFAULT_GEMMA_MODEL,
    source: Annotated[Path | None, typer.Option("--source", help="Source video/audio for gemma audio grounding.")] = None,
) -> None:
    """Ask Gemini (cloud) or Gemma 4 (local MLX) to select and score clips from a transcript."""
    transcript_obj = load_transcript(transcript)
    if planner == "gemma":
        result = plan_clips_with_gemma(
            transcript_obj,
            clips=clips,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            model=gemma_model,
            source=source,
        )
    else:
        result = plan_clips_with_gemini(
            transcript_obj,
            clips=clips,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            model=gemini_model,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2))
    console.print(f"Wrote {len(result.clips)} planned clips to {output}")


@app.command()
def normalize_transcript(
    input_path: Annotated[Path, typer.Argument(help="WhisperX JSON transcript.")],
    output: Annotated[Path, typer.Option("--out", "-o", help="Normalized transcript path.")] = Path("transcript.json"),
) -> None:
    """Normalize WhisperX JSON into clipcli's transcript format."""
    transcript = load_transcript(input_path)
    save_transcript(transcript, output)
    console.print(f"Wrote {len(transcript.segments)} transcript segments to {output}")


@app.command()
def validate_plan(
    input_path: Annotated[Path, typer.Argument(help="Raw Gemini JSON response or plan JSON.")],
    output: Annotated[Path | None, typer.Option("--out", "-o", help="Optional normalized plan path.")] = None,
) -> None:
    """Validate and normalize a Gemini clip plan without calling the API."""
    result = parse_plan_json(input_path.read_text())
    table = Table(title="Clip Plan")
    table.add_column("#", justify="right")
    table.add_column("Title")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Score")
    for index, clip in enumerate(result.clips, start=1):
        table.add_row(str(index), clip.title, f"{clip.start:.1f}", f"{clip.end:.1f}", str(clip.virality.total_score))
    console.print(table)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2))
        console.print(f"Wrote normalized plan to {output}")


def _print_result(
    transcript: Path,
    plan: Path,
    clips: list[Path],
    broll: list[Path],
    sounds: list[Path],
) -> None:
    table = Table(title="Generated Assets")
    table.add_column("Type")
    table.add_column("Path")
    table.add_row("transcript", str(transcript))
    table.add_row("plan", str(plan))
    for clip in clips:
        table.add_row("clip", str(clip))
    for asset in broll:
        table.add_row("broll", str(asset))
    for asset in sounds:
        table.add_row("sound", str(asset))
    console.print(table)
