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
from .models import OutputMode, RenderStyle, SoundIntensity, SoundSource
from .pipeline import GenerateOptions, generate
from .transcribe import load_transcript, save_transcript

app = typer.Typer(
    help="CLI-first AI video clipping with ffmpeg, WhisperX, Gemini, and SeedDance.",
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
    gemini_model: Annotated[str, typer.Option("--gemini-model", help="Gemini model for clip planning.")] = DEFAULT_GEMINI_MODEL,
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
            gemini_model=gemini_model,
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


@app.command()
def plan(
    transcript: Annotated[Path, typer.Argument(help="Transcript JSON from WhisperX or clipcli.")],
    output: Annotated[Path, typer.Option("--out", "-o", help="Plan output path.")] = Path("plan.json"),
    clips: Annotated[int, typer.Option("--clips", "-n", min=1, max=20)] = 3,
    min_seconds: Annotated[int, typer.Option("--min-seconds", min=5)] = 15,
    max_seconds: Annotated[int, typer.Option("--max-seconds", min=10)] = 60,
    gemini_model: Annotated[str, typer.Option("--gemini-model")] = DEFAULT_GEMINI_MODEL,
) -> None:
    """Ask Gemini to select and score clips from a transcript."""
    transcript_obj = load_transcript(transcript)
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
