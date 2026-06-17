# clipcli

`clipcli` is a local-first command line pipeline for turning long videos into short-form clips and marketing montages. It combines transcription, AI clip planning, vertical rendering, captions, AI voiceover, optional speech enhancement, optional licensed music and sound beds, and optional SeedDance b-roll generation.

It is designed for creators and teams who want a practical, scriptable workflow instead of a hosted editing product.

## Features

- Transcribe source videos with WhisperX.
- Ask Gemini to select source-grounded clip ranges and metadata.
- Or plan fully on-device with `--planner gemma`: Gemma 4 12B on MLX watches sampled keyframes and listens to the soundtrack (native image + audio inputs) — no upload, no API key.
- Build ~30s marketing montages: the planner watches the video, picks shots, writes a voiceover script and on-screen key messages (`clipcli promo`).
- Ground promo planning in a production document (`--doc` run of show / brief) and reuse its embedded images (logos, banners) as scene assets.
- Mix branded scene assets into montages: fetch real logos/images online, generate stills with Nano Banana 2 (Gemini image models), or generate animated b-roll clips with SeedDance 2.0 (text- or image-to-video).
- Synthesize the voiceover with Gemini TTS (or macOS `say`) and cut scene lengths to the narration.
- Fall back to deterministic transcript-based clip planning when AI planning returns no clips.
- Render vertical social clips with ffmpeg.
- Auto-crop around the speaker with face detection.
- Burn in ASS captions from word-level transcript timings.
- Enhance rough speech audio with DeepFilterNet plus ffmpeg mastering.
- Search Freesound for licensed music and sound beds and mix them with sidechain ducking.
- Generate SeedDance 2.0 b-roll through fal.ai or Ark/Volcengine.
- Reuse transcript and plan artifacts so expensive stages do not need to run every time.

## How It Works

```text
source video (+ optional production document)
  -> ffmpeg audio extraction
  -> WhisperX transcript
  -> Gemini (cloud) or Gemma 4 on MLX (local) clip plan, or fallback plan
  -> captions and optional b-roll references
  -> ffmpeg render
  -> optional audio enhancement
  -> optional sound-bed mix
  -> final MP4 clips
```

Generated artifacts are written under the selected output directory:

```text
outputs/my-video/
  transcript.json
  plan.json
  clips/
  broll/
  work/
```

## Commands

`clipcli` has five subcommands. Run `clipcli <command> --help` for the full option list of any of them.

| Command | What it does |
|---------|--------------|
| `clipcli generate <video>` | Full short-form pipeline: transcribe → plan clips → render vertical clips with captions (optionally b-roll, audio enhancement, sound beds). |
| `clipcli promo <video>` | Build a ~30–70s marketing montage: multi-shot edit, AI voiceover, on-screen key messages, music, end card. Supports `--doc` grounding and scene assets (logos / generated stills / SeedDance video). |
| `clipcli plan <transcript>` | Select and score clips from an existing transcript with Gemini (cloud) or Gemma 4 (`--planner gemma`, local MLX). |
| `clipcli normalize-transcript <whisperx.json>` | Convert a raw WhisperX JSON into clipcli's transcript format. |
| `clipcli validate-plan <plan.json>` | Validate and normalize a clip plan without calling any API. |

`generate` and `promo` take a source video; `plan`/`validate-plan`/`normalize-transcript` work on JSON artifacts so you can iterate without re-running expensive stages.

## Requirements

- Python 3.11+
- ffmpeg and ffprobe
- Optional: WhisperX for transcription
- Optional: DeepFilterNet for speech enhancement
- Optional: Gemini API key for cloud AI clip planning
- Optional: Apple Silicon + `pip install -e '.[local]'` (mlx-vlm) for local Gemma 4 multimodal planning
- Optional: Freesound API key for sound-bed search
- Optional: fal.ai or Ark/Volcengine credentials for SeedDance b-roll

Install ffmpeg on macOS:

```bash
brew install ffmpeg
```

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[transcribe,dev]'
```

For speech enhancement support:

```bash
pip install -e '.[audio]'
```

If you use `uv`:

```bash
uv pip install --python .venv/bin/python -e '.[transcribe,audio,dev]'
```

## Configuration

Copy the example environment file and fill only the providers you need:

```bash
cp .env.example .env.local
```

The CLI automatically loads `.env.local`, then `.env` for any values not already set.

Common variables:

```env
GEMINI_API_KEY=
GOOGLE_API_KEY=
HUGGING_FACE_TOKEN=

FREESOUND_API_KEY=

SEEDDANCE_PROVIDER=fal
FAL_KEY=
SEEDDANCE_MODEL=bytedance/seedance-2.0/text-to-video

SEEDDANCE_API_KEY=
ARK_API_KEY=
SEEDDANCE_BASE_URL=
```

Do not commit `.env.local`; it is ignored by default.

## Quick Start

Generate three clips from a source video:

```bash
clipcli generate input.mp4 --out outputs/my-video --clips 3 --device cpu
```

On Apple Silicon, use CPU for WhisperX:

```bash
clipcli generate input.mp4 \
  --out outputs/my-video \
  --device cpu \
  --compute-type int8
```

With CUDA:

```bash
clipcli generate input.mp4 \
  --out outputs/my-video \
  --device cuda \
  --compute-type float16
```

PyTorch may report that `mps` is available, but WhisperX uses faster-whisper/CTranslate2 and rejects `device="mps"` in this environment.

## Marketing Promo Montage

When you need a marketing edit instead of raw clips — a ~30 second montage with music, voiceover, and key messages — use `promo`:

```bash
clipcli promo input.mp4 --out outputs/promo --duration 30 --language fr-CA
```

The promo pipeline:

1. Lets the planner watch the actual footage (not just the transcript) and plan visually strong shots — Gemini gets a small uploaded proxy; `--planner gemma` samples keyframes and the audio track and reasons over them fully on-device.
2. Returns a promo plan: scenes with source timestamps, a voiceover line and an on-screen key message per scene, an end-card title/tagline, and a music search query.
3. Synthesizes the voiceover per scene, then derives each scene's duration from its narration.
4. Searches Freesound for an energetic, montage-length music track (ambient soundscapes are rejected), loudness-normalizes it, and mixes it under the voiceover with gentle sidechain ducking.
5. Cuts the shots — or renders asset scenes (fetched logos, document images, Nano Banana 2 generations) — crossfades them, burns in key messages and the end card.

Useful options:

```bash
clipcli promo input.mp4 --out outputs/promo \
  --duration 30 \
  --scenes 6 \
  --mode original \            # or vertical / vertical_left / vertical_right
  --language fr-CA \           # voiceover + key message language
  --doc "Run of Show.docx" \   # production document grounding (names, partners, messaging)
  --vo-provider gemini \       # default; or: chatterbox (local), say (offline)
  --music-query "uplifting corporate technology" \
  --music-volume 0.45
```

### Document grounding and scene assets

`--doc` accepts a production document (`.docx`, `.md`, `.txt`) — a run of show, a brief, a script. Its text becomes authoritative planning context (real names, titles, partners, program order), and images embedded in a `.docx` (logos, banners) are extracted and offered to the planner as scene assets.

Plans may give any scene an `asset` instead of source footage:

```jsonc
{"start": 0, "end": 5, "vo": "…", "key_message": "PARTENAIRE",
 "asset": {"kind": "generate|video|logo|url|file", "value": "image prompt | motion prompt | domain | URL | local path",
            "fit": "cover|card", "card_color": "0xFFFFFF", "accent": true, "image": "seed-frame.png"}}
```

- `file` — a local image. **This is the recommended path for real partner logos:** drop the official files somewhere and reference them with `kind: file`, `fit: card`, and `card_color: 0xFFFFFF`. Animated/odd formats (`.gif`/`.webp`) are normalized to a still frame automatically.
- `generate` — a still created with Nano Banana 2 (`gemini-3-pro-image`); fills the frame with a slow push-in. Needs `GEMINI_API_KEY`.
- `video` — an animated b-roll clip from SeedDance 2.0, looped/trimmed to the scene and stripped of audio. Set `image` to a still and SeedDance animates that exact frame (image-to-video). Needs `FAL_KEY` (or Ark credentials) — see the b-roll section. If generation fails (e.g. exhausted balance) and an `image` seed is set, the scene falls back to that still so the render still completes.
- `url` — a direct image URL; `logo` — a best-effort logo lookup by website domain. Both depend on the remote host serving the file, so they are less reliable than `file`; prefer `file` when you have the asset.

Card-framed assets render centered on the card with a thin brand accent line beneath (set `accent: false` to drop it, e.g. for the main event logo). Generated stills get a slow push-in. Assets are cached in `work/assets/`, and voiceover takes in `work/voiceover/` are reused when the line is unchanged — so re-rendering after a tweak (`--plan`) doesn't regenerate art, re-call SeedDance, or re-bill TTS for lines that did not change.

Voiceover providers:

- `gemini` (default, best quality): Gemini TTS prebuilt voices, styled by the plan's narrator direction. Pick a voice with `--vo-voice` (for example `Charon`, `Kore`, `Puck`).
- `chatterbox` (local/offline fallback): [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) by Resemble AI — MIT-licensed, runs locally, supports French and 20+ languages, with emotion intensity control (`--vo-exaggeration`, 0.3 calm to 0.7+ energetic). Install with `pip install -e '.[tts]'`.

  Chatterbox's built-in voice is an English speaker. For non-English voiceover, clipcli automatically drops classifier-free guidance to reduce the English accent, but the natural-sounding option is to clone a native speaker: pass `--vo-voice path/to/reference.wav`, or `--vo-voice source:120-138` to extract (and denoise, when DeepFilterNet is installed) a reference straight from the input video. Only publish cloned voices with the speaker's permission. Truncated takes are detected and retried automatically.
- `say`: offline macOS system voices.

Reuse or hand-edit a plan, then re-render without calling Gemini planning again:

```bash
clipcli promo input.mp4 --out outputs/promo --plan outputs/promo/promo-plan.json
```

Skip the proxy upload and plan from a transcript only:

```bash
clipcli promo input.mp4 --out outputs/promo --transcript transcript.json --no-video-planning
```

### Local planning with Gemma 4 (MLX)

On Apple Silicon you can keep planning entirely on-device. [Gemma 4 12B](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/) takes images and raw audio natively (encoder-free), so the planner *watches* sampled keyframes and *listens* to the soundtrack without uploading anything:

```bash
pip install -e '.[local]'   # mlx-vlm; needs transformers 5+, conflicts with [tts] pins
clipcli promo input.mp4 --out outputs/promo --planner gemma --language fr-CA
```

Audio is understood two ways at once: WhisperX provides the full-timeline text transcript (auto-generated and cached in `work/transcript.json` when none is passed), while Gemma itself listens to the soundtrack in 30-second clips — the model's per-clip hearing limit — and writes a timestamped digest of announcements, speakers, and energy. The planning pass then reads frames + transcript + digest together, so the montage anchors on what the video actually announces even when the transcript is garbled.

The first run downloads `mlx-community/gemma-4-12B-it-4bit` (~9 GB; ~8 GB RAM while planning). Pick another quantization with `--gemma-model mlx-community/gemma-4-12B-it-8bit`. `--planner gemma` also works for `generate` and `plan`, where the same listening digest grounds clip selection when the WhisperX transcript is noisy.

Artifacts land in the output directory: `promo.mp4`, `promo-plan.json`, plus `work/` with the proxy, per-scene segments, voiceover WAVs, scene assets (`work/assets/`), document media (`work/doc-media/`), the music bed and its license metadata.

## Render Modes

The default mode is `vertical_auto`, which samples frames, detects faces, and picks a horizontal crop anchor automatically.

```bash
clipcli generate input.mp4 --out outputs/my-video --mode vertical_auto
```

Manual crop modes:

```bash
clipcli generate input.mp4 --out outputs/my-video --mode vertical_left
clipcli generate input.mp4 --out outputs/my-video --mode vertical_right
```

Fine-grained crop anchor:

```bash
clipcli generate input.mp4 --out outputs/my-video --crop-x 0.15
```

Keep the original aspect ratio:

```bash
clipcli generate input.mp4 --out outputs/my-video --mode original
```

## Captions

Captions are burned in by default. Disable them when you want a clean export:

```bash
clipcli generate input.mp4 --out outputs/my-video --no-captions
```

The default visual style is `viral`: punchier color, sharpened image, top hook, and large short-form captions. Use `clean` for simpler captions and no color treatment:

```bash
clipcli generate input.mp4 --out outputs/my-video --render-style clean
```

## Speech Enhancement

Enhance rough speech audio:

```bash
clipcli generate input.mp4 --out outputs/my-video --enhance-audio
```

This extracts each clip's audio as 48 kHz WAV, runs DeepFilterNet denoise, applies a speech mastering chain with high-pass/low-pass filtering, compression, loudness normalization, and a limiter, then muxes the enhanced AAC track back into the MP4.

When you are not reusing an existing transcript, `--enhance-audio` also denoises the audio before WhisperX transcription.

## Sound Beds

Add a subtle licensed sound bed under speech:

```bash
clipcli generate input.mp4 \
  --out outputs/my-video \
  --sound-search \
  --sound-query "calm background music" \
  --sound-intensity low
```

`--sound-search` currently uses Freesound. It:

- searches public Freesound metadata;
- filters out obvious NonCommercial and Sampling-only licenses;
- rejects obvious bad beds such as noise, horror, applause, rain, crowd walla, and traffic;
- downloads the selected public preview;
- loops, fades, and sidechain-ducks it under the speaker;
- writes source and license metadata to `work/sounds/`.

Always verify the saved source page and license before publishing commercial content. Some platform-native sounds are licensed only for use inside that platform, so trending songs are usually safer to add inside TikTok, Instagram, or YouTube Shorts rather than baking them into the MP4.

## SeedDance B-Roll

Generate b-roll from Gemini's planned prompts:

```bash
clipcli generate input.mp4 --out outputs/my-video --broll
```

By default, b-roll generation uses `--broll-reference source-frame`: the CLI extracts a real frame from the source video at the b-roll timestamp and sends it as image input when supported. This keeps generated cutaways closer to the real product, speaker, event, or location.

Reference modes:

```bash
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference source-frame
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference source-frame-left
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference source-frame-center
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference source-frame-right
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference source-frame:0.12
clipcli generate input.mp4 --out outputs/my-video --broll --broll-reference none
```

For fal.ai:

```env
SEEDDANCE_PROVIDER=fal
FAL_KEY=your_fal_key
SEEDDANCE_MODEL=bytedance/seedance-2.0/text-to-video
```

For Ark/Volcengine:

```env
SEEDDANCE_PROVIDER=ark
SEEDDANCE_API_KEY=your_api_key
SEEDDANCE_MODEL=doubao-seedance-2-0-260128
```

Provider model IDs can change by account and region. Override `SEEDDANCE_MODEL` when your provider exposes a different endpoint.

## Reuse Stages

Write transcript and plan only:

```bash
clipcli generate input.mp4 --out outputs/my-video --no-render
```

Normalize a WhisperX JSON transcript:

```bash
clipcli normalize-transcript whisperx-result.json --out transcript.json
```

Ask Gemini to plan clips from an existing transcript:

```bash
clipcli plan transcript.json --out plan.json --clips 5
```

Or plan locally with Gemma 4 on MLX, passing the source so it can listen to the audio when the transcript is noisy:

```bash
clipcli plan transcript.json --out plan.json --clips 5 --planner gemma --source input.mp4
```

Validate a raw Gemini JSON response or saved plan:

```bash
clipcli validate-plan plan.json
```

Render using existing transcript and plan:

```bash
clipcli generate input.mp4 \
  --transcript transcript.json \
  --plan plan.json \
  --out outputs/reuse
```

## CLI Reference

Every command prints its full option list with `--help`:

```bash
clipcli --help                      # list all commands
clipcli generate --help             # clip pipeline options
clipcli promo --help                # montage options (assets, --doc, voiceover, music)
clipcli plan --help                 # planning options (--planner gemini|gemma)
clipcli normalize-transcript --help
clipcli validate-plan --help
```

Show version:

```bash
clipcli --version
```

Most-used options at a glance:

| Option | Commands | Purpose |
|--------|----------|---------|
| `--out, -o` | all that write output | Output directory. |
| `--planner gemini\|gemma` | `generate`, `promo`, `plan` | Cloud Gemini or local Gemma 4 on MLX. |
| `--mode` | `generate`, `promo` | `vertical_auto` (default), `vertical_left/right`, `original`. |
| `--language` | `promo` | Voiceover + key-message language (e.g. `fr-CA`, `en`). |
| `--doc` | `promo` | Ground planning in a `.docx`/`.md`/`.txt`; extract its images as assets. |
| `--plan` | `generate`, `promo` | Reuse/hand-edit a saved plan and skip planning. |
| `--transcript` | `generate`, `promo` | Reuse a transcript and skip transcription. |
| `--vo-provider` | `promo` | `gemini` (default), `chatterbox` (local), `say` (offline). |

## Testing

Run the test suite:

```bash
python -m pytest -q
```

The tests mock external providers and focus on command construction, parsing, filtering, and pipeline behavior.

## Development Notes

- Keep generated media under `outputs/`; it is ignored by git.
- Keep provider credentials in `.env.local`; it is ignored by git.
- Prefer reusing `--transcript` and `--plan` during iteration.
- For product or healthcare clips, keep sound beds low and speech-forward.
- For commercial publishing, verify every downloaded sound's source page and license.

## Roadmap

- Additional sound providers with stronger commercial-license metadata.
- Better transcript cleanup before caption rendering.
- Multi-segment dynamic crop plans.
- More robust speaker tracking.
- Provider-specific SeedDance model presets.

## Related Projects

- SupoClip: https://github.com/FujiwaraChoki/supoclip
- WhisperX: https://github.com/m-bain/whisperX
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- Freesound API: https://freesound.org/docs/api/
- Gemini API: https://ai.google.dev/gemini-api/docs

## License

MIT. See [LICENSE](LICENSE).
