# clipcli

`clipcli` is a local-first command line pipeline for turning long videos into short-form clips. It combines transcription, AI clip planning, vertical rendering, captions, optional speech enhancement, optional licensed sound beds, and optional SeedDance b-roll generation.

It is designed for creators and teams who want a practical, scriptable workflow instead of a hosted editing product.

## Features

- Transcribe source videos with WhisperX.
- Ask Gemini to select source-grounded clip ranges and metadata.
- Fall back to deterministic transcript-based clip planning when AI planning returns no clips.
- Render vertical social clips with ffmpeg.
- Auto-crop around the speaker with face detection.
- Burn in ASS captions from word-level transcript timings.
- Enhance rough speech audio with DeepFilterNet plus ffmpeg mastering.
- Search Freesound for licensed sound beds and mix them quietly under speech with sidechain ducking.
- Generate SeedDance 2.0 b-roll through fal.ai or Ark/Volcengine.
- Reuse transcript and plan artifacts so expensive stages do not need to run every time.

## How It Works

```text
source video
  -> ffmpeg audio extraction
  -> WhisperX transcript
  -> Gemini clip plan, or fallback plan
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

## Requirements

- Python 3.11+
- ffmpeg and ffprobe
- Optional: WhisperX for transcription
- Optional: DeepFilterNet for speech enhancement
- Optional: Gemini API key for AI clip planning
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

Show all options:

```bash
clipcli generate --help
```

Show version:

```bash
clipcli --version
```

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
