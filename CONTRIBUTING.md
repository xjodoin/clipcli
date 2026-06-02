# Contributing

Thanks for taking a look at `clipcli`.

## Development Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[transcribe,audio,dev]'
```

Install ffmpeg:

```bash
brew install ffmpeg
```

Run tests:

```bash
python -m pytest -q
```

## Pull Requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Do not commit generated media, credentials, caches, or local output files.
- Prefer reusable CLI behavior over one-off scripts.
- Document new flags in `README.md`.

## Provider Credentials

Use `.env.local` for local credentials. It is ignored by git.

Use `.env.example` for placeholder variable names only. Never add real API keys.

## Media and Licensing

Generated output under `outputs/` is ignored by git. If you share sample media in an issue or PR, make sure you have the right to share it publicly.

For sound-bed changes, preserve license metadata and avoid adding sources that cannot be used commercially.
