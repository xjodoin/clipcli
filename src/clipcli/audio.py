from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class AudioEnhancementError(RuntimeError):
    pass


def enhance_with_deepfilternet(
    source: Path,
    output: Path,
    *,
    work_dir: Path,
    post_filter: bool = True,
) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    executable = _deepfilternet_executable()
    if executable is None:
        raise AudioEnhancementError(
            "DeepFilterNet is not installed. Install it with "
            "`uv pip install --python .venv/bin/python deepfilternet` "
            "or `pip install deepfilternet`."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    deepfilter_dir = work_dir / "deepfilternet"
    deepfilter_dir.mkdir(parents=True, exist_ok=True)
    before = set(deepfilter_dir.glob("*.wav"))
    command = [
        str(executable),
        "--output-dir",
        str(deepfilter_dir),
        "--log-level",
        "error",
        "--no-suffix",
    ]
    if post_filter:
        command.append("--pf")
    command.append(str(source))

    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if result.returncode != 0:
        raise AudioEnhancementError(
            f"DeepFilterNet failed: {' '.join(command)}\n{result.stderr.strip()}"
        )

    enhanced = _latest_deepfilternet_output(deepfilter_dir, source, before)
    shutil.copyfile(enhanced, output)
    return output


def _deepfilternet_executable() -> Path | None:
    candidates = ("deep-filter", "deep-filter-py", "deepFilter")
    return _find_executable(candidates)


def _find_executable(candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)

    script_bins = [Path(sys.executable).parent, Path(sys.argv[0]).parent]
    script_bins.append(Path(sys.executable).resolve().parent)
    for script_bin in script_bins:
        for name in candidates:
            found = script_bin / name
            if found.exists():
                return found
    return None


def _latest_deepfilternet_output(
    output_dir: Path,
    source: Path,
    before: set[Path],
) -> Path:
    expected = output_dir / source.name
    if expected.exists():
        return expected

    after = set(output_dir.glob("*.wav"))
    created = after - before
    candidates = created or after
    if not candidates:
        raise AudioEnhancementError("DeepFilterNet finished but did not write a WAV output.")
    return max(candidates, key=lambda path: path.stat().st_mtime)
