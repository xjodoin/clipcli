from __future__ import annotations

import shutil
import subprocess
import sys
import time
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

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        enhanced = _wait_for_deepfilternet(process, command, deepfilter_dir, source, before)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    shutil.copyfile(enhanced, output)
    return output


def _wait_for_deepfilternet(
    process: subprocess.Popen[str],
    command: list[str],
    deepfilter_dir: Path,
    source: Path,
    before: set[Path],
    *,
    timeout: float = 3600.0,
) -> Path:
    """Wait for the DeepFilterNet output without trusting the process to exit.

    deep-filter-py sometimes deadlocks in interpreter shutdown after writing its
    output, so a finished, size-stable WAV counts as success even while the
    process is still alive.
    """
    deadline = time.monotonic() + timeout
    last_size: int | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        enhanced = _try_latest_deepfilternet_output(deepfilter_dir, source, before)
        if returncode is not None:
            if returncode != 0:
                stderr = process.stderr.read().strip() if process.stderr else ""
                raise AudioEnhancementError(
                    f"DeepFilterNet failed: {' '.join(command)}\n{stderr}"
                )
            if enhanced is not None:
                return enhanced
            raise AudioEnhancementError("DeepFilterNet finished but did not write a WAV output.")
        if enhanced is not None:
            size = enhanced.stat().st_size
            if size > 0 and size == last_size:
                return enhanced
            last_size = size
        time.sleep(1.0)
    raise AudioEnhancementError("DeepFilterNet timed out without writing a WAV output.")


def _try_latest_deepfilternet_output(
    output_dir: Path,
    source: Path,
    before: set[Path],
) -> Path | None:
    try:
        return _latest_deepfilternet_output(output_dir, source, before)
    except AudioEnhancementError:
        return None


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
