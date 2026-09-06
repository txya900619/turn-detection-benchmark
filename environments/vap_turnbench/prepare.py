from __future__ import annotations

import os
import subprocess
from pathlib import Path

VAP_REPOSITORY = "https://github.com/ErikEkstedt/VoiceActivityProjection.git"
VAP_REVISION = "f39a78b23a6dccdbedd106e00b48c410b8739f5d"


def _run_git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def prepare(*, local_files_only: bool = False) -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError("HF_HOME must be set for VapGPT TurnBench execution")
    source = Path(hf_home) / "turn-eval-assets" / "vap" / "VoiceActivityProjection"
    if not source.exists():
        if local_files_only:
            raise FileNotFoundError(f"VAP source is not cached at {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", "--filter=blob:none", VAP_REPOSITORY, str(source))
        _run_git("checkout", "--detach", VAP_REVISION, cwd=source)
    if not (source / ".git").is_dir():
        raise RuntimeError(f"cached VAP source is not a Git checkout: {source}")
    revision = _run_git("rev-parse", "HEAD", cwd=source)
    if revision != VAP_REVISION:
        raise RuntimeError(
            f"cached VAP source revision is {revision}; expected {VAP_REVISION}: {source}"
        )
    checkpoint = source / "assets" / "checkpoints" / "cpc" / "60k_epoch4-d0f474de.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAP CPC checkpoint is missing: {checkpoint}")
    return source
