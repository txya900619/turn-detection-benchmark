"""Index the raw benchmark results so a restored copy can be proven identical.

Run artifacts are too large for Git, so `results/` lives only on the machine
that produced it. This writes a complete inventory of that tree, hashed, which
is small enough to commit. Restoring the results elsewhere and re-running this
script reproduces the same digest for every file that survived the move.

    uv run python scripts/archive_manifest.py --output results/ARCHIVE_MANIFEST.json

The manifest records whether each run directory carries a SUCCESS marker, so a
partial run cannot be mistaken for a complete one after a restore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", str(root)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line for line in completed.stdout.splitlines() if line}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-hash-over-mb",
        type=float,
        default=0.0,
        help="record size only for files above this many megabytes (0 hashes everything)",
    )
    args = parser.parse_args()

    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"results root does not exist: {root}")
    repo = Path.cwd().resolve()
    tracked = _tracked_files(args.results_root)
    limit = args.skip_hash_over_mb * 1024 * 1024

    files: list[dict[str, object]] = []
    per_area: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    total_bytes = 0
    unhashed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(repo).as_posix()
        size = path.stat().st_size
        entry: dict[str, object] = {
            "path": relative,
            "bytes": size,
            "tracked_in_git": relative in tracked,
        }
        if limit and size > limit:
            entry["sha256"] = None
            unhashed += 1
        else:
            entry["sha256"] = _sha256(path)
        files.append(entry)
        total_bytes += size
        area = path.relative_to(root).parts[0]
        per_area[area]["files"] += 1
        per_area[area]["bytes"] += size

    run_dirs = []
    for marker in sorted(root.rglob("SUCCESS")):
        run_dirs.append(marker.parent.relative_to(repo).as_posix())

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "results_root": str(root),
        "totals": {
            "files": len(files),
            "bytes": total_bytes,
            "gibibytes": round(total_bytes / 1024**3, 3),
            "files_without_hash": unhashed,
            "completed_run_directories": len(run_dirs),
            "tracked_in_git": sum(1 for entry in files if entry["tracked_in_git"]),
        },
        "by_area": {
            area: {
                "files": counts["files"],
                "bytes": counts["bytes"],
                "gibibytes": round(counts["bytes"] / 1024**3, 3),
            }
            for area, counts in sorted(per_area.items())
        },
        "completed_run_directories": run_dirs,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    totals = manifest["totals"]
    print(
        f"{totals['files']} files, {totals['gibibytes']} GiB, "
        f"{totals['completed_run_directories']} completed runs -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
