from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from turn_eval.benchmarks.turnbench import (  # noqa: E402
    TurnBenchProbabilityFile,
    TurnBenchProbabilityTask,
)


def _sort_key(entry) -> tuple[int, int | str]:
    try:
        return (0, int(entry.conversation_id))
    except ValueError:
        return (1, entry.conversation_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Kyutai Semantic VAD TurnBench shards")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--physical-devices", nargs="+")
    args = parser.parse_args()
    if args.physical_devices is not None and len(args.physical_devices) != len(args.shards):
        raise ValueError("--physical-devices must provide exactly one label per shard")

    metadata_rows = [json.loads((path / "metadata.json").read_text()) for path in args.shards]
    first = metadata_rows[0]
    identity = (
        first["benchmark"]["dataset_revision"],
        first["model"],
        first["evaluation"],
    )
    shard_count = int(first["benchmark"]["sharding"]["count"])
    indices: set[int] = set()
    for metadata in metadata_rows:
        current = (
            metadata["benchmark"]["dataset_revision"],
            metadata["model"],
            metadata["evaluation"],
        )
        if current != identity:
            raise ValueError("Kyutai Semantic VAD shard identities do not match")
        sharding = metadata["benchmark"]["sharding"]
        if int(sharding["count"]) != shard_count or sharding["limited"]:
            raise ValueError("cannot merge limited or differently partitioned shards")
        index = int(sharding["index"])
        if index in indices:
            raise ValueError(f"duplicate shard index: {index}")
        indices.add(index)
    if indices != set(range(shard_count)):
        raise ValueError(f"incomplete shard set: got {sorted(indices)}, expected {shard_count}")

    merged: dict[TurnBenchProbabilityTask, TurnBenchProbabilityFile] = {}
    for task, filename in (
        (TurnBenchProbabilityTask.EOT, "probs-eot.json"),
        (TurnBenchProbabilityTask.INTERRUPTION, "probs-int.json"),
    ):
        files = [TurnBenchProbabilityFile.load(path / filename) for path in args.shards]
        if len({artifact.frame_rate_hz for artifact in files}) != 1:
            raise ValueError(f"Kyutai Semantic VAD shard frame rates differ for {task}")
        entries = [entry for artifact in files for entry in artifact.probs]
        ids = [entry.conversation_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate conversation IDs while merging {task}")
        merged[task] = TurnBenchProbabilityFile(
            task=task,
            frame_rate_hz=files[0].frame_rate_hz,
            probs=tuple(sorted(entries, key=_sort_key)),
        )

    eot_ids = {entry.conversation_id for entry in merged[TurnBenchProbabilityTask.EOT].probs}
    int_ids = {
        entry.conversation_id for entry in merged[TurnBenchProbabilityTask.INTERRUPTION].probs
    }
    if eot_ids != int_ids:
        raise ValueError("merged EOT and interruption artifacts cover different conversations")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    merged[TurnBenchProbabilityTask.EOT].write(output / "probs-eot.json")
    merged[TurnBenchProbabilityTask.INTERRUPTION].write(output / "probs-int.json")
    metadata = deepcopy(first)
    metadata["created_at"] = datetime.now(UTC).isoformat()
    metadata["benchmark"]["conversation_count"] = len(eot_ids)
    metadata["benchmark"]["leaderboard_eligible"] = True
    metadata["benchmark"]["sharding"] = {
        "merged": True,
        "count": shard_count,
        "source_directories": [str(path.resolve()) for path in args.shards],
    }
    runtime_rows = [row["runtime"] for row in metadata_rows]
    audio_s = sum(float(row["audio_s"]) for row in runtime_rows)
    inference_s = sum(float(row["inference_s"]) for row in runtime_rows)
    metadata["runtime"] = {
        "devices": [row["device"] for row in runtime_rows],
        "physical_devices": args.physical_devices,
        "initialization_s_per_shard": [row["initialization_s"] for row in runtime_rows],
        "inference_s_sum": inference_s,
        "resampling_s_sum": sum(float(row["resampling_s"]) for row in runtime_rows),
        "audio_s": audio_s,
        "aggregate_compute_rtf": inference_s / audio_s,
        "peak_vram_bytes_per_shard": [row["peak_vram_bytes"] for row in runtime_rows],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
