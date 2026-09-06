from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from model import MimiFeatureExtractor, load_endpointer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from turn_eval.benchmarks.turnbench import (  # noqa: E402
    TURNBENCH_DEV_DATASET,
    TURNBENCH_DEV_REVISION,
    TurnBenchConversationProbabilities,
    TurnBenchProbabilityFile,
    TurnBenchProbabilityTask,
    TurnBenchSpeakerProbabilities,
    load_turnbench,
)

ENDPOINTER_REPOSITORY = "viks66/mimi-endpointer"
ENDPOINTER_REVISION = "f7459f95f43766a52c612ee1bc72b2444e86fc6b"
ENDPOINTER_FILENAME = "oto_d1f.pt"
ENDPOINTER_SHA256 = "1b06a15d02d6405f2b9e4c5641d40d277128fb62c525e1a3e04716f4233b9bb4"
MIMI_REPOSITORY = "kyutai/moshiko-pytorch-bf16"
MIMI_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
MIMI_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
MODEL_LICENSE = "CC-BY-4.0"
SAMPLE_RATE = 24_000
FRAME_RATE_HZ = 25.0
IDX_SYSTEM = 3
IDX_USER = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(filename: str, repository: str, revision: str, local: bool) -> Path:
    return Path(
        hf_hub_download(
            repository,
            filename,
            revision=revision,
            local_files_only=local,
        )
    )


def _load_models(device: str, local_files_only: bool):
    endpointer_path = _download(
        ENDPOINTER_FILENAME,
        ENDPOINTER_REPOSITORY,
        ENDPOINTER_REVISION,
        local_files_only,
    )
    if _sha256(endpointer_path) != ENDPOINTER_SHA256:
        raise RuntimeError("Mimi Endpointer checkpoint checksum mismatch")
    mimi_path = _download(
        MIMI_FILENAME,
        MIMI_REPOSITORY,
        MIMI_REVISION,
        local_files_only,
    )
    endpointer = load_endpointer(endpointer_path, device)
    extractor = MimiFeatureExtractor(mimi_path, device)
    return endpointer, extractor, endpointer_path, mimi_path


def _resample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    tensor = torch.from_numpy(np.array(samples, dtype=np.float32, copy=True))
    if sample_rate != SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, sample_rate, SAMPLE_RATE)
    return tensor.numpy()


@torch.no_grad()
def _infer(conversation, endpointer, extractor, device: str):
    resampling_started = perf_counter()
    speaker_1 = _resample(
        conversation.speaker_1_audio.samples,
        conversation.speaker_1_audio.sample_rate,
    )
    speaker_2 = _resample(
        conversation.speaker_2_audio.samples,
        conversation.speaker_2_audio.sample_rate,
    )
    resampling_s = perf_counter() - resampling_started
    inference_started = perf_counter()
    features_1 = extractor(speaker_1)
    features_2 = extractor(speaker_2)
    frame_count = min(features_1.shape[-1], features_2.shape[-1])
    features = torch.stack((features_1[:, :, :frame_count], features_2[:, :, :frame_count]), dim=1)
    probabilities = torch.softmax(endpointer.infer(features)[0], dim=-1).cpu().numpy()
    inference_s = perf_counter() - inference_started
    return (
        np.asarray(probabilities[:, IDX_USER], dtype=np.float32),
        np.asarray(probabilities[:, IDX_SYSTEM], dtype=np.float32),
        inference_s,
        resampling_s,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    selected_shards = dataset.parquet_shards[args.shard_index :: args.shard_count]
    if not selected_shards:
        raise ValueError("selected TurnBench shard contains no Parquet files")
    dataset = replace(dataset, parquet_shards=selected_shards)
    expected = min(args.limit or len(dataset.conversation_ids()), len(dataset.conversation_ids()))

    initialized = perf_counter()
    endpointer, extractor, endpointer_path, mimi_path = _load_models(
        args.device, args.local_files_only
    )
    initialization_s = perf_counter() - initialized
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    inference_s = 0.0
    resampling_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if cached.exists():
            with np.load(cached) as arrays:
                p_user = arrays["p_user"]
                p_system = arrays["p_system"]
                item_inference_s = float(arrays["inference_s"])
                item_resampling_s = float(arrays["resampling_s"])
            print(f"conversation={conversation.conversation_id} resumed={index}", flush=True)
        else:
            p_user, p_system, item_inference_s, item_resampling_s = _infer(
                conversation, endpointer, extractor, args.device
            )
            temporary = cached.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    p_user=p_user,
                    p_system=p_system,
                    duration_s=conversation.duration_s,
                    inference_s=item_inference_s,
                    resampling_s=item_resampling_s,
                )
            temporary.replace(cached)
            print(
                f"conversation={conversation.conversation_id} completed={index}/{expected} "
                f"duration_s={conversation.duration_s:.1f} inference_s={item_inference_s:.1f}",
                flush=True,
            )
        inference_s += item_inference_s
        resampling_s += item_resampling_s
        frame_count = int(np.floor(conversation.duration_s * FRAME_RATE_HZ))
        if len(p_user) < frame_count or len(p_system) < frame_count:
            raise RuntimeError(f"conversation {conversation.conversation_id}: too few Mimi frames")
        rows.append(
            (
                conversation.conversation_id,
                conversation.duration_s,
                p_user[:frame_count],
                p_system[:frame_count],
            )
        )

    if not rows:
        raise RuntimeError("Mimi Endpointer produced no conversations")
    durations = {row[0]: row[1] for row in rows}

    def artifact(task: TurnBenchProbabilityTask) -> TurnBenchProbabilityFile:
        entries = []
        for conversation_id, _, p_user, p_system in rows:
            first, second = (
                (np.float32(1.0) - p_user, np.float32(1.0) - p_system)
                if task is TurnBenchProbabilityTask.EOT
                else (p_user, p_system)
            )
            entries.append(
                TurnBenchConversationProbabilities(
                    conversation_id,
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in first)),
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in second)),
                )
            )
        return TurnBenchProbabilityFile(task, FRAME_RATE_HZ, tuple(entries))

    eot = artifact(TurnBenchProbabilityTask.EOT)
    interruption = artifact(TurnBenchProbabilityTask.INTERRUPTION)
    eot.validate(durations)
    interruption.validate(durations)
    output.mkdir(parents=True, exist_ok=False)
    eot.write(output / "probs-eot.json")
    interruption.write(output / "probs-int.json")
    audio_s = sum(row[1] for row in rows)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "name": "turnbench",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(rows),
            "leaderboard_eligible": args.limit is None and args.shard_count == 1,
            "sharding": {
                "index": args.shard_index,
                "count": args.shard_count,
                "limited": args.limit is not None,
                "files": [path.name for path in selected_shards],
            },
        },
        "model": {
            "name": "mimi-endpointer-oto-d1f",
            "repository": ENDPOINTER_REPOSITORY,
            "revision": ENDPOINTER_REVISION,
            "filename": ENDPOINTER_FILENAME,
            "sha256": ENDPOINTER_SHA256,
            "license": MODEL_LICENSE,
            "mimi_repository": MIMI_REPOSITORY,
            "mimi_revision": MIMI_REVISION,
            "mimi_filename": MIMI_FILENAME,
            "mimi_sha256": _sha256(mimi_path),
            "moshi_version": "0.2.11",
            "artifact_size_bytes": endpointer_path.stat().st_size,
        },
        "evaluation": {
            "mode": "native-dual-channel",
            "execution": "causal-model-full-sequence-inference",
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamp": "frame-end",
            "native_sample_rate": SAMPLE_RATE,
            "algorithmic_lookahead_ms": 0,
            "eot_mapping": "1 - P(active speaker)",
            "interruption_mapping": "P(active speaker)",
            "signal_provenance": {"eot": "derived", "interruption": "derived-floor-state"},
        },
        "runtime": {
            "device": args.device,
            "initialization_s": initialization_s,
            "inference_s": inference_s,
            "resampling_s": resampling_s,
            "audio_s": audio_s,
            "rtf": inference_s / audio_s,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned Mimi Endpointer TurnBench backend")
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
