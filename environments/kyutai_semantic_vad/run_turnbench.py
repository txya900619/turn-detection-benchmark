from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import moshi.models
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders

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

MODEL_REPOSITORY = "kyutai/stt-1b-en_fr-candle"
MODEL_REVISION = "095e38f6242006a93c2541149b181988397f5c7c"
MODEL_LICENSE = "CC-BY-4.0"
FILES = {
    "config.json": "a3f1c6f7a39fca1fb1bbff68eaabc560b8037d2cdc68aa1f489859949a4223de",
    "mimi-pytorch-e351c8d8@125.safetensors": (
        "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
    ),
    "model.safetensors": "b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f",
    "tokenizer_en_fr_audio_8000.model": (
        "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69"
    ),
}
VAD_HEAD_INDEX = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(local_files_only: bool) -> dict[str, Path]:
    paths = {
        filename: Path(
            hf_hub_download(
                MODEL_REPOSITORY,
                filename,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )
        for filename in FILES
    }
    for filename, expected in FILES.items():
        if _sha256(paths[filename]) != expected:
            raise RuntimeError(f"Kyutai checkpoint checksum mismatch: {filename}")
    return paths


def _load_models(device: str, local_files_only: bool):
    paths = _download(local_files_only)
    info = loaders.CheckpointInfo.from_hf_repo(
        MODEL_REPOSITORY,
        moshi_weights=paths["model.safetensors"],
        mimi_weights=paths["mimi-pytorch-e351c8d8@125.safetensors"],
        tokenizer=paths["tokenizer_en_fr_audio_8000.model"],
        config_path=paths["config.json"],
    )
    mimi = info.get_mimi(device=device).eval()
    language_model = info.get_moshi(device=device, dtype=torch.bfloat16).eval()
    generator = moshi.models.LMGen(language_model, temp=0, temp_text=0.0)
    return info, mimi, generator, paths


def _resample(samples: np.ndarray, source_rate: int, target_rate: int, device: str) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.array(samples, dtype=np.float32, copy=True)).unsqueeze(0).to(device)
    )
    if source_rate != target_rate:
        tensor = torchaudio.functional.resample(tensor, source_rate, target_rate)
    return tensor


@torch.no_grad()
def _infer(conversation, info, mimi, generator, device: str):
    resampling_started = perf_counter()
    first = _resample(
        conversation.speaker_1_audio.samples,
        conversation.speaker_1_audio.sample_rate,
        mimi.sample_rate,
        device,
    )
    second = _resample(
        conversation.speaker_2_audio.samples,
        conversation.speaker_2_audio.sample_rate,
        mimi.sample_rate,
        device,
    )
    sample_count = max(first.shape[-1], second.shape[-1])
    sample_count = math.ceil(sample_count / mimi.frame_size) * mimi.frame_size
    first = torch.nn.functional.pad(first, (0, sample_count - first.shape[-1]))
    second = torch.nn.functional.pad(second, (0, sample_count - second.shape[-1]))
    audio = torch.cat((first, second), dim=0)
    resampling_s = perf_counter() - resampling_started

    prefix_frames = math.ceil(
        info.stt_config.get("audio_silence_prefix_seconds", 1.0) * mimi.frame_rate
    )
    suffix_frames = math.ceil(info.stt_config.get("audio_delay_seconds", 5.0) * mimi.frame_rate)
    silence = torch.zeros((2, 1, mimi.frame_size), dtype=torch.float32, device=device)
    chunks = itertools.chain(
        itertools.repeat(silence, prefix_frames),
        torch.split(audio[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence, suffix_frames),
    )

    inference_started = perf_counter()
    scores: list[np.ndarray] = []
    with mimi.streaming(2), generator.streaming(2):
        for chunk in chunks:
            audio_tokens = mimi.encode(chunk)
            _, vad_heads = generator.step_with_extra_heads(audio_tokens)
            if vad_heads:
                scores.append(vad_heads[VAD_HEAD_INDEX][:, 0, 0].cpu().float().numpy())
    inference_s = perf_counter() - inference_started
    if not scores:
        raise RuntimeError(f"conversation {conversation.conversation_id}: no VAD head output")
    values = np.asarray(scores, dtype=np.float32)
    return values[:, 0], values[:, 1], inference_s, resampling_s


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
    info, mimi, generator, paths = _load_models(args.device, args.local_files_only)
    initialization_s = perf_counter() - initialized
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    rows: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    inference_s = 0.0
    resampling_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if cached.exists():
            with np.load(cached) as arrays:
                first = arrays["first"]
                second = arrays["second"]
                item_inference_s = float(arrays["inference_s"])
                item_resampling_s = float(arrays["resampling_s"])
            print(f"conversation={conversation.conversation_id} resumed={index}", flush=True)
        else:
            first, second, item_inference_s, item_resampling_s = _infer(
                conversation, info, mimi, generator, args.device
            )
            temporary = cached.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    first=first,
                    second=second,
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
        frame_count = int(np.floor(conversation.duration_s * float(mimi.frame_rate)))
        if len(first) < frame_count or len(second) < frame_count:
            raise RuntimeError(
                f"conversation {conversation.conversation_id}: too few semantic VAD frames"
            )
        rows.append(
            (
                conversation.conversation_id,
                conversation.duration_s,
                first[:frame_count],
                second[:frame_count],
            )
        )
    if not rows:
        raise RuntimeError("Kyutai Semantic VAD produced no conversations")

    durations = {row[0]: row[1] for row in rows}

    def artifact(task: TurnBenchProbabilityTask) -> TurnBenchProbabilityFile:
        entries = []
        for conversation_id, _, first, second in rows:
            speaker_1, speaker_2 = (
                (first, second)
                if task is TurnBenchProbabilityTask.EOT
                else (np.float32(1.0) - first, np.float32(1.0) - second)
            )
            entries.append(
                TurnBenchConversationProbabilities(
                    conversation_id,
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in speaker_1)),
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in speaker_2)),
                )
            )
        return TurnBenchProbabilityFile(task, float(mimi.frame_rate), tuple(entries))

    eot = artifact(TurnBenchProbabilityTask.EOT)
    interruption = artifact(TurnBenchProbabilityTask.INTERRUPTION)
    eot.validate(durations)
    interruption.validate(durations)
    output.mkdir(parents=True, exist_ok=False)
    eot.write(output / "probs-eot.json")
    interruption.write(output / "probs-int.json")

    audio_s = sum(row[1] for row in rows)
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(torch.device(args.device))
        if args.device.startswith("cuda")
        else None
    )
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
            "name": "kyutai-semantic-vad-stt-1b-en-fr-candle",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "files": {
                name: {
                    "sha256": FILES[name],
                    "size_bytes": paths[name].stat().st_size,
                }
                for name in FILES
            },
            "moshi_version": "0.2.11",
        },
        "evaluation": {
            "mode": "native-dual-channel",
            "execution": "streaming-batched-two-speaker",
            "frame_rate_hz": float(mimi.frame_rate),
            "frame_timestamp": "frame-end",
            "native_sample_rate": int(mimi.sample_rate),
            "resampling_device": args.device,
            "algorithmic_lookahead_ms": 0,
            "audio_silence_prefix_seconds": info.stt_config.get(
                "audio_silence_prefix_seconds", 1.0
            ),
            "audio_delay_seconds": info.stt_config.get("audio_delay_seconds", 5.0),
            "vad_head_index": VAD_HEAD_INDEX,
            "eot_mapping": "vad_heads[2] P(turn ending)",
            "interruption_mapping": "1 - vad_heads[2]",
            "signal_provenance": {"eot": "native-vad-head", "interruption": "derived"},
        },
        "runtime": {
            "device": args.device,
            "initialization_s": initialization_s,
            "inference_s": inference_s,
            "resampling_s": resampling_s,
            "audio_s": audio_s,
            "rtf": inference_s / audio_s,
            "peak_vram_bytes": peak_vram_bytes,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned Kyutai Semantic VAD TurnBench backend")
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
