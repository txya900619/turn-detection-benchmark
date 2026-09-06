from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

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
from turn_eval.core.audio import StreamingLinearResampler  # noqa: E402

MODEL_REPOSITORY = "anyreach-ai/dualturn-endpointing"
MODEL_REVISION = "c3860ed71210fe0144af35af340fd7a4dec3d2d3"
MODEL_FILENAME = "stream_tick.onnx"
MODEL_SHA256 = "6700ea2f919b4c66355b3dafb4f91f2db9dae9e18a302b812730c04340bf819f"
MODEL_LICENSE = "Apache-2.0"
SAMPLE_RATE = 24_000
FRAME_RATE_HZ = 12.5
FRAME_SAMPLES = 1_920


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_state() -> dict[str, np.ndarray]:
    zeros = np.zeros
    return {
        "audio_ctx": zeros((2, 0), np.float32),
        "pk": zeros((8, 2, 8, 0, 64), np.float32),
        "pv": zeros((8, 2, 8, 0, 64), np.float32),
        "tf_hist": zeros((2, 260, 512), np.float32),
        "h": zeros((2, 1, 256), np.float32),
        "c": zeros((2, 1, 256), np.float32),
        "pos": np.asarray([0], dtype=np.int64),
    }


class DualTurnStream:
    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.cuda = session.get_providers()[0] == "CUDAExecutionProvider"
        initial = _initial_state()
        if self.cuda:
            self.state = {
                name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
                for name, value in initial.items()
            }
        else:
            self.state = initial

    def push(self, chunk: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        if chunk.shape != (2, FRAME_SAMPLES) or chunk.dtype != np.float32:
            raise ValueError(f"DualTurn chunk must be float32 [2, {FRAME_SAMPLES}]")
        if self.cuda:
            return self._push_cuda(chunk)
        eot, vad, fvad, audio_ctx, pk, pv, tf_hist, h, c, pos = self.session.run(
            None,
            {"chunk": chunk, **self.state},
        )
        self.state = {
            "audio_ctx": audio_ctx,
            "pk": pk,
            "pv": pv,
            "tf_hist": tf_hist,
            "h": h,
            "c": c,
            "pos": pos,
        }
        return float(eot[0, 0, 0]), vad[0, 0], fvad[0, 0]

    def _push_cuda(self, chunk: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        binding = self.session.io_binding()
        binding.bind_cpu_input("chunk", chunk)
        for name, value in self.state.items():
            binding.bind_ortvalue_input(name, value)
        for name in ("eot", "vad", "fvad"):
            binding.bind_output(name, "cpu")
        state_outputs = (
            "audio_ctx2",
            "pk2",
            "pv2",
            "tf_hist2",
            "h2",
            "c2",
            "pos2",
        )
        for name in state_outputs:
            binding.bind_output(name, "cuda", 0)
        self.session.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        eot, vad, fvad = (value.numpy() for value in outputs[:3])
        self.state = dict(
            zip(
                ("audio_ctx", "pk", "pv", "tf_hist", "h", "c", "pos"),
                outputs[3:],
                strict=True,
            )
        )
        return float(eot[0, 0, 0]), vad[0, 0], fvad[0, 0]


def _session(model_path: Path, provider: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "cuda":
        # Importing the pinned Torch wheel preloads its bundled CUDA/cuDNN
        # libraries so ONNX Runtime can resolve its provider dependencies.
        import torch  # noqa: F401

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime CUDAExecutionProvider is unavailable")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("DualTurn failed to activate the CUDAExecutionProvider")
    return session


def _resample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    return StreamingLinearResampler(sample_rate, SAMPLE_RATE).push(samples)


def _infer_conversation(conversation, session: ort.InferenceSession):
    resampling_started = perf_counter()
    speaker_1 = _resample(
        conversation.speaker_1_audio.samples, conversation.speaker_1_audio.sample_rate
    )
    speaker_2 = _resample(
        conversation.speaker_2_audio.samples, conversation.speaker_2_audio.sample_rate
    )
    resampling_s = perf_counter() - resampling_started
    original = DualTurnStream(session)
    swapped = DualTurnStream(session)
    eot_1: list[float] = []
    eot_2: list[float] = []
    interruption_1: list[float] = []
    interruption_2: list[float] = []
    inference_started = perf_counter()
    tick_count = math.ceil(len(speaker_1) / FRAME_SAMPLES)
    for tick in range(tick_count):
        start = tick * FRAME_SAMPLES
        stop = start + FRAME_SAMPLES
        chunk = np.zeros((2, FRAME_SAMPLES), dtype=np.float32)
        available = max(0, min(FRAME_SAMPLES, len(speaker_1) - start))
        chunk[0, :available] = speaker_1[start:stop]
        chunk[1, :available] = speaker_2[start:stop]
        user_eot, _, fvad = original.push(chunk)
        other_eot, _, _ = swapped.push(chunk[::-1].copy())
        eot_1.append(user_eot)
        eot_2.append(other_eot)
        interruption_1.append(float(fvad[0, 0]))
        interruption_2.append(float(fvad[1, 0]))
    inference_s = perf_counter() - inference_started
    return (
        np.asarray(eot_1, dtype=np.float32),
        np.asarray(eot_2, dtype=np.float32),
        np.asarray(interruption_1, dtype=np.float32),
        np.asarray(interruption_2, dtype=np.float32),
        inference_s,
        resampling_s,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    selected_shards = dataset.parquet_shards[args.shard_index :: args.shard_count]
    if not selected_shards:
        raise ValueError("selected TurnBench shard contains no Parquet files")
    dataset = replace(dataset, parquet_shards=selected_shards)

    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
        )
    )
    if _sha256(model_path) != MODEL_SHA256:
        raise RuntimeError("DualTurn streaming ONNX checksum mismatch")
    initialization_started = perf_counter()
    session = _session(model_path, args.provider, args.threads)
    initialization_s = perf_counter() - initialization_started

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    rows: list[tuple[str, float]] = []
    inference_s = 0.0
    resampling_s = 0.0
    expected_count = len(dataset.conversation_ids())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        expected_count = min(expected_count, args.limit)
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        rows.append((conversation.conversation_id, conversation.duration_s))
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if cached.exists():
            with np.load(cached) as arrays:
                raw[conversation.conversation_id] = (
                    arrays["eot_1"],
                    arrays["eot_2"],
                    arrays["interruption_1"],
                    arrays["interruption_2"],
                    float(arrays["duration_s"]),
                )
                inference_s += float(arrays["inference_s"])
                resampling_s += float(arrays["resampling_s"])
            print(f"conversation={conversation.conversation_id} resumed={index}", flush=True)
            continue
        result = _infer_conversation(conversation, session)
        eot_1, eot_2, int_1, int_2, item_inference_s, item_resampling_s = result
        raw[conversation.conversation_id] = (
            eot_1,
            eot_2,
            int_1,
            int_2,
            conversation.duration_s,
        )
        inference_s += item_inference_s
        resampling_s += item_resampling_s
        temporary = cached.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                eot_1=eot_1,
                eot_2=eot_2,
                interruption_1=int_1,
                interruption_2=int_2,
                duration_s=conversation.duration_s,
                inference_s=item_inference_s,
                resampling_s=item_resampling_s,
            )
        temporary.replace(cached)
        print(
            f"conversation={conversation.conversation_id} completed={index}/{expected_count} "
            f"duration_s={conversation.duration_s:.1f} inference_s={item_inference_s:.1f}",
            flush=True,
        )

    eot_entries = []
    int_entries = []
    durations = {}
    for conversation_id, duration_s in rows:
        eot_1, eot_2, int_1, int_2, _ = raw[conversation_id]
        frames = int(np.floor(duration_s * FRAME_RATE_HZ))
        arrays = (eot_1[:frames], eot_2[:frames], int_1[:frames], int_2[:frames])
        if any(len(values) != frames for values in arrays):
            raise RuntimeError(f"conversation {conversation_id}: DualTurn emitted too few frames")
        eot_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id,
                TurnBenchSpeakerProbabilities(tuple(float(value) for value in arrays[0])),
                TurnBenchSpeakerProbabilities(tuple(float(value) for value in arrays[1])),
            )
        )
        int_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id,
                TurnBenchSpeakerProbabilities(tuple(float(value) for value in arrays[2])),
                TurnBenchSpeakerProbabilities(tuple(float(value) for value in arrays[3])),
            )
        )
        durations[conversation_id] = duration_s

    eot_file = TurnBenchProbabilityFile(
        TurnBenchProbabilityTask.EOT, FRAME_RATE_HZ, tuple(eot_entries)
    )
    int_file = TurnBenchProbabilityFile(
        TurnBenchProbabilityTask.INTERRUPTION, FRAME_RATE_HZ, tuple(int_entries)
    )
    eot_file.validate(durations)
    int_file.validate(durations)
    output.mkdir(parents=True, exist_ok=False)
    eot_file.write(output / "probs-eot.json")
    int_file.write(output / "probs-int.json")
    audio_s = sum(duration for _, duration in rows)
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
            "name": "dualturn-endpointing",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": MODEL_SHA256,
            "license": MODEL_LICENSE,
        },
        "evaluation": {
            "mode": "native-dual-channel-two-orientations",
            "execution": "streaming",
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamp": "frame-end",
            "native_sample_rate": SAMPLE_RATE,
            "algorithmic_lookahead_ms": 0,
            "resampler": "causal-linear-streaming",
            "eot_mapping": "native eot_probs; channel swap for speaker 2",
            "interruption_mapping": "fvad_probs[speaker, 0-240ms]",
            "signal_provenance": {"eot": "native", "interruption": "derived-fvad"},
        },
        "runtime": {
            "provider": args.provider,
            "state_placement": "cuda-resident-iobinding" if args.provider == "cuda" else "cpu",
            "threads": args.threads,
            "initialization_s": initialization_s,
            "inference_s": inference_s,
            "resampling_s": resampling_s,
            "audio_s": audio_s,
            "rtf_two_orientations": inference_s / audio_s,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned DualTurn Endpointing TurnBench backend")
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    print(json.dumps(run(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
