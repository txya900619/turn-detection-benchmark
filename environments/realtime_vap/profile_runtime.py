from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_turnbench import (  # noqa: E402
    CONTEXT_WINDOW_S,
    FRAME_RATE_HZ,
    MAAI_CODE_REVISION,
    MAAI_VERSION,
    MAAI_WHEEL_SHA256,
    MODEL_FILENAME,
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NATIVE_SAMPLE_RATE,
    _build_model,
    _sha256,
)

from turn_eval.runtime_profile import (  # noqa: E402
    ProcessResourceSampler,
    hardware_identity,
    process_age_seconds,
    summarize_latencies_seconds,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _push(model, channel_1: np.ndarray, channel_2: np.ndarray) -> None:
    model.process(channel_1, channel_2)
    try:
        result = model.result_dict_queue.get_nowait()
    except queue.Empty as error:
        raise RuntimeError("Realtime-VAP emitted no result during profiling") from error
    if len(result["p_now"]) != 2:
        raise RuntimeError("Realtime-VAP profiling output must contain two speakers")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.warmup_frames <= 0 or args.measurement_frames <= 0:
        raise ValueError("warmup and measurement frame counts must be positive")
    if args.sampling_interval_ms <= 0:
        raise ValueError("sampling interval must be positive")
    if args.output.exists():
        raise FileExistsError(f"runtime profile already exists: {args.output}")

    torch.set_num_threads(args.threads)
    cuda = args.device.startswith("cuda")
    sampler = ProcessResourceSampler(
        interval_s=args.sampling_interval_ms / 1_000,
        sample_gpu=cuda,
    ).start()
    initialization_started = perf_counter()
    model, checkpoint, cpc_path = _build_model(
        args.device,
        local_files_only=args.local_files_only,
    )
    model_initialization_s = perf_counter() - initialization_started
    process_to_model_constructed_s = process_age_seconds()

    samples_per_frame = round(NATIVE_SAMPLE_RATE / FRAME_RATE_HZ)
    rng = np.random.default_rng(args.seed)
    channel_1 = rng.normal(0, 0.01, samples_per_frame).astype(np.float32)
    channel_2 = np.zeros_like(channel_1)
    model.reset_runtime_state()
    model.result_dict_queue.queue.clear()

    torch_baseline_allocated = None
    torch_baseline_reserved = None
    if cuda:
        device = torch.device(args.device)
        _synchronize(args.device)
        torch_baseline_allocated = torch.cuda.memory_allocated(device)
        torch_baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

    _synchronize(args.device)
    cold_started = perf_counter()
    _push(model, channel_1, channel_2)
    _synchronize(args.device)
    cold_inference_s = perf_counter() - cold_started
    process_to_first_output_s = process_age_seconds()
    for _ in range(args.warmup_frames):
        _push(model, channel_1, channel_2)
    _synchronize(args.device)

    warm_latencies_s: list[float] = []
    measurement_started = perf_counter()
    for _ in range(args.measurement_frames):
        _synchronize(args.device)
        started = perf_counter()
        _push(model, channel_1, channel_2)
        _synchronize(args.device)
        warm_latencies_s.append(perf_counter() - started)
    measurement_wall_s = perf_counter() - measurement_started
    torch_peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
    torch_peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
    resources = sampler.stop()

    measured_audio_s = args.measurement_frames / FRAME_RATE_HZ
    artifact_sizes = {
        "checkpoint_bytes": checkpoint.stat().st_size,
        "cpc_bytes": cpc_path.stat().st_size,
    }
    artifact_sizes["total_bytes"] = sum(artifact_sizes.values())
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile_type": "isolated-cold-start-and-warm-streaming-ticks",
        "model": {
            "name": "maai-realtime-vap-trilingual-cpc-10hz-5s",
            "maai_version": MAAI_VERSION,
            "maai_code_revision": MAAI_CODE_REVISION,
            "maai_wheel_sha256": MAAI_WHEEL_SHA256,
            "checkpoint_repository": MODEL_REPOSITORY,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_filename": MODEL_FILENAME,
            "checkpoint_sha256": _sha256(checkpoint),
            "cpc_sha256": _sha256(cpc_path),
            "license": MODEL_LICENSE,
            "artifact_sizes": artifact_sizes,
        },
        "execution": {
            "device": args.device,
            "threads": torch.get_num_threads(),
            "native_sample_rate": NATIVE_SAMPLE_RATE,
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_interval_ms": 1_000 / FRAME_RATE_HZ,
            "context_window_s": CONTEXT_WINDOW_S,
            "batch_size": 1,
            "input": "deterministic low-amplitude user audio plus silent agent channel",
            "seed": args.seed,
            "concurrency": 1,
        },
        "measurement": {
            "process_start_to_model_constructed_s": process_to_model_constructed_s,
            "process_start_to_first_output_s": process_to_first_output_s,
            "model_initialization_s": model_initialization_s,
            "cold_first_tick_latency_ms": cold_inference_s * 1_000,
            "cold_first_tick_count": 1,
            "warmup_frames": args.warmup_frames,
            "warm_inference_latency_ms": summarize_latencies_seconds(warm_latencies_s),
            "warm_tick_latencies_ms": [value * 1_000 for value in warm_latencies_s],
            "measurement_frames": args.measurement_frames,
            "measurement_audio_s": measured_audio_s,
            "measurement_wall_s": measurement_wall_s,
            "rtf": measurement_wall_s / measured_audio_s,
            "realtime_multiplier": measured_audio_s / measurement_wall_s,
        },
        "resources": {
            **resources,
            "torch_cuda": {
                "baseline_allocated_bytes": torch_baseline_allocated,
                "baseline_reserved_bytes": torch_baseline_reserved,
                "peak_allocated_bytes_during_inference": torch_peak_allocated,
                "peak_reserved_bytes_during_inference": torch_peak_reserved,
                "note": "NVML process VRAM is the cross-model primary metric",
            },
        },
        "hardware": hardware_identity(),
        "software": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "torch": version("torch"),
            "huggingface_hub": version("huggingface-hub"),
        },
        "environment": {"hf_home": os.environ.get("HF_HOME")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile isolated Realtime-VAP runtime")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=64)
    parser.add_argument("--measurement-frames", type=int, default=256)
    parser.add_argument("--sampling-interval-ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
        _write_json(args.output.resolve(), result)
        print(args.output.resolve())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Realtime-VAP runtime profile: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
