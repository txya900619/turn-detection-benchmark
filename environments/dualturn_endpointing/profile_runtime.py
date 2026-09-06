from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_turnbench import (  # noqa: E402
    FRAME_RATE_HZ,
    FRAME_SAMPLES,
    MODEL_FILENAME,
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
    SAMPLE_RATE,
    DualTurnStream,
    _session,
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


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.warmup_frames <= 0 or args.measurement_frames <= 0:
        raise ValueError("warmup and measurement frame counts must be positive")
    if args.sampling_interval_ms <= 0:
        raise ValueError("sampling interval must be positive")
    if args.output.exists():
        raise FileExistsError(f"runtime profile already exists: {args.output}")

    sampler = ProcessResourceSampler(
        interval_s=args.sampling_interval_ms / 1_000,
        sample_gpu=args.provider == "cuda",
    ).start()
    artifact_started = perf_counter()
    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
        )
    )
    artifact_resolution_s = perf_counter() - artifact_started
    if _sha256(model_path) != MODEL_SHA256:
        raise RuntimeError("DualTurn streaming ONNX checksum mismatch")

    initialization_started = perf_counter()
    session = _session(model_path, args.provider, args.threads)
    model_initialization_s = perf_counter() - initialization_started
    process_to_model_constructed_s = process_age_seconds()

    rng = np.random.default_rng(args.seed)
    user = rng.normal(0, 0.01, FRAME_SAMPLES).astype(np.float32)
    chunk = np.stack((user, np.zeros_like(user)))
    stream = DualTurnStream(session)

    cold_started = perf_counter()
    stream.push(chunk)
    cold_inference_s = perf_counter() - cold_started
    process_to_first_output_s = process_age_seconds()
    for _ in range(args.warmup_frames):
        stream.push(chunk)

    warm_latencies_s: list[float] = []
    measurement_started = perf_counter()
    for _ in range(args.measurement_frames):
        started = perf_counter()
        stream.push(chunk)
        warm_latencies_s.append(perf_counter() - started)
    measurement_wall_s = perf_counter() - measurement_started
    resources = sampler.stop()

    measured_audio_s = args.measurement_frames / FRAME_RATE_HZ
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile_type": "isolated-cold-start-and-warm-streaming-ticks",
        "model": {
            "name": "dualturn-endpointing",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": MODEL_SHA256,
            "license": MODEL_LICENSE,
            "artifact_size_bytes": model_path.stat().st_size,
        },
        "execution": {
            "provider": args.provider,
            "threads": args.threads,
            "native_sample_rate": SAMPLE_RATE,
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_interval_ms": 1_000 / FRAME_RATE_HZ,
            "batch_size": 1,
            "state_placement": ("cuda-resident-iobinding" if args.provider == "cuda" else "cpu"),
            "input": "deterministic low-amplitude user audio plus silent agent channel",
            "seed": args.seed,
            "concurrency": 1,
        },
        "measurement": {
            "process_start_to_model_constructed_s": process_to_model_constructed_s,
            "process_start_to_first_output_s": process_to_first_output_s,
            "artifact_resolution_s": artifact_resolution_s,
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
        "resources": resources,
        "hardware": hardware_identity(),
        "software": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "onnxruntime_gpu": version("onnxruntime-gpu"),
            "huggingface_hub": version("huggingface-hub"),
        },
        "environment": {"hf_home": os.environ.get("HF_HOME")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile isolated DualTurn runtime")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
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
        parser.exit(2, f"DualTurn runtime profile: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
