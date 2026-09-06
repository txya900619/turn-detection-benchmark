from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
    _resample,
    _session,
    _sha256,
)

from turn_eval.artifacts import content_hash  # noqa: E402
from turn_eval.benchmarks.sid_bench import (  # noqa: E402
    SID_BENCH_REVISION,
    SID_BENCH_SCORER_REVISION,
    load_sid_bench,
    prepare_sid_bench_dataset,
)
from turn_eval.metrics.interruption import evaluate_sid_bench  # noqa: E402

DEFAULT_THRESHOLD = 0.9896328579889585
POLICY_SOURCE = "TurnBench EN dev, DualTurn derived FVAD interruption operating point, FPR<=0.10"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cache_path(cache: Path, index: int, example_id: str) -> Path:
    return cache / f"{index:05d}-{content_hash(example_id)[:10]}.npz"


def _infer_clip(session, item) -> dict[str, object]:
    audio = item.load_audio()
    resampling_started = perf_counter()
    samples = _resample(audio.samples, audio.sample_rate)
    resampling_s = perf_counter() - resampling_started

    stream = DualTurnStream(session)
    frame_count = len(samples) // FRAME_SAMPLES
    probabilities = np.empty(frame_count, dtype=np.float32)
    inference_started = perf_counter()
    for frame in range(frame_count):
        start = frame * FRAME_SAMPLES
        user = samples[start : start + FRAME_SAMPLES]
        chunk = np.stack((user, np.zeros_like(user)))
        _, _, fvad = stream.push(chunk)
        probabilities[frame] = fvad[0, 0]
    inference_s = perf_counter() - inference_started
    return {
        "probabilities": probabilities,
        "source_sample_rate": audio.sample_rate,
        "source_audio_s": audio.duration_s,
        "resampling_s": resampling_s,
        "inference_s": inference_s,
    }


def _load_or_run_clip(session, item, path: Path) -> dict[str, object]:
    if path.is_file():
        with np.load(path) as arrays:
            cached_id = str(arrays["example_id"].item())
            if cached_id != item.id:
                raise ValueError(f"SID-Bench resume cache identity mismatch: {path}")
            return {
                "probabilities": arrays["probabilities"],
                "source_sample_rate": int(arrays["source_sample_rate"]),
                "source_audio_s": float(arrays["source_audio_s"]),
                "resampling_s": float(arrays["resampling_s"]),
                "inference_s": float(arrays["inference_s"]),
            }

    result = _infer_clip(session, item)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            example_id=np.asarray(item.id),
            probabilities=result["probabilities"],
            source_sample_rate=np.asarray(result["source_sample_rate"], dtype=np.int64),
            source_audio_s=np.asarray(result["source_audio_s"], dtype=np.float64),
            resampling_s=np.asarray(result["resampling_s"], dtype=np.float64),
            inference_s=np.asarray(result["inference_s"], dtype=np.float64),
        )
    os.replace(temporary, path)
    return result


def _first_commit(probabilities: np.ndarray, threshold: float) -> float | None:
    matches = np.flatnonzero(probabilities >= threshold)
    return None if not len(matches) else (int(matches[0]) + 1) / FRAME_RATE_HZ


def run(args: argparse.Namespace) -> Path:
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    root = prepare_sid_bench_dataset(
        args.split,
        with_audio=True,
        local_files_only=args.local_files_only,
    )
    dataset = load_sid_bench(root, args.split)
    items = dataset.items[: args.limit]
    if not items:
        raise RuntimeError("SID-Bench execution selected no examples")

    output = args.output_dir.resolve()
    if (output / "SUCCESS").exists():
        raise FileExistsError(f"completed SID-Bench output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "resume"
    cache.mkdir(exist_ok=True)
    run_config = {
        "split": args.split,
        "dataset_revision": SID_BENCH_REVISION,
        "model_revision": MODEL_REVISION,
        "threshold": args.threshold,
        "limit": args.limit,
        "frame_rate_hz": FRAME_RATE_HZ,
        "channel_mapping": "user=CH0, silence=CH1",
        "signal": "fvad[0,0] (user, 0-240ms horizon)",
    }
    running_path = output / "RUNNING.json"
    if running_path.exists() and json.loads(running_path.read_text()) != run_config:
        raise ValueError("SID-Bench output contains a different run configuration")
    _write_json(running_path, run_config)

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

    predicted: dict[str, float | None] = {}
    rows: list[dict[str, object]] = []
    inference_s = 0.0
    resampling_s = 0.0
    audio_s = 0.0
    for index, item in enumerate(items):
        cached = _load_or_run_clip(session, item, _cache_path(cache, index, item.id))
        probabilities = np.asarray(cached["probabilities"], dtype=np.float32)
        break_time = _first_commit(probabilities, args.threshold)
        predicted[item.id] = break_time
        inference_s += float(cached["inference_s"])
        resampling_s += float(cached["resampling_s"])
        audio_s += float(cached["source_audio_s"])
        rows.append(
            {
                "example_id": item.id,
                "language": item.language,
                "category": item.category,
                "annotated_duration_s": item.duration_s,
                "audio_duration_s": float(cached["source_audio_s"]),
                "gold_break_time_s": item.gold_break_time_s,
                "predicted_break_time_s": break_time,
                "frame_count": len(probabilities),
                "maximum_user_fvad_0_240": (
                    float(np.max(probabilities)) if len(probabilities) else None
                ),
            }
        )
        if not args.quiet and ((index + 1) % 25 == 0 or index + 1 == len(items)):
            print(f"progress={index + 1}/{len(items)} example={item.id}", file=sys.stderr)

    records = dataset.prediction_records(predicted, require_exact_coverage=args.limit is None)
    metrics = evaluate_sid_bench(records)
    policy = {
        "signal": "fvad[0,0] (user, 0-240ms horizon)",
        "signal_provenance": "derived-fvad",
        "threshold": args.threshold,
        "commit_rule": "first complete frame-end probability >= threshold",
        "frame_interval_ms": 1_000 / FRAME_RATE_HZ,
        "partial_final_frame": "discarded to preserve causality",
        "calibration_source": POLICY_SOURCE,
        "calibration_split_role": "external-development",
        "sid_test_labels_used_for_policy": False,
    }
    partial = args.limit is not None
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            **dataset.benchmark_metadata,
            "official_scorer_revision": SID_BENCH_SCORER_REVISION,
            "partial_run": partial,
            "primary_leaderboard_eligible": not partial,
            "leaderboard_group": "single-channel-fallback-derived-interruption",
        },
        "dataset": {**dataset.metadata, "evaluated_rows": len(items)},
        "model": {
            "name": "dualturn-endpointing",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": MODEL_SHA256,
            "model_license": MODEL_LICENSE,
            "code_license": MODEL_LICENSE,
        },
        "evaluation_mode": {
            "channel_mode": "single_channel_fallback",
            "execution_mode": "streaming",
            "signal_provenance": "derived-fvad",
            "transcript_mode": "none",
            "service_mode": "local",
            "language_support": "supported",
            "channel_mapping": "user=CH0, silence=CH1",
        },
        "policy_hash": content_hash(policy),
        "environment": {"hf_home": os.environ.get("HF_HOME")},
        "execution": {
            "provider": args.provider,
            "threads": args.threads,
            "state_placement": ("cuda-resident-iobinding" if args.provider == "cuda" else "cpu"),
            "native_sample_rate": SAMPLE_RATE,
            "frame_rate_hz": FRAME_RATE_HZ,
            "algorithmic_lookahead_ms": 0,
            "source_resampler": "causal-linear-streaming",
        },
    }
    runtime = {
        "initialization_s": initialization_s,
        "inference_s": inference_s,
        "resampling_s": resampling_s,
        "audio_s": audio_s,
        "rtf": inference_s / audio_s if audio_s else None,
        "realtime_multiplier": audio_s / inference_s if inference_s else None,
        "resume_cache_entries": len(items),
    }
    official_rows = [
        {
            "audio": item.id,
            "total_nonbreak": predicted[item.id] is None,
            "break_time": -1 if predicted[item.id] is None else predicted[item.id],
        }
        for item in items
    ]
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in official_rows),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(rows), output / "predictions.parquet")
    _write_json(output / "policy.json", policy)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "runtime.json", runtime)
    _write_json(output / "manifest.json", manifest)
    (output / "SUCCESS").write_text("\n", encoding="utf-8")
    running_path.unlink()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pinned DualTurn Endpointing derived FVAD on SID-Bench"
    )
    parser.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        print(run(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.exit(2, f"DualTurn SID-Bench: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
