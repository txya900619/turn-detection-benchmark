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
    _resample,
    _run_frames,
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

DEFAULT_THRESHOLD = 0.82
POLICY_SOURCE = (
    "TurnBench EN dev, MaAI trilingual Realtime-VAP interruption operating point, FPR<=0.10"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _cache_path(cache: Path, index: int, example_id: str) -> Path:
    return cache / f"{index:05d}-{content_hash(example_id)[:10]}.json"


def _load_or_run_clip(model, item, path: Path) -> dict[str, object]:
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("example_id") != item.id:
            raise ValueError(f"SID-Bench resume cache identity mismatch: {path}")
        return cached

    audio = item.load_audio()
    resampling_started = perf_counter()
    samples = _resample(audio.samples, audio.sample_rate)
    resampling_s = perf_counter() - resampling_started
    inference_started = perf_counter()
    p_user, _ = _run_frames(model, samples, np.zeros_like(samples))
    inference_s = perf_counter() - inference_started
    cached = {
        "schema_version": 1,
        "example_id": item.id,
        "source_sample_rate": audio.sample_rate,
        "source_audio_s": audio.duration_s,
        "frame_rate_hz": FRAME_RATE_HZ,
        "p_user": p_user,
        "resampling_s": resampling_s,
        "inference_s": inference_s,
    }
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, cached)
    os.replace(temporary, path)
    return cached


def _first_commit(probabilities: list[float], threshold: float) -> float | None:
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            return (index + 1) / FRAME_RATE_HZ
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pinned MaAI multilingual Realtime-VAP on SID-Bench"
    )
    parser.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between zero and one")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

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
        "channel_mapping": "user=channel_1, silence=channel_2",
    }
    running_path = output / "RUNNING.json"
    if running_path.exists() and json.loads(running_path.read_text()) != run_config:
        raise ValueError("SID-Bench output contains a different run configuration")
    _write_json(running_path, run_config)

    initialization_started = perf_counter()
    model, checkpoint, cpc_path = _build_model(
        args.device,
        local_files_only=args.local_files_only,
    )
    initialization_s = perf_counter() - initialization_started
    predicted: dict[str, float | None] = {}
    rows: list[dict[str, object]] = []
    inference_s = 0.0
    resampling_s = 0.0
    audio_s = 0.0
    for index, item in enumerate(items):
        cached = _load_or_run_clip(model, item, _cache_path(cache, index, item.id))
        probabilities = [float(value) for value in cached["p_user"]]
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
                "maximum_p_user": max(probabilities, default=None),
            }
        )
        if not args.quiet and ((index + 1) % 25 == 0 or index + 1 == len(items)):
            print(f"progress={index + 1}/{len(items)} example={item.id}", file=sys.stderr)

    records = dataset.prediction_records(predicted, require_exact_coverage=args.limit is None)
    metrics = evaluate_sid_bench(records)
    policy = {
        "signal": "p_now[channel_1]",
        "signal_provenance": "derived",
        "threshold": args.threshold,
        "commit_rule": "first frame-end probability >= threshold",
        "frame_interval_ms": 100.0,
        "calibration_source": POLICY_SOURCE,
        "calibration_split_role": "external-development",
        "sid_test_labels_used_for_policy": False,
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            **dataset.benchmark_metadata,
            "official_scorer_revision": SID_BENCH_SCORER_REVISION,
            "partial_run": args.limit is not None,
            "primary_leaderboard_eligible": args.limit is None,
        },
        "dataset": {**dataset.metadata, "evaluated_rows": len(items)},
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
            "model_license": MODEL_LICENSE,
            "code_license": "MIT",
        },
        "evaluation_mode": {
            "channel_mode": "single_channel_fallback",
            "execution_mode": "streaming",
            "signal_provenance": "derived",
            "transcript_mode": "none",
            "service_mode": "local",
            "language_support": "supported",
        },
        "policy_hash": content_hash(policy),
        "environment": {"hf_home": os.environ.get("HF_HOME")},
        "execution": {
            "device": args.device,
            "native_sample_rate": NATIVE_SAMPLE_RATE,
            "frame_rate_hz": FRAME_RATE_HZ,
            "context_window_s": CONTEXT_WINDOW_S,
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
    (output / "SUCCESS").write_text("\n")
    running_path.unlink()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
