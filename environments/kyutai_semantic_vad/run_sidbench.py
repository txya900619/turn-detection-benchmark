"""Run Kyutai STT-1B Semantic VAD on SID-Bench.

Unlike every other dual-channel model in this repository, Kyutai's TurnBench
runner processes the two speakers as independent items of a batch: the streams
are concatenated along the batch dimension, never attended across. Its committed
interruption signal, ``1 - vad_heads[2]``, is therefore an absolute property of
one speaker's own stream.

That matters here. SID-Bench provides no agent channel, and a signal defined
relative to the partner channel degenerates when that channel is silent (see
`docs/BENCHMARK_COVERAGE_AUDIT.md`). Kyutai is not exposed to that failure, so
this run is reported as ``native_single`` rather than ``single_channel_fallback``.
``verify_stream_independence.py`` checks the claim empirically.

The threshold is frozen from the completed TurnBench EN dev selection at the
official 10% FPR budget. No SID-Bench label is used to choose it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_turnbench import (  # noqa: E402
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    VAD_HEAD_INDEX,
    _infer,
    _load_models,
    _sha256,
)

from turn_eval.benchmarks.sid_bench import (  # noqa: E402
    SID_BENCH_REVISION,
    SID_BENCH_SCORER_REVISION,
    load_sid_bench,
    prepare_sid_bench_dataset,
)
from turn_eval.core.hashing import content_hash  # noqa: E402
from turn_eval.metrics.interruption import evaluate_sid_bench  # noqa: E402

# Selected on TurnBench EN dev at the official 10% FPR budget and recorded in
# results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final/scored/scores.json.
DEFAULT_THRESHOLD = 0.9735107421875
POLICY_SOURCE = "TurnBench EN dev, Kyutai Semantic VAD interruption operating point, FPR<=0.10"
TURNBENCH_SELECTION_RUN = "results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final"


@dataclass(frozen=True, slots=True)
class _Audio:
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True, slots=True)
class _Pair:
    """Adapter that lets a SID clip reuse the pinned TurnBench inference path."""

    conversation_id: str
    speaker_1_audio: _Audio
    speaker_2_audio: _Audio
    duration_s: float


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _cache_path(cache: Path, index: int, example_id: str) -> Path:
    return cache / f"{index:05d}-{content_hash(example_id)[:10]}.json"


def _infer_user_stream(model_state, item, audio, device: str):
    """Return the per-frame turn-ending probability for the user's own stream."""
    info, mimi, generator = model_state
    silence = np.zeros_like(audio.samples)
    pair = _Pair(
        conversation_id=item.id,
        speaker_1_audio=_Audio(audio.samples, audio.sample_rate),
        speaker_2_audio=_Audio(silence, audio.sample_rate),
        duration_s=audio.duration_s,
    )
    user, _, inference_s, resampling_s = _infer(pair, info, mimi, generator, device)
    return user, inference_s, resampling_s, float(mimi.frame_rate)


def _load_or_run_clip(model_state, item, path: Path, device: str) -> dict[str, object]:
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("example_id") != item.id:
            raise ValueError(f"SID-Bench resume cache identity mismatch: {path}")
        return cached

    audio = item.load_audio()
    user, inference_s, resampling_s, frame_rate = _infer_user_stream(
        model_state, item, audio, device
    )
    frame_count = int(np.floor(audio.duration_s * frame_rate))
    if len(user) < frame_count:
        raise RuntimeError(f"clip {item.id}: too few semantic VAD frames")
    # Interruption is the committed derived signal 1 - P(turn ending), matching
    # the pinned TurnBench mapping exactly.
    interruption = 1.0 - np.asarray(user[:frame_count], dtype=np.float32)
    cached = {
        "schema_version": 1,
        "example_id": item.id,
        "source_sample_rate": audio.sample_rate,
        "source_audio_s": audio.duration_s,
        "frame_rate_hz": frame_rate,
        "p_interruption": [float(value) for value in interruption],
        "resampling_s": resampling_s,
        "inference_s": inference_s,
    }
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, cached)
    os.replace(temporary, path)
    return cached


def _first_commit(probabilities: list[float], threshold: float, frame_rate: float) -> float | None:
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            return (index + 1) / frame_rate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pinned Kyutai STT-1B Semantic VAD on a SID-Bench split"
    )
    parser.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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
        "channel_mapping": "user stream only; the batched partner slot carries silence",
    }
    running_path = output / "RUNNING.json"
    if running_path.exists() and json.loads(running_path.read_text()) != run_config:
        raise ValueError("SID-Bench output contains a different run configuration")
    _write_json(running_path, run_config)

    initialization_started = perf_counter()
    info, mimi, generator, paths = _load_models(args.device, args.local_files_only)
    initialization_s = perf_counter() - initialization_started
    model_state = (info, mimi, generator)

    predicted: dict[str, float | None] = {}
    rows: list[dict[str, object]] = []
    inference_s = 0.0
    resampling_s = 0.0
    audio_s = 0.0
    frame_rate = float(mimi.frame_rate)
    for index, item in enumerate(items):
        cached = _load_or_run_clip(
            model_state, item, _cache_path(cache, index, item.id), args.device
        )
        probabilities = [float(value) for value in cached["p_interruption"]]
        break_time = _first_commit(probabilities, args.threshold, float(cached["frame_rate_hz"]))
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
                "maximum_p_interruption": max(probabilities, default=None),
            }
        )
        if not args.quiet and ((index + 1) % 25 == 0 or index + 1 == len(items)):
            print(f"progress={index + 1}/{len(items)} example={item.id}", file=sys.stderr)

    records = dataset.prediction_records(predicted, require_exact_coverage=args.limit is None)
    metrics = evaluate_sid_bench(records)
    policy = {
        "signal": "1 - vad_heads[2] on the user's own stream",
        "signal_provenance": "derived",
        "threshold": args.threshold,
        "commit_rule": "first frame-end probability >= threshold",
        "frame_interval_ms": 1_000.0 / frame_rate,
        "calibration_source": POLICY_SOURCE,
        "calibration_selection_run": TURNBENCH_SELECTION_RUN,
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
            "name": "kyutai-semantic-vad-stt-1b-en-fr-candle",
            "checkpoint_repository": MODEL_REPOSITORY,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_sha256": {name: _sha256(path) for name, path in paths.items()},
            "vad_head_index": VAD_HEAD_INDEX,
            "model_license": MODEL_LICENSE,
            "code_license": "MIT",
        },
        "evaluation_mode": {
            # The two TurnBench streams are batch items, never attended across, so
            # SID-Bench costs this model no partner-channel information at all.
            "channel_mode": "native_single",
            "execution_mode": "streaming",
            "signal_provenance": "derived",
            "transcript_mode": "none",
            "service_mode": "local",
            # The checkpoint is EN/FR; Chinese would be unverified zero-shot.
            "language_support": "supported" if args.split == "en" else "unverified_zero_shot",
        },
        "policy_hash": content_hash(policy),
        "environment": {"hf_home": os.environ.get("HF_HOME")},
        "execution": {
            "device": args.device,
            "frame_rate_hz": frame_rate,
            "source_resampler": "torchaudio.functional.resample on the selected device",
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
