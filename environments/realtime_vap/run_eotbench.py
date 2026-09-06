from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_turnbench import (  # noqa: E402
    CONTEXT_WINDOW_S,
    FRAME_RATE_HZ,
    MAAI_CODE_REVISION,
    MAAI_VERSION,
    MAAI_WHEEL_SHA256,
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NATIVE_SAMPLE_RATE,
    _build_model,
    _sha256,
)

from turn_eval.artifacts import (  # noqa: E402
    build_manifest,
    make_run_id,
    write_run_artifacts,
)
from turn_eval.benchmarks.eot_bench import EOT_BENCH_REVISION, load_eot_bench  # noqa: E402
from turn_eval.core.types import (  # noqa: E402
    ChannelMode,
    EvaluationMode,
    ExecutionMode,
    ServiceMode,
    SignalProvenance,
    SupportStatus,
    TranscriptMode,
)
from turn_eval.metrics.eot import eot_bench_policy_grid, evaluate_eot_sweep  # noqa: E402
from turn_eval.models.realtime_vap import RealtimeVAPSingleChannelEOTAdapter  # noqa: E402
from turn_eval.runner.decision_point import run_decision_points  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pinned multilingual Realtime-VAP on LiveKit eot-bench"
    )
    parser.add_argument("--lang", choices=("en", "zh"), required=True)
    parser.add_argument("--revision", default=EOT_BENCH_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame-rate-hz", type=float, default=FRAME_RATE_HZ)
    parser.add_argument("--context-window-s", type=float, default=CONTEXT_WINDOW_S)
    args = parser.parse_args()

    dataset = load_eot_bench(
        args.lang,
        revision=args.revision,
        limit=args.limit,
        inference_interval_s=1 / args.frame_rate_hz,
        min_silence_span_s=1 / args.frame_rate_hz,
        local_files_only=args.local_files_only,
    )
    started = perf_counter()
    runtime_model, checkpoint, cpc_path = _build_model(
        args.device,
        local_files_only=args.local_files_only,
        frame_rate_hz=args.frame_rate_hz,
        context_window_s=args.context_window_s,
    )
    model = RealtimeVAPSingleChannelEOTAdapter(
        runtime_model,
        artifact_checksum=f"sha256:{_sha256(checkpoint)}",
        artifact_size_bytes=checkpoint.stat().st_size,
        frame_rate_hz=args.frame_rate_hz,
        context_s=args.context_window_s,
    )
    initialization_time_ms = (perf_counter() - started) * 1_000
    mode = EvaluationMode(
        channel_mode=ChannelMode.SINGLE_CHANNEL_FALLBACK,
        execution_mode=ExecutionMode.STREAMING,
        signal_provenance=SignalProvenance.DERIVED,
        transcript_mode=TranscriptMode.NONE,
        service_mode=ServiceMode.LOCAL,
        language_support=SupportStatus.SUPPORTED,
    )
    run = run_decision_points(
        dataset.examples,
        model,
        mode,
        initialization_time_ms=initialization_time_ms,
    )
    policy = eot_bench_policy_grid()
    metrics = evaluate_eot_sweep(run.predictions, policy)
    model_options = {
        "maai_version": MAAI_VERSION,
        "maai_code_revision": MAAI_CODE_REVISION,
        "maai_wheel_sha256": MAAI_WHEEL_SHA256,
        "checkpoint_repository": MODEL_REPOSITORY,
        "checkpoint_revision": MODEL_REVISION,
        "checkpoint_filename": checkpoint.name,
        "frame_rate_hz": args.frame_rate_hz,
        "context_window_s": args.context_window_s,
        "cpc_sha256": _sha256(cpc_path),
        "dataset_language": args.lang,
        "dataset_revision": args.revision,
        "limit": args.limit,
        "source_channel_mapping": "user=channel_1, silence=channel_2",
        "decision_alignment": "latest-complete-100ms-frame-at-or-before-target",
        "eot_mapping": "1 - p_now[channel_1]",
    }
    run_id = make_run_id(model.descriptor, policy, model_options)
    dataset_metadata = dict(dataset.metadata)
    dataset_metadata["evaluated_rows"] = len(dataset.examples)
    benchmark_metadata = dict(dataset.benchmark_metadata)
    benchmark_metadata["partial_run"] = args.limit is not None
    manifest = build_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        policy=policy,
        model_options=model_options,
        dataset_metadata=dataset_metadata,
        benchmark_metadata=benchmark_metadata,
        execution_metadata={
            "device": args.device,
            "thread_count": 1,
            "batch_size": 1,
            "seed": 0,
            "native_sample_rate": NATIVE_SAMPLE_RATE,
            "frame_rate_hz": args.frame_rate_hz,
            "context_window_s": args.context_window_s,
            "model_license": MODEL_LICENSE,
        },
    )
    # The default configuration keeps its established directory; a checkpoint
    # variant gets its own so the two never share a leaderboard folder.
    variant = ""
    if (args.frame_rate_hz, args.context_window_s) != (FRAME_RATE_HZ, CONTEXT_WINDOW_S):
        variant = f"-{args.frame_rate_hz:g}hz-{args.context_window_s:g}s"
    destination = write_run_artifacts(
        args.output_dir / args.lang / "eotbench" / f"realtime_vap_single_channel_fallback{variant}",
        run_id=run_id,
        manifest=manifest,
        run=run,
        policy=policy,
        metrics=metrics,
    )
    if not args.quiet:
        print(
            f"examples={len(dataset.examples)} decisions={len(run.predictions)} rtf={run.rtf:.6f}",
            file=sys.stderr,
        )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
