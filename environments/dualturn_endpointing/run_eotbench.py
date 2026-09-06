from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_turnbench import (  # noqa: E402
    FRAME_RATE_HZ,
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
    TranscriptMode,
)
from turn_eval.metrics.eot import eot_bench_policy_grid, evaluate_eot_sweep  # noqa: E402
from turn_eval.models.dualturn import DualTurnSingleChannelEOTAdapter  # noqa: E402
from turn_eval.runner.decision_point import run_decision_points  # noqa: E402


class _Progress:
    def __init__(self) -> None:
        self.started = perf_counter()

    def __call__(self, completed: int, total: int) -> None:
        if completed != total and completed % 500:
            return
        elapsed = perf_counter() - self.started
        rate = completed / elapsed if elapsed else 0.0
        eta = (total - completed) / rate if rate else 0.0
        print(
            f"progress={completed}/{total} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            file=sys.stderr,
            flush=True,
        )


def run(args: argparse.Namespace) -> Path:
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    dataset = load_eot_bench(
        args.lang,
        revision=args.revision,
        limit=args.limit,
        inference_interval_s=1 / FRAME_RATE_HZ,
        min_silence_span_s=1 / FRAME_RATE_HZ,
        local_files_only=args.local_files_only,
    )
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
    started = perf_counter()
    session = _session(model_path, args.provider, args.threads)
    model = DualTurnSingleChannelEOTAdapter(lambda: DualTurnStream(session))
    initialization_time_ms = (perf_counter() - started) * 1_000
    language_support = model.descriptor.support_for_language(args.lang)
    mode = EvaluationMode(
        channel_mode=ChannelMode.SINGLE_CHANNEL_FALLBACK,
        execution_mode=ExecutionMode.STREAMING,
        signal_provenance=SignalProvenance.NATIVE,
        transcript_mode=TranscriptMode.NONE,
        service_mode=ServiceMode.LOCAL,
        language_support=language_support,
    )
    run = run_decision_points(
        dataset.examples,
        model,
        mode,
        initialization_time_ms=initialization_time_ms,
        progress=None if args.quiet else _Progress(),
    )
    policy = eot_bench_policy_grid()
    metrics = evaluate_eot_sweep(run.predictions, policy)
    model_options = {
        "checkpoint_repository": MODEL_REPOSITORY,
        "checkpoint_revision": MODEL_REVISION,
        "checkpoint_filename": MODEL_FILENAME,
        "dataset_language": args.lang,
        "dataset_revision": args.revision,
        "limit": args.limit,
        "model_card_languages": ["en"],
        "language_evaluation": language_support.value,
        "source_channel_mapping": "user=CH0, silence=CH1",
        "decision_alignment": "latest-complete-80ms-frame-at-or-before-target",
        "eot_mapping": "native eot_probs[0,0,0]",
        "resampler": "causal-linear-streaming",
    }
    run_id = make_run_id(model.descriptor, policy, model_options)
    dataset_metadata = dict(dataset.metadata)
    dataset_metadata["evaluated_rows"] = len(dataset.examples)
    benchmark_metadata = dict(dataset.benchmark_metadata)
    benchmark_metadata["partial_run"] = args.limit is not None
    benchmark_metadata["primary_leaderboard_eligible"] = args.limit is None
    manifest = build_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        policy=policy,
        model_options=model_options,
        dataset_metadata=dataset_metadata,
        benchmark_metadata=benchmark_metadata,
        execution_metadata={
            "device": args.provider,
            "provider": args.provider,
            "thread_count": args.threads,
            "batch_size": 1,
            "seed": 0,
            "native_sample_rate": SAMPLE_RATE,
            "frame_rate_hz": FRAME_RATE_HZ,
            "algorithmic_lookahead_ms": 0,
            "model_license": MODEL_LICENSE,
        },
    )
    destination = write_run_artifacts(
        args.output_dir / args.lang / "eotbench" / "dualturn_single_channel_fallback",
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
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pinned DualTurn Endpointing on LiveKit eot-bench"
    )
    parser.add_argument("--lang", choices=("en", "zh"), required=True)
    parser.add_argument("--revision", default=EOT_BENCH_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        print(run(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.exit(2, f"DualTurn eot-bench: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
