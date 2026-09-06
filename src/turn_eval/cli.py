from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from turn_eval import __version__
from turn_eval.artifacts import (
    build_manifest,
    build_state_manifest,
    content_hash,
    make_run_id,
    make_state_run_id,
    validate_run_artifacts,
    write_run_artifacts,
    write_state_run_artifacts,
)
from turn_eval.benchmarks.eot_bench import EOT_BENCH_REVISION, load_eot_bench
from turn_eval.benchmarks.fastturn import load_fastturn, prepare_fastturn_dataset
from turn_eval.benchmarks.sid_bench import (
    SID_BENCH_REVISION,
    load_sid_bench,
    load_sid_bench_predictions,
    prepare_sid_bench_dataset,
)
from turn_eval.benchmarks.synthetic import synthetic_decision_point_examples
from turn_eval.benchmarks.turnbench import (
    TURNBENCH_DEV_DATASET,
    TURNBENCH_DEV_REVISION,
    TurnBenchConversationPrediction,
    TurnBenchSpeakerEvents,
    TurnBenchSubmission,
    load_turnbench,
    score_with_official_turnbench,
    sweep_with_official_turnbench,
)
from turn_eval.calibration import (
    make_calibration_split,
    require_heldout_evaluation,
    write_calibration_split,
    write_frozen_policy,
)
from turn_eval.core.types import (
    ChannelMode,
    EvaluationMode,
    ExecutionMode,
    ServiceMode,
    SignalProvenance,
    SupportStatus,
    TranscriptMode,
    TurnState,
)
from turn_eval.metrics.eot import (
    default_policy_grid,
    eot_bench_policy_grid,
    eot_bench_vad_policy_grid,
    evaluate_eot_sweep,
)
from turn_eval.metrics.interruption import evaluate_sid_bench
from turn_eval.metrics.state import evaluate_turn_states
from turn_eval.models.easy_turn import (
    EASY_TURN_QWEN_REPO_ID,
    EASY_TURN_QWEN_REVISION,
    EASY_TURN_REPO_ID,
    EASY_TURN_REVISION,
    EASY_TURN_SOURCE_REVISION,
    EasyTurnFastTurnAdapter,
)
from turn_eval.models.fixed_state import FixedStateAdapter
from turn_eval.models.fixed_timeout import FixedTimeoutAdapter
from turn_eval.models.rms_vad import RMS_VAD_THRESHOLD, RMS_VAD_WINDOW_MS, RMSVADAdapter
from turn_eval.models.smart_turn import (
    SMART_TURN_FASTTURN_THRESHOLD,
    SMART_TURN_REVISION,
    SmartTurnFastTurnAdapter,
    SmartTurnV32Adapter,
)
from turn_eval.models.turnsense import (
    TURNSENSE_REPO_ID,
    TURNSENSE_REVISION,
    TURNSENSE_SOURCE_REVISION,
    TurnSenseFastTurnAdapter,
    TurnSensePointwiseAdapter,
)
from turn_eval.models.x2_turn import (
    X2_TURN_DEFAULT_DELAY_MS,
    X2_TURN_REPO_ID,
    X2_TURN_REVISION,
    X2_TURN_SOURCE_REVISION,
    X2TurnFastTurnAdapter,
)
from turn_eval.reports.comparison import write_comparison_report
from turn_eval.reports.eot_analysis import write_eot_hold_analysis
from turn_eval.reports.fastturn_rescore import (
    EOT_STATES,
    derive_eot_run,
    derived_benchmark_metadata,
    derived_model_options,
    descriptor_from_manifest,
    evaluation_mode_from_manifest,
    read_source_run,
)
from turn_eval.reports.plots import write_suite_plots
from turn_eval.reports.runtime import write_runtime_report
from turn_eval.reports.sid_analysis import write_sid_paired_analysis
from turn_eval.reports.sid_calibration import (
    FAST_THRESHOLD_GRID,
    PERSISTENCE_GRID,
    THRESHOLD_GRID,
    load_traces,
    render_report,
    score,
    search,
)
from turn_eval.reports.summary import recompute_run_metrics, render_markdown
from turn_eval.runner.clip_classification import ClipClassificationRun, run_clip_classification
from turn_eval.runner.decision_point import run_decision_points


class _CliProgress:
    def __init__(self, *, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("progress interval must be positive")
        self.interval_s = interval_s
        self.started = perf_counter()
        self.last_update = self.started

    def __call__(self, completed: int, total: int) -> None:
        now = perf_counter()
        if completed != total and now - self.last_update < self.interval_s:
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed else 0.0
        eta = (total - completed) / rate if rate else 0.0
        percent = completed * 100 / total if total else 100.0
        print(
            f"progress: {completed}/{total} ({percent:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
            file=sys.stderr,
        )
        self.last_update = now


def _write_json_file(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_mode(
    signal_provenance: SignalProvenance = SignalProvenance.NATIVE,
    language_support: SupportStatus = SupportStatus.SUPPORTED,
) -> EvaluationMode:
    return EvaluationMode(
        channel_mode=ChannelMode.NATIVE_SINGLE,
        execution_mode=ExecutionMode.POINTWISE,
        signal_provenance=signal_provenance,
        transcript_mode=TranscriptMode.NONE,
        service_mode=ServiceMode.LOCAL,
        language_support=language_support,
    )


def _run_synthetic(args: argparse.Namespace) -> int:
    started = perf_counter()
    model = FixedTimeoutAdapter(
        timeout_ms=args.model_timeout_ms,
        lookahead_ms=args.lookahead_ms,
    )
    initialization_time_ms = (perf_counter() - started) * 1_000
    policy = default_policy_grid()
    run = run_decision_points(
        synthetic_decision_point_examples(),
        model,
        _evaluation_mode(),
        initialization_time_ms=initialization_time_ms,
    )
    metrics = evaluate_eot_sweep(run.predictions, policy)
    model_options = {
        "timeout_ms": args.model_timeout_ms,
        "lookahead_ms": args.lookahead_ms,
    }
    run_id = make_run_id(model.descriptor, policy, model_options)
    manifest = build_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=_evaluation_mode(),
        policy=policy,
        model_options=model_options,
    )
    destination = write_run_artifacts(
        args.output_dir / "synthetic-eot",
        run_id=run_id,
        manifest=manifest,
        run=run,
        policy=policy,
        metrics=metrics,
    )
    print(destination)
    return 0


def _run_eot_bench_vap_subprocess(args: argparse.Namespace) -> int:
    frame_rate_hz = args.vap_frame_rate_hz or 10.0
    native_grid_ms = 1_000.0 / frame_rate_hz
    if args.inference_interval_ms != native_grid_ms or args.min_silence_span_ms != native_grid_ms:
        raise ValueError(
            f"Realtime-VAP at {frame_rate_hz:g} Hz uses a native "
            f"{native_grid_ms:g} ms eot-bench decision grid"
        )
    project = Path("environments/realtime_vap").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_eotbench.py"),
        "--lang",
        args.lang,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    if args.vap_frame_rate_hz is not None:
        command.extend(("--frame-rate-hz", str(args.vap_frame_rate_hz)))
    if args.vap_context_window_s is not None:
        command.extend(("--context-window-s", str(args.vap_context_window_s)))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Realtime-VAP backend exited with status {completed.returncode}")
    return 0


def _run_eot_bench_dualturn_subprocess(args: argparse.Namespace) -> int:
    if args.inference_interval_ms != 80.0 or args.min_silence_span_ms != 80.0:
        raise ValueError("DualTurn uses its native 80 ms eot-bench decision grid")
    project = Path("environments/dualturn_endpointing").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_eotbench.py"),
        "--lang",
        args.lang,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--provider",
        args.provider,
        "--threads",
        str(args.threads),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"DualTurn backend exited with status {completed.returncode}")
    return 0


def _run_eot_bench_turnsense_subprocess(args: argparse.Namespace) -> int:
    project = Path("environments/turnsense").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "turn-eval",
        "run-eot-bench",
        "--lang",
        args.lang,
        "--revision",
        args.revision,
        "--model",
        "turnsense",
        "--turnsense-revision",
        args.turnsense_revision,
        "--threads",
        str(args.threads),
        "--output-dir",
        str(args.output_dir),
        "--inference-interval-ms",
        str(args.inference_interval_ms),
        "--min-silence-span-ms",
        str(args.min_silence_span_ms),
        "--progress-interval-s",
        str(args.progress_interval_s),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["TURN_EVAL_TURNSENSE_BACKEND"] = "1"
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"TurnSense backend exited with status {completed.returncode}")
    return 0


def _run_eot_bench(args: argparse.Namespace) -> int:
    if args.model == "dualturn":
        return _run_eot_bench_dualturn_subprocess(args)
    if args.model == "realtime-vap":
        return _run_eot_bench_vap_subprocess(args)
    if args.model == "turnsense" and os.environ.get("TURN_EVAL_TURNSENSE_BACKEND") != "1":
        return _run_eot_bench_turnsense_subprocess(args)
    dataset = load_eot_bench(
        args.lang,
        revision=args.revision,
        limit=args.limit,
        inference_interval_s=args.inference_interval_ms / 1_000,
        min_silence_span_s=args.min_silence_span_ms / 1_000,
        local_files_only=args.local_files_only,
    )
    started = perf_counter()
    if args.model == "fixed-timeout":
        model = FixedTimeoutAdapter(
            timeout_ms=args.model_timeout_ms,
            sample_rate=16_000,
            lookahead_ms=args.lookahead_ms,
        )
    elif args.model == "turnsense":
        if args.lookahead_ms != 0:
            raise ValueError("TurnSense declares zero lookahead; --lookahead-ms is invalid")
        model = TurnSensePointwiseAdapter(
            revision=args.turnsense_revision,
            threads=args.threads,
            local_files_only=args.local_files_only,
        )
    else:
        if args.lookahead_ms != 0:
            raise ValueError("SmartTurn v3.2 declares zero lookahead; --lookahead-ms is invalid")
        model = SmartTurnV32Adapter(
            revision=args.model_revision,
            intra_op_threads=args.threads,
            local_files_only=args.local_files_only,
        )
    initialization_time_ms = (perf_counter() - started) * 1_000
    mode = _evaluation_mode(
        SignalProvenance.DERIVED if args.model == "fixed-timeout" else SignalProvenance.NATIVE,
        # TurnSense declares Chinese support only, so an English run is an
        # explicitly unverified zero-shot row.
        SupportStatus.UNVERIFIED_ZERO_SHOT
        if args.model == "turnsense" and args.lang not in model.descriptor.supported_languages
        else SupportStatus.SUPPORTED,
    )
    policy = (
        eot_bench_vad_policy_grid()
        if args.model == "fixed-timeout"
        else eot_bench_policy_grid(score_point_ms=model.descriptor.natural_score_point_ms)
    )
    run = run_decision_points(
        dataset.examples,
        model,
        mode,
        initialization_time_ms=initialization_time_ms,
        score_point_s=(
            None
            if model.descriptor.natural_score_point_ms is None
            else model.descriptor.natural_score_point_ms / 1_000
        ),
        progress=(None if args.quiet else _CliProgress(interval_s=args.progress_interval_s)),
    )
    metrics = evaluate_eot_sweep(run.predictions, policy)
    model_options = {
        "model": args.model,
        "model_revision": model.descriptor.revision,
        "dataset_language": args.lang,
        "dataset_revision": args.revision,
        "limit": args.limit,
        "inference_interval_ms": args.inference_interval_ms,
        "min_silence_span_ms": args.min_silence_span_ms,
    }
    if args.model == "fixed-timeout":
        model_options.update(
            {
                "timeout_ms": args.model_timeout_ms,
                "lookahead_ms": args.lookahead_ms,
            }
        )
    else:
        model_options["onnx_intra_op_threads"] = args.threads
    if args.model == "turnsense":
        model_options.update(
            {
                "turnsense_source_revision": TURNSENSE_SOURCE_REVISION,
                "eot_mapping": "raw P(complete); invalid mass left in the denominator",
                "invalid_policy": "no-renormalisation-low-confidence-keeps-waiting",
                "invalid_policy_selected_on": "predeclared, not selected on eot-bench",
            }
        )
    run_id = make_run_id(model.descriptor, policy, model_options)
    manifest = build_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        policy=policy,
        model_options=model_options,
        dataset_metadata=dataset.metadata,
        benchmark_metadata=dataset.benchmark_metadata,
        execution_metadata={
            "device": "cpu",
            "thread_count": (args.threads if args.model in {"smart-turn-v3.2", "turnsense"} else 1),
            "batch_size": 1,
            "seed": 0,
        },
    )
    destination = write_run_artifacts(
        args.output_dir / args.lang / "eotbench" / model.descriptor.name.replace("-", "_"),
        run_id=run_id,
        manifest=manifest,
        run=run,
        policy=policy,
        metrics=metrics,
    )
    print(destination)
    return 0


def _report(args: argparse.Namespace) -> int:
    metrics = recompute_run_metrics(args.run_dir)
    stored = json.loads((args.run_dir / "metrics.json").read_text())
    if metrics != stored:
        raise ValueError("stored metrics differ from metrics recomputed from predictions")
    markdown = render_markdown(args.run_dir, metrics)
    if args.output is None:
        print(markdown, end="")
    else:
        run_path = args.run_dir.resolve()
        output_path = args.output.resolve()
        if output_path.is_relative_to(run_path):
            raise ValueError("reports must not mutate immutable run directories")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(args.output)
    return 0


def _validate(args: argparse.Namespace) -> int:
    validate_run_artifacts(args.run_dir)
    print(f"valid: {args.run_dir}")
    return 0


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _validate_turnbench_submission(args: argparse.Namespace) -> int:
    submission = TurnBenchSubmission.load(args.predictions)
    if args.conversation_ids is not None:
        conversation_ids = _load_json(args.conversation_ids)
        if not isinstance(conversation_ids, list) or not all(
            isinstance(item, str) for item in conversation_ids
        ):
            raise ValueError("conversation IDs file must be a JSON list of strings")
        submission.validate_coverage(conversation_ids)
    if args.durations is not None:
        raw_durations = _load_json(args.durations)
        if not isinstance(raw_durations, dict) or not all(
            isinstance(key, str) and not isinstance(value, bool) and isinstance(value, int | float)
            for key, value in raw_durations.items()
        ):
            raise ValueError("durations file must be a JSON object of numeric seconds")
        submission.validate_durations({key: float(value) for key, value in raw_durations.items()})
    print(f"valid: {args.predictions} ({len(submission.predictions)} conversations)")
    return 0


def _score_turnbench(args: argparse.Namespace) -> int:
    TurnBenchSubmission.load(args.predictions)
    result = score_with_official_turnbench(
        args.predictions,
        dataset=args.dataset,
        revision=args.revision,
        scorer_project=args.scorer_project,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


def _run_turnbench_vap(args: argparse.Namespace) -> int:
    project = Path("environments/realtime_vap").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    command.extend(("--shard-index", str(args.shard_index)))
    command.extend(("--shard-count", str(args.shard_count)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Realtime-VAP backend exited with status {completed.returncode}")
    return 0


def _run_sid_bench_vap(args: argparse.Namespace) -> int:
    project = Path("environments/realtime_vap").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_sidbench.py"),
        "--split",
        args.split,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--threshold",
        str(args.threshold),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"Realtime-VAP SID-Bench backend exited with status {completed.returncode}"
        )
    return 0


def _run_sid_bench_vap_gpt(args: argparse.Namespace) -> int:
    project = Path("environments/vap_turnbench").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_sidbench.py"),
        "--split",
        args.split,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--threshold",
        str(args.threshold),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"VapGPT OTO SID-Bench backend exited with status {completed.returncode}"
        )
    return 0


def _run_sid_bench_kyutai(args: argparse.Namespace) -> int:
    project = Path("environments/kyutai_semantic_vad").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_sidbench.py"),
        "--split",
        args.split,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--threshold",
        str(args.threshold),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Kyutai SID-Bench backend exited with status {completed.returncode}")
    return 0


def _run_sid_bench_dualturn(args: argparse.Namespace) -> int:
    project = Path("environments/dualturn_endpointing").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_sidbench.py"),
        "--split",
        args.split,
        "--output-dir",
        str(args.output_dir),
        "--provider",
        args.provider,
        "--threads",
        str(args.threads),
        "--threshold",
        str(args.threshold),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"DualTurn SID-Bench backend exited with status {completed.returncode}")
    return 0


def _profile_runtime(args: argparse.Namespace) -> int:
    projects = {
        "dualturn": Path("environments/dualturn_endpointing").resolve(),
        "realtime-vap": Path("environments/realtime_vap").resolve(),
    }
    project = projects[args.model]
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "profile_runtime.py"),
        "--output",
        str(args.output),
        "--threads",
        str(args.threads),
        "--warmup-frames",
        str(args.warmup_frames),
        "--measurement-frames",
        str(args.measurement_frames),
        "--sampling-interval-ms",
        str(args.sampling_interval_ms),
        "--seed",
        str(args.seed),
    ]
    if args.model == "dualturn":
        command.extend(("--provider", args.backend))
    else:
        command.extend(("--device", "cuda:0" if args.backend == "cuda" else "cpu"))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"{args.model} runtime profiler exited with status {completed.returncode}"
        )
    return 0


def _report_runtime(args: argparse.Namespace) -> int:
    destination = write_runtime_report(args.profiles, args.output_dir)
    print(destination)
    return 0


def _report_suite(args: argparse.Namespace) -> int:
    languages = {"en", "zh"} if args.lang == "all" else {args.lang}
    destination = write_comparison_report(
        results_root=args.results_root,
        output_dir=args.output_dir,
        languages=languages,
        runtime_summary=args.runtime_summary,
    )
    print(destination)
    return 0


def _plot_suite(args: argparse.Namespace) -> int:
    destination = write_suite_plots(args.comparison_summary, args.output_dir)
    print(destination)
    return 0


def _run_turnbench_vap_gpt(args: argparse.Namespace) -> int:
    project = Path("environments/vap_turnbench").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    command.extend(("--shard-index", str(args.shard_index)))
    command.extend(("--shard-count", str(args.shard_count)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"VapGPT backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_dualturn(args: argparse.Namespace) -> int:
    project = Path("environments/dualturn_endpointing").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--provider",
        args.provider,
        "--threads",
        str(args.threads),
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"DualTurn backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_espnet(args: argparse.Namespace) -> int:
    project = Path("environments/espnet_turntaking").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--tf32" if args.tf32 else "--no-tf32",
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"ESPnet TurnTaking backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_mimi(args: argparse.Namespace) -> int:
    project = Path("environments/mimi_endpointer").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Mimi Endpointer backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_kyutai(args: argparse.Namespace) -> int:
    project = Path("environments/kyutai_semantic_vad").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Kyutai Semantic VAD backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_smart_turn(args: argparse.Namespace) -> int:
    project = Path("environments/smart_turn_turnbench").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "python",
        str(project / "run_turnbench.py"),
        "--version",
        args.model_version,
        "--artifact",
        args.artifact,
        "--provider",
        args.provider,
        "--workers",
        str(args.workers),
        "--dataset",
        args.dataset,
        "--revision",
        args.revision,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.local_files_only:
        command.append("--local-files-only")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"SmartTurn backend exited with status {completed.returncode}")
    return 0


def _run_turnbench_rms_vad(args: argparse.Namespace) -> int:
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    model = RMSVADAdapter(window_ms=args.window_ms, threshold=args.threshold)
    predictions: list[TurnBenchConversationPrediction] = []
    durations: dict[str, float] = {}
    inference_wall_s = 0.0
    audio_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        started = perf_counter()
        speaker_1 = model.predict(conversation.speaker_1_audio)
        speaker_2 = model.predict(conversation.speaker_2_audio)
        inference_wall_s += perf_counter() - started
        predictions.append(
            TurnBenchConversationPrediction(
                conversation_id=conversation.conversation_id,
                speaker_1=TurnBenchSpeakerEvents(
                    eot=speaker_1.eot,
                    interruption=speaker_1.interruption,
                ),
                speaker_2=TurnBenchSpeakerEvents(
                    eot=speaker_2.eot,
                    interruption=speaker_2.interruption,
                ),
            )
        )
        durations[conversation.conversation_id] = conversation.duration_s
        audio_s += conversation.duration_s
        if not args.quiet:
            print(
                f"conversation={conversation.conversation_id} completed={index} "
                f"duration_s={conversation.duration_s:.1f}",
                file=sys.stderr,
            )
    if not predictions:
        raise RuntimeError("RMS VAD produced no TurnBench predictions")

    submission = TurnBenchSubmission(tuple(predictions))
    submission.validate_durations(durations)
    options = {
        "dataset": dataset.source,
        "dataset_revision": dataset.revision,
        "window_ms": model.window_ms,
        "threshold": model.threshold,
        "limit": args.limit,
    }
    created = datetime.now(UTC)
    run_id = f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{content_hash(options)[:8]}"
    destination = args.output_dir.resolve() / "en" / "turnbench" / "rms_vad" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    predictions_path = destination / "predictions-dev.json"
    submission.write(predictions_path)

    scores = None
    leaderboard_eligible = (
        args.limit is None
        and args.dataset == TURNBENCH_DEV_DATASET
        and dataset.revision == TURNBENCH_DEV_REVISION
        and model.window_ms == RMS_VAD_WINDOW_MS
        and model.threshold == RMS_VAD_THRESHOLD
    )
    if not args.no_score and args.limit is None:
        scores = score_with_official_turnbench(
            predictions_path,
            dataset=args.dataset,
            revision=args.revision,
            scorer_project=args.scorer_project,
        )
        (destination / "scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "schema_version": 1,
        "created_at": created.isoformat(),
        "benchmark": {
            "name": "turnbench",
            "scorer_revision": "e02d547803fd0e45221afcaeb1cd24b61ae7568e",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(predictions),
            "leaderboard_eligible": leaderboard_eligible,
            "partial_run": args.limit is not None,
        },
        "model": {
            "name": model.descriptor.name,
            "revision": model.descriptor.revision,
            "model_license": model.descriptor.model_license,
            "code_license": model.descriptor.code_license,
            "artifact_size_bytes": 0,
        },
        "evaluation": {
            "mode": "native-dual-channel",
            "execution": "streaming",
            "channel_processing": "independent",
            "window_ms": model.window_ms,
            "threshold": model.threshold,
            "event_timestamp": "revealing-window-end",
            "eot_mapping": "speech-offset",
            "interruption_mapping": "speech-onset",
            "signal_provenance": {"eot": "derived", "interruption": "derived"},
        },
        "runtime": {
            "device": "cpu",
            "inference_s": inference_wall_s,
            "audio_s": audio_s,
            "rtf": inference_wall_s / audio_s if audio_s else None,
        },
        "environment": {"hf_home": os.environ.get("HF_HOME")},
        "official_score_written": scores is not None,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "SUCCESS").write_text("\n", encoding="utf-8")
    print(destination)
    return 0


def _sweep_turnbench(args: argparse.Namespace) -> int:
    result = sweep_with_official_turnbench(
        args.eot_probabilities,
        args.interruption_probabilities,
        args.output_dir,
        fp_budget=args.fp_budget,
        dataset=args.dataset,
        revision=args.revision,
        scorer_project=args.scorer_project,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _prepare_fastturn(args: argparse.Namespace) -> int:
    root = prepare_fastturn_dataset(
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    dataset = load_fastturn(root)
    print(
        json.dumps(
            {
                "root": str(root),
                "dataset": dataset.metadata,
                "benchmark": dataset.benchmark_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_fastturn_x2_subprocess(args: argparse.Namespace) -> int:
    project = Path("environments/x2_turn").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "turn-eval",
        "run-fastturn",
        "--suite",
        args.suite,
        "--model",
        "x2-turn",
        "--model-revision",
        args.model_revision,
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--delay-ms",
        str(args.delay_ms),
        "--output-dir",
        str(args.output_dir),
        "--progress-interval-s",
        str(args.progress_interval_s),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["TURN_EVAL_X2_BACKEND"] = "1"
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"X2-Turn backend exited with status {completed.returncode}")
    return 0


def _run_fastturn_turnsense_subprocess(args: argparse.Namespace) -> int:
    project = Path("environments/turnsense").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "turn-eval",
        "run-fastturn",
        "--suite",
        args.suite,
        "--model",
        "turnsense",
        "--turnsense-revision",
        args.turnsense_revision,
        "--threads",
        str(args.threads),
        "--output-dir",
        str(args.output_dir),
        "--progress-interval-s",
        str(args.progress_interval_s),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["TURN_EVAL_TURNSENSE_BACKEND"] = "1"
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"TurnSense backend exited with status {completed.returncode}")
    return 0


def _run_fastturn_easy_turn_subprocess(args: argparse.Namespace) -> int:
    project = Path("environments/easy_turn").resolve()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "--frozen",
        "turn-eval",
        "run-fastturn",
        "--suite",
        args.suite,
        "--model",
        "easy-turn",
        "--easy-turn-revision",
        args.easy_turn_revision,
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
        "--progress-interval-s",
        str(args.progress_interval_s),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.easy_turn_source_root is not None:
        command.extend(("--easy-turn-source-root", str(args.easy_turn_source_root.resolve())))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.quiet:
        command.append("--quiet")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["TURN_EVAL_EASY_TURN_BACKEND"] = "1"
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Easy Turn backend exited with status {completed.returncode}")
    return 0


def _run_fastturn(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.batch_size > 1 and args.model != "easy-turn":
        raise ValueError("batched FastTurn inference is currently supported only by Easy Turn")
    if args.model == "smart-turn-v3.2" and args.suite != "eot":
        raise ValueError("SmartTurn supports only fastturn-eot")
    if args.model == "x2-turn" and os.environ.get("TURN_EVAL_X2_BACKEND") != "1":
        if args.suite == "full":
            raise ValueError("X2-Turn cannot run fastturn-full: Wait has no native class")
        return _run_fastturn_x2_subprocess(args)
    if args.model == "turnsense" and os.environ.get("TURN_EVAL_TURNSENSE_BACKEND") != "1":
        if args.suite != "eot":
            raise ValueError("TurnSense supports only fastturn-eot")
        return _run_fastturn_turnsense_subprocess(args)
    if args.model == "easy-turn" and os.environ.get("TURN_EVAL_EASY_TURN_BACKEND") != "1":
        return _run_fastturn_easy_turn_subprocess(args)

    root = prepare_fastturn_dataset(
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    dataset = load_fastturn(root)
    labels_by_suite = {
        "eot": (TurnState.COMPLETE, TurnState.INCOMPLETE),
        "real": (TurnState.COMPLETE, TurnState.INCOMPLETE, TurnState.BACKCHANNEL),
        "full": tuple(TurnState),
    }
    labels = labels_by_suite[args.suite]
    model_started = perf_counter()
    if args.model == "fixed-state":
        fixed_label = TurnState(args.fixed_state)
        model = FixedStateAdapter(fixed_label, labels)
    elif args.model == "x2-turn":
        model = X2TurnFastTurnAdapter(
            labels,
            revision=args.model_revision,
            device=args.device,
            delay_ms=args.delay_ms,
            local_files_only=args.local_files_only,
            model_path=args.model_path,
        )
    elif args.model == "easy-turn":
        model = EasyTurnFastTurnAdapter(
            labels,
            revision=args.easy_turn_revision,
            device=args.device,
            source_root=args.easy_turn_source_root,
            local_files_only=args.local_files_only,
        )
    elif args.model == "smart-turn-v3.2":
        model = SmartTurnFastTurnAdapter(
            labels,
            threshold=args.smart_turn_threshold,
            intra_op_threads=args.threads,
            local_files_only=args.local_files_only,
        )
    else:
        model = TurnSenseFastTurnAdapter(
            labels,
            revision=args.turnsense_revision,
            threads=args.threads,
            local_files_only=args.local_files_only,
        )
    initialization_time_ms = (perf_counter() - model_started) * 1_000
    mode = EvaluationMode(
        channel_mode=ChannelMode.NATIVE_SINGLE,
        execution_mode=(
            ExecutionMode.CAUSAL_OFFLINE if args.model == "x2-turn" else ExecutionMode.OFFLINE
        ),
        signal_provenance=(
            SignalProvenance.DERIVED
            if args.model in {"fixed-state", "x2-turn"}
            else SignalProvenance.NATIVE
        ),
        transcript_mode=TranscriptMode.NONE,
        service_mode=ServiceMode.LOCAL,
        language_support=SupportStatus.SUPPORTED,
    )
    eligible_rows = sum(dataset.metadata["class_counts"][label.value] for label in labels)
    expected_rows = min(args.limit, eligible_rows) if args.limit is not None else eligible_rows
    examples = dataset.iter_examples(labels=labels, limit=args.limit)
    run = run_clip_classification(
        examples,
        model,
        mode,
        initialization_time_ms=initialization_time_ms,
        progress=(None if args.quiet else _CliProgress(interval_s=args.progress_interval_s)),
        total_examples=expected_rows,
        batch_size=args.batch_size,
    )
    metrics = evaluate_turn_states(run.predictions, labels)
    if args.model == "fixed-state":
        label_mapping = {
            "strategy": "diagnostic_constant_prediction",
            "output_labels": [label.value for label in labels],
            "constant_prediction": fixed_label.value,
            "calibration_source": "none-diagnostic",
            "split_role": "none",
        }
        model_options = {
            "suite": args.suite,
            "fixed_state": fixed_label.value,
            "limit": args.limit,
        }
    elif args.model == "x2-turn":
        label_mapping = {
            "strategy": "paper_last_non_idle_exact_only_v1",
            "output_labels": [label.value for label in labels],
            "native_mapping": {
                "turn_end": "complete",
                "speaking": "incomplete",
                "backchannel": "backchannel",
            },
            "abstain_states": ["idle", "noidle", "uncertain"],
            "abstain_scoring": "closed-set-incorrect",
            "wait_mapping": "unsupported",
            "calibration_source": "none-predeclared-from-x2-paper",
            "split_role": "none",
        }
        model_options = {
            "suite": args.suite,
            "limit": args.limit,
            "model_repo": X2_TURN_REPO_ID,
            "model_revision": args.model_revision,
            "source_revision": X2_TURN_SOURCE_REVISION,
            "model_path": args.model_path,
            "target_delay_ms": args.delay_ms,
            "frontend_lookahead_ms": 2.5,
            "frontend_lookback_ms": 52.5,
            "inference_backend": "transformers-two-pass",
            "transcript_input": "model-generated-only",
            "fastturn_train_overlap": "unknown-not-auditable",
        }
    elif args.model == "turnsense":
        label_mapping = {
            "strategy": "native_complete_incomplete_with_invalid_abstention_v1",
            "output_labels": [label.value for label in labels],
            "native_mapping": {"complete": "complete", "incomplete": "incomplete"},
            "abstain_states": ["invalid"],
            "abstain_scoring": "closed-set-incorrect",
            "backchannel_mapping": "unsupported",
            "wait_mapping": "unsupported",
            "calibration_source": "none-native-argmax",
            "split_role": "none",
        }
        model_options = {
            "suite": args.suite,
            "limit": args.limit,
            "model_repo": TURNSENSE_REPO_ID,
            "model_revision": args.turnsense_revision,
            "source_revision": TURNSENSE_SOURCE_REVISION,
            "artifact": "TenSense/model_int8.onnx",
            "context_selection": "last-8-seconds",
            "inference_backend": "onnxruntime-cpu",
            "transcript_input": "none-audio-only",
            "fastturn_train_overlap": "unknown-not-auditable",
        }
    elif args.model == "smart-turn-v3.2":
        label_mapping = {
            "strategy": "fixed_native_eot_probability_threshold_v1",
            "output_labels": [label.value for label in labels],
            "native_mapping": {
                "eot_probability_at_or_above_threshold": "complete",
                "eot_probability_below_threshold": "incomplete",
            },
            "threshold": args.smart_turn_threshold,
            "threshold_source": "predeclared-upstream-default-not-fastturn-calibrated",
            "calibration_source": "none",
            "split_role": "none",
        }
        model_options = {
            "suite": args.suite,
            "limit": args.limit,
            "model_repo": "pipecat-ai/smart-turn-v3",
            "model_revision": SMART_TURN_REVISION,
            "artifact": "smart-turn-v3.2-cpu.onnx",
            "context_selection": "last-8-seconds-left-padded",
            "inference_backend": "onnxruntime-cpu",
            "resampling": "soxr-hq",
            "transcript_input": "none-audio-only",
            "fastturn_train_overlap": "unknown-not-auditable",
        }
    else:
        label_mapping = {
            "strategy": "native_terminal_tag_exact_only_v1",
            "output_labels": [label.value for label in labels],
            "native_mapping": {
                "complete": "complete",
                "incomplete": "incomplete",
                "backchannel": "backchannel",
                "wait": "wait",
            },
            "invalid_outputs": "missing-multiple-or-nonterminal-tag",
            "out_of_suite_labels": "abstain-closed-set-incorrect",
            "score_semantics": "hard-decoded-one-hot-not-probabilities",
            "calibration_source": "none-native-generation",
            "split_role": "none",
        }
        model_options = {
            "suite": args.suite,
            "limit": args.limit,
            "model_repo": EASY_TURN_REPO_ID,
            "model_revision": args.easy_turn_revision,
            "source_revision": EASY_TURN_SOURCE_REVISION,
            "qwen_repo": EASY_TURN_QWEN_REPO_ID,
            "qwen_revision": EASY_TURN_QWEN_REVISION,
            "inference_backend": "upstream-wenet-pytorch",
            "batching": "equivalent-left-padded-generation-v1",
            "batch_size": args.batch_size,
            "prompt_variant": 0,
            "transcript_input": "none-model-generated-only",
            "fastturn_train_overlap": "unknown-not-auditable",
        }
    run_id = make_state_run_id(model.descriptor, label_mapping, model_options)
    dataset_metadata = dict(dataset.metadata)
    dataset_metadata["evaluated_rows"] = len(run.predictions)
    benchmark_metadata = dict(dataset.benchmark_metadata)
    benchmark_metadata["name"] = f"fastturn-{args.suite}"
    benchmark_metadata["model_team_authored_benchmark"] = (
        model.descriptor.name.startswith("fastturn-") or args.model == "easy-turn"
    )
    benchmark_metadata["partial_run"] = len(run.predictions) != eligible_rows
    manifest = build_state_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        label_mapping=label_mapping,
        model_options=model_options,
        dataset_metadata=dataset_metadata,
        benchmark_metadata=benchmark_metadata,
        execution_metadata={
            "device": (
                "cpu"
                if args.model in {"fixed-state", "turnsense", "smart-turn-v3.2"}
                else args.device
            ),
            "thread_count": (args.threads if args.model in {"turnsense", "smart-turn-v3.2"} else 1),
            "batch_size": args.batch_size,
            "seed": 0,
        },
    )
    destination = write_state_run_artifacts(
        args.output_dir / "zh" / "fastturn" / args.suite / model.descriptor.name,
        run_id=run_id,
        manifest=manifest,
        run=run,
        label_mapping=label_mapping,
        metrics=metrics,
    )
    print(destination)
    return 0


def _rescore_fastturn_eot(args: argparse.Namespace) -> int:
    source = read_source_run(args.run)
    derived = derive_eot_run(source)
    metrics = evaluate_turn_states(derived.records, EOT_STATES)

    source_runtime = json.loads((source.run_dir / "runtime.json").read_text(encoding="utf-8"))
    # Wall-clock and audio totals describe the full source suite. They are
    # inherited rather than split, because per-clip audio duration is not
    # recorded in the artifact and a scaled estimate would be invented. Only the
    # latency percentiles are recomputed, from the derived subset's own records.
    run = ClipClassificationRun(
        predictions=derived.records,
        initialization_time_ms=source_runtime["initialization_time_ms"],
        inference_wall_s=source_runtime["inference_wall_s"],
        resampling_wall_s=source_runtime["resampling_wall_s"],
        audio_s=source_runtime["audio_s"],
    )
    derivation = dict(derived.derivation)
    derivation["runtime_aggregates"] = "inherited-from-source-run-not-subset-measured"
    derivation["runtime_latency_percentiles"] = "recomputed-on-derived-subset"

    descriptor = descriptor_from_manifest(source.manifest)
    mode = evaluation_mode_from_manifest(source.manifest)
    model_options = derived_model_options(source.manifest, derivation)
    run_id = make_state_run_id(descriptor, derived.label_mapping, model_options)

    dataset_metadata = dict(source.manifest["dataset"])
    class_counts = dict(dataset_metadata.get("class_counts", {}))
    dataset_metadata["class_counts"] = {
        state.value: class_counts[state.value]
        for state in EOT_STATES
        if state.value in class_counts
    }
    dataset_metadata["evaluated_rows"] = len(derived.records)

    manifest = build_state_manifest(
        run_id=run_id,
        model=descriptor,
        mode=mode,
        label_mapping=derived.label_mapping,
        model_options=model_options,
        dataset_metadata=dataset_metadata,
        benchmark_metadata=derived_benchmark_metadata(source.manifest, derived.records),
        execution_metadata=dict(source.manifest["execution"]),
    )
    destination = write_state_run_artifacts(
        args.output_dir / "zh" / "fastturn" / "eot" / descriptor.name,
        run_id=run_id,
        manifest=manifest,
        run=run,
        label_mapping=derived.label_mapping,
        metrics=metrics,
    )
    print(destination)
    return 0


def _calibrate_sid_bench(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"calibration artifact already exists: {output}")

    traces = load_traces(args.run)
    source_manifest = json.loads((traces.source_run / "manifest.json").read_text(encoding="utf-8"))
    split_name = source_manifest["dataset"]["split"]
    root = prepare_sid_bench_dataset(
        split_name,
        with_audio=False,
        local_files_only=args.local_files_only,
    )
    dataset = load_sid_bench(root, split_name)
    missing = {item.id for item in dataset.items} - set(traces.traces)
    if missing:
        raise ValueError(f"resume cache is missing {len(missing)} clips; cannot calibrate")

    split = make_calibration_split(
        dataset=f"sid-bench-{split_name}",
        dataset_revision=SID_BENCH_REVISION,
        groups={item.id: item.category for item in dataset.items},
        seed=args.seed,
        fraction=args.fraction,
        group_field="example_id",
    )
    policy, sweep = search(
        dataset,
        traces,
        split,
        thresholds=THRESHOLD_GRID,
        persistence_grid=PERSISTENCE_GRID,
        fast_thresholds=FAST_THRESHOLD_GRID,
        nonbreak_budget=args.nonbreak_budget,
    )

    def scored(
        ids: tuple[str, ...],
        threshold: float,
        persistence: int,
        fast: float | None = None,
    ) -> dict[str, object]:
        return score(
            dataset,
            traces,
            ids,
            threshold=threshold,
            persistence_frames=persistence,
            fast_threshold=fast,
        )

    incumbent = (traces.incumbent_threshold, 1)
    calibration_incumbent = scored(split.calibration_ids, *incumbent)
    # Selection is finished; only now is the held-out side touched, and the
    # guard refuses if the policy was not frozen on this exact split.
    require_heldout_evaluation(policy, split, split.heldout_ids)
    frozen_fast = policy.parameters["fast_threshold"]
    heldout_frozen = scored(
        split.heldout_ids,
        float(policy.parameters["threshold"]),
        int(policy.parameters["persistence_frames"]),
        None if frozen_fast is None else float(frozen_fast),
    )
    heldout_incumbent = scored(split.heldout_ids, *incumbent)

    temporary = output.with_name(f".{output.name}-partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        _write_calibration_artifact(
            temporary,
            split=split,
            policy=policy,
            traces=traces,
            args=args,
            calibration_incumbent=calibration_incumbent,
            heldout_frozen=heldout_frozen,
            heldout_incumbent=heldout_incumbent,
            incumbent=incumbent,
            sweep=sweep,
        )
    except BaseException:
        # A partial artifact must never survive a failure; it would be
        # indistinguishable from a completed one apart from its name.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    temporary.rename(output)
    print(output)
    return 0


def _write_calibration_artifact(
    temporary: Path,
    *,
    split: object,
    policy: object,
    traces: object,
    args: argparse.Namespace,
    calibration_incumbent: dict[str, object],
    heldout_frozen: dict[str, object],
    heldout_incumbent: dict[str, object],
    incumbent: tuple[float, int],
    sweep: list[dict[str, object]],
) -> None:
    write_calibration_split(temporary / "split.json", split)
    write_frozen_policy(temporary / "policy.json", policy)
    _write_json_file(temporary / "sweep.json", sweep)
    _write_json_file(
        temporary / "heldout.json",
        {
            "frozen_policy": heldout_frozen,
            "incumbent": heldout_incumbent,
            "incumbent_parameters": {"threshold": incumbent[0], "persistence_frames": 1},
        },
    )
    _write_json_file(
        temporary / "manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_run": str(traces.source_run),
            "source_manifest_sha256": _sha256_path(traces.source_run / "manifest.json"),
            "source_policy_sha256": _sha256_path(traces.source_run / "policy.json"),
            "source_metrics_sha256": _sha256_path(traces.source_run / "metrics.json"),
            "inference_rerun": False,
            "nonbreak_budget": args.nonbreak_budget,
            "threshold_grid": list(THRESHOLD_GRID),
            "persistence_grid": list(PERSISTENCE_GRID),
            "fast_threshold_grid": list(FAST_THRESHOLD_GRID),
        },
    )
    (temporary / "report.md").write_text(
        render_report(
            split=split,
            policy=policy,
            incumbent_threshold=incumbent[0],
            nonbreak_budget=args.nonbreak_budget,
            calibration_incumbent=calibration_incumbent,
            heldout_frozen=heldout_frozen,
            heldout_incumbent=heldout_incumbent,
            source_run=traces.source_run,
        ),
        encoding="utf-8",
    )
    (temporary / "SUCCESS").write_text("ok\n", encoding="utf-8")


def _prepare_sid_bench(args: argparse.Namespace) -> int:
    root = prepare_sid_bench_dataset(
        args.split,
        with_audio=not args.metadata_only,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    dataset = load_sid_bench(
        root,
        args.split,
        validate_audio_headers=not args.metadata_only,
        require_audio=not args.metadata_only,
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "dataset": dataset.metadata,
                "benchmark": dataset.benchmark_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _score_sid_bench(args: argparse.Namespace) -> int:
    root = (
        args.dataset_root
        if args.dataset_root is not None
        else prepare_sid_bench_dataset(
            args.split,
            with_audio=False,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    dataset = load_sid_bench(
        root,
        args.split,
        validate_audio_headers=False,
        require_audio=False,
    )
    predictions = load_sid_bench_predictions(args.predictions)
    records = dataset.prediction_records(
        predictions,
        require_exact_coverage=not args.allow_partial,
    )
    metrics = evaluate_sid_bench(records, tolerance_s=args.tolerance_ms / 1_000)
    result = {
        "benchmark": {
            **dataset.benchmark_metadata,
            "partial_run": args.allow_partial,
        },
        "dataset": {**dataset.metadata, "evaluated_rows": len(records)},
        "predictions": str(args.predictions.resolve()),
        "metrics": metrics,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


def _analyze_sid_bench(args: argparse.Namespace) -> int:
    report = write_sid_paired_analysis(
        args.candidate_run,
        args.reference_run,
        args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(report)
    return 0


def _analyze_eot_bench(args: argparse.Namespace) -> int:
    report = write_eot_hold_analysis(
        args.run,
        args.output_dir,
        candidate_model=args.candidate_model,
        operating_point_name=args.operating_point,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turn-eval",
        description="Reproducible turn-detection model evaluation",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser(
        "run-synthetic", help="run the deterministic M0 EOT vertical slice"
    )
    synthetic.add_argument("--output-dir", type=Path, default=Path("results"))
    synthetic.add_argument("--model-timeout-ms", type=float, default=500.0)
    synthetic.add_argument("--lookahead-ms", type=float, default=0.0)
    synthetic.set_defaults(handler=_run_synthetic)

    eot_bench = commands.add_parser(
        "run-eot-bench",
        help="run the pinned LiveKit eot-bench EN/ZH data with a selected model",
    )
    eot_bench.add_argument("--lang", choices=("en", "zh"), required=True)
    eot_bench.add_argument("--revision", default=EOT_BENCH_REVISION)
    eot_bench.add_argument("--output-dir", type=Path, default=Path("results"))
    eot_bench.add_argument(
        "--model",
        choices=("fixed-timeout", "smart-turn-v3.2", "turnsense", "realtime-vap", "dualturn"),
        default="fixed-timeout",
    )
    eot_bench.add_argument("--model-revision", default=SMART_TURN_REVISION)
    eot_bench.add_argument("--turnsense-revision", default=TURNSENSE_REVISION)
    eot_bench.add_argument("--limit", type=int)
    eot_bench.add_argument("--model-timeout-ms", type=float, default=0.0)
    eot_bench.add_argument("--vap-frame-rate-hz", type=float)
    eot_bench.add_argument("--vap-context-window-s", type=float)
    eot_bench.add_argument("--lookahead-ms", type=float, default=0.0)
    eot_bench.add_argument("--inference-interval-ms", type=float, default=100.0)
    eot_bench.add_argument("--min-silence-span-ms", type=float, default=100.0)
    eot_bench.add_argument("--threads", type=int, default=1)
    eot_bench.add_argument("--device", default="cuda:0")
    eot_bench.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    eot_bench.add_argument("--progress-interval-s", type=float, default=5.0)
    eot_bench.add_argument("--quiet", action="store_true")
    eot_bench.add_argument("--local-files-only", action="store_true")
    eot_bench.set_defaults(handler=_run_eot_bench)

    report = commands.add_parser("report", help="recompute and render a completed run")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--output", type=Path)
    report.set_defaults(handler=_report)

    validate = commands.add_parser("validate-run", help="validate immutable run artifacts")
    validate.add_argument("run_dir", type=Path)
    validate.set_defaults(handler=_validate)

    turnbench_validate = commands.add_parser(
        "turnbench-validate",
        help="validate an official-format TurnBench predictions JSON",
    )
    turnbench_validate.add_argument("predictions", type=Path)
    turnbench_validate.add_argument(
        "--conversation-ids",
        type=Path,
        help="optional JSON list used for exact split-coverage validation",
    )
    turnbench_validate.add_argument(
        "--durations",
        type=Path,
        help="optional JSON object mapping conversation IDs to durations in seconds",
    )
    turnbench_validate.set_defaults(handler=_validate_turnbench_submission)

    turnbench_score = commands.add_parser(
        "turnbench-score",
        help="score TurnBench predictions using the isolated pinned official scorer",
    )
    turnbench_score.add_argument("predictions", type=Path)
    turnbench_score.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_score.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_score.add_argument(
        "--scorer-project",
        type=Path,
        default=None,
    )
    turnbench_score.add_argument("--output", type=Path)
    turnbench_score.set_defaults(handler=_score_turnbench)

    turnbench_sweep = commands.add_parser(
        "turnbench-sweep",
        help="select EOT/INT dev thresholds with the isolated pinned official scorer",
    )
    turnbench_sweep.add_argument("--eot-probabilities", type=Path, required=True)
    turnbench_sweep.add_argument("--interruption-probabilities", type=Path, required=True)
    turnbench_sweep.add_argument("--output-dir", type=Path, required=True)
    turnbench_sweep.add_argument("--fp-budget", type=float, default=0.10)
    turnbench_sweep.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_sweep.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_sweep.add_argument("--scorer-project", type=Path, default=None)
    turnbench_sweep.set_defaults(handler=_sweep_turnbench)

    turnbench_vap = commands.add_parser(
        "run-turnbench-vap",
        help="run pinned MaAI multilingual Realtime-VAP on TurnBench",
    )
    turnbench_vap.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_vap.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_vap.add_argument("--output-dir", type=Path, required=True)
    turnbench_vap.add_argument("--device", default="cpu")
    turnbench_vap.add_argument("--limit", type=int)
    turnbench_vap.add_argument("--shard-index", type=int, default=0)
    turnbench_vap.add_argument("--shard-count", type=int, default=1)
    turnbench_vap.add_argument("--local-files-only", action="store_true")
    turnbench_vap.set_defaults(handler=_run_turnbench_vap)

    turnbench_vap_gpt = commands.add_parser(
        "run-turnbench-vap-gpt",
        help="run the pinned official VapGPT OTO baseline on TurnBench",
    )
    turnbench_vap_gpt.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_vap_gpt.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_vap_gpt.add_argument("--output-dir", type=Path, required=True)
    turnbench_vap_gpt.add_argument("--device", default="cuda")
    turnbench_vap_gpt.add_argument("--limit", type=int)
    turnbench_vap_gpt.add_argument("--shard-index", type=int, default=0)
    turnbench_vap_gpt.add_argument("--shard-count", type=int, default=1)
    turnbench_vap_gpt.add_argument("--local-files-only", action="store_true")
    turnbench_vap_gpt.set_defaults(handler=_run_turnbench_vap_gpt)

    turnbench_dualturn = commands.add_parser(
        "run-turnbench-dualturn",
        help="run pinned DualTurn Endpointing with native EOT and derived FVAD interruption",
    )
    turnbench_dualturn.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_dualturn.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_dualturn.add_argument("--output-dir", type=Path, required=True)
    turnbench_dualturn.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    turnbench_dualturn.add_argument("--threads", type=int, default=4)
    turnbench_dualturn.add_argument("--limit", type=int)
    turnbench_dualturn.add_argument("--shard-index", type=int, default=0)
    turnbench_dualturn.add_argument("--shard-count", type=int, default=1)
    turnbench_dualturn.add_argument("--local-files-only", action="store_true")
    turnbench_dualturn.set_defaults(handler=_run_turnbench_dualturn)

    turnbench_espnet = commands.add_parser(
        "run-turnbench-espnet",
        help="run the pinned ESPnet TurnTaking mixed-mono baseline on TurnBench",
    )
    turnbench_espnet.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_espnet.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_espnet.add_argument("--output-dir", type=Path, required=True)
    turnbench_espnet.add_argument("--device", default="cuda")
    turnbench_espnet.add_argument("--batch-size", type=int, default=8)
    turnbench_espnet.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    turnbench_espnet.add_argument("--limit", type=int)
    turnbench_espnet.add_argument("--shard-index", type=int, default=0)
    turnbench_espnet.add_argument("--shard-count", type=int, default=1)
    turnbench_espnet.add_argument("--local-files-only", action="store_true")
    turnbench_espnet.set_defaults(handler=_run_turnbench_espnet)

    turnbench_mimi = commands.add_parser(
        "run-turnbench-mimi",
        help="run the pinned Mimi Endpointer oto_d1f baseline on TurnBench",
    )
    turnbench_mimi.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_mimi.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_mimi.add_argument("--output-dir", type=Path, required=True)
    turnbench_mimi.add_argument("--device", default="cuda")
    turnbench_mimi.add_argument("--limit", type=int)
    turnbench_mimi.add_argument("--shard-index", type=int, default=0)
    turnbench_mimi.add_argument("--shard-count", type=int, default=1)
    turnbench_mimi.add_argument("--local-files-only", action="store_true")
    turnbench_mimi.set_defaults(handler=_run_turnbench_mimi)

    turnbench_kyutai = commands.add_parser(
        "run-turnbench-kyutai",
        help="run the pinned Kyutai STT-1B semantic VAD baseline on TurnBench",
    )
    turnbench_kyutai.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_kyutai.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_kyutai.add_argument("--output-dir", type=Path, required=True)
    turnbench_kyutai.add_argument("--device", default="cuda")
    turnbench_kyutai.add_argument("--limit", type=int)
    turnbench_kyutai.add_argument("--shard-index", type=int, default=0)
    turnbench_kyutai.add_argument("--shard-count", type=int, default=1)
    turnbench_kyutai.add_argument("--local-files-only", action="store_true")
    turnbench_kyutai.set_defaults(handler=_run_turnbench_kyutai)

    turnbench_smart_turn = commands.add_parser(
        "run-turnbench-smart-turn",
        help="run pinned SmartTurn v3.1/v3.2 with the official TurnBench timeline policy",
    )
    turnbench_smart_turn.add_argument("--model-version", choices=("3.1", "3.2"), required=True)
    turnbench_smart_turn.add_argument("--artifact", choices=("cpu", "gpu"), default="gpu")
    turnbench_smart_turn.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    turnbench_smart_turn.add_argument("--workers", type=int, default=4)
    turnbench_smart_turn.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_smart_turn.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_smart_turn.add_argument("--output-dir", type=Path, required=True)
    turnbench_smart_turn.add_argument("--limit", type=int)
    turnbench_smart_turn.add_argument("--local-files-only", action="store_true")
    turnbench_smart_turn.set_defaults(handler=_run_turnbench_smart_turn)

    turnbench_rms = commands.add_parser(
        "run-turnbench-rms-vad",
        help="run the exact TurnBench RMS energy VAD reference baseline",
    )
    turnbench_rms.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    turnbench_rms.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    turnbench_rms.add_argument("--output-dir", type=Path, default=Path("results"))
    turnbench_rms.add_argument("--limit", type=int)
    turnbench_rms.add_argument("--window-ms", type=float, default=RMS_VAD_WINDOW_MS)
    turnbench_rms.add_argument("--threshold", type=float, default=RMS_VAD_THRESHOLD)
    turnbench_rms.add_argument("--scorer-project", type=Path, default=None)
    turnbench_rms.add_argument("--local-files-only", action="store_true")
    turnbench_rms.add_argument("--no-score", action="store_true")
    turnbench_rms.add_argument("--quiet", action="store_true")
    turnbench_rms.set_defaults(handler=_run_turnbench_rms_vad)

    fastturn_prepare = commands.add_parser(
        "prepare-fastturn",
        help="download, safely extract, and strictly validate the pinned FastTurn Test Set",
    )
    fastturn_prepare.add_argument("--cache-dir", type=Path)
    fastturn_prepare.add_argument("--local-files-only", action="store_true")
    fastturn_prepare.set_defaults(handler=_prepare_fastturn)

    fastturn_run = commands.add_parser(
        "run-fastturn",
        help="run a FastTurn clip-classification suite",
    )
    fastturn_run.add_argument("--suite", choices=("eot", "real", "full"), default="full")
    fastturn_run.add_argument(
        "--model",
        choices=("fixed-state", "turnsense", "smart-turn-v3.2", "easy-turn", "x2-turn"),
        default="fixed-state",
    )
    fastturn_run.add_argument(
        "--fixed-state", choices=tuple(state.value for state in TurnState), default="complete"
    )
    fastturn_run.add_argument("--limit", type=int)
    fastturn_run.add_argument("--model-revision", default=X2_TURN_REVISION)
    fastturn_run.add_argument("--model-path", default=X2_TURN_REPO_ID)
    fastturn_run.add_argument("--turnsense-revision", default=TURNSENSE_REVISION)
    fastturn_run.add_argument(
        "--smart-turn-threshold",
        type=float,
        default=SMART_TURN_FASTTURN_THRESHOLD,
    )
    fastturn_run.add_argument("--easy-turn-revision", default=EASY_TURN_REVISION)
    fastturn_run.add_argument("--easy-turn-source-root", type=Path)
    fastturn_run.add_argument("--device", default="cuda:0")
    fastturn_run.add_argument("--threads", type=int, default=4)
    fastturn_run.add_argument("--batch-size", type=int, default=1)
    fastturn_run.add_argument("--delay-ms", type=int, default=X2_TURN_DEFAULT_DELAY_MS)
    fastturn_run.add_argument("--cache-dir", type=Path)
    fastturn_run.add_argument("--output-dir", type=Path, default=Path("results"))
    fastturn_run.add_argument("--progress-interval-s", type=float, default=5.0)
    fastturn_run.add_argument("--quiet", action="store_true")
    fastturn_run.add_argument("--local-files-only", action="store_true")
    fastturn_run.set_defaults(handler=_run_fastturn)

    fastturn_rescore = commands.add_parser(
        "rescore-fastturn-eot",
        help="derive the two-state FastTurn EOT row from a completed multi-state run",
    )
    fastturn_rescore.add_argument("run", type=Path)
    fastturn_rescore.add_argument("--output-dir", type=Path, default=Path("results"))
    fastturn_rescore.set_defaults(handler=_rescore_fastturn_eot)

    sid_prepare = commands.add_parser(
        "prepare-sidbench",
        help="download and validate a pinned SID-Bench EN/ZH/noise split",
    )
    sid_prepare.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_prepare.add_argument("--metadata-only", action="store_true")
    sid_prepare.add_argument("--cache-dir", type=Path)
    sid_prepare.add_argument("--local-files-only", action="store_true")
    sid_prepare.set_defaults(handler=_prepare_sid_bench)

    sid_score = commands.add_parser(
        "score-sidbench",
        help="score official-format SID-Bench JSONL predictions",
    )
    sid_score.add_argument("predictions", type=Path)
    sid_score.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_score.add_argument("--dataset-root", type=Path)
    sid_score.add_argument("--cache-dir", type=Path)
    sid_score.add_argument("--tolerance-ms", type=float, default=50.0)
    sid_score.add_argument("--allow-partial", action="store_true")
    sid_score.add_argument("--output", type=Path)
    sid_score.add_argument("--local-files-only", action="store_true")
    sid_score.set_defaults(handler=_score_sid_bench)

    sid_analysis = commands.add_parser(
        "analyze-sidbench",
        help="compare frozen SID-Bench policies with paired bootstrap intervals",
    )
    sid_analysis.add_argument("--candidate-run", type=Path, action="append", required=True)
    sid_analysis.add_argument("--reference-run", type=Path, action="append", required=True)
    sid_analysis.add_argument("--output-dir", type=Path, required=True)
    sid_analysis.add_argument("--bootstrap-resamples", type=int, default=10_000)
    sid_analysis.add_argument("--seed", type=int, default=0)
    sid_analysis.set_defaults(handler=_analyze_sid_bench)

    eot_analysis = commands.add_parser(
        "analyze-eotbench",
        help="replay frozen EOT policies and analyze HOLD false-cutoff causes",
    )
    eot_analysis.add_argument("--run", type=Path, action="append", required=True)
    eot_analysis.add_argument("--candidate-model", required=True)
    eot_analysis.add_argument(
        "--operating-point",
        choices=(
            "false_cutoff_at_300ms",
            "false_cutoff_at_600ms",
            "latency_at_5pct_cutoff",
            "latency_at_10pct_cutoff",
        ),
        default="false_cutoff_at_300ms",
    )
    eot_analysis.add_argument("--output-dir", type=Path, required=True)
    eot_analysis.add_argument("--bootstrap-resamples", type=int, default=10_000)
    eot_analysis.add_argument("--seed", type=int, default=0)
    eot_analysis.set_defaults(handler=_analyze_eot_bench)

    sid_vap = commands.add_parser(
        "run-sidbench-vap",
        help="run pinned multilingual Realtime-VAP on a SID-Bench split",
    )
    sid_vap.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_vap.add_argument("--output-dir", type=Path, required=True)
    sid_vap.add_argument("--device", default="cuda:0")
    sid_vap.add_argument("--threshold", type=float, default=0.82)
    sid_vap.add_argument("--limit", type=int)
    sid_vap.add_argument("--local-files-only", action="store_true")
    sid_vap.add_argument("--quiet", action="store_true")
    sid_vap.set_defaults(handler=_run_sid_bench_vap)

    sid_vap_gpt = commands.add_parser(
        "run-sidbench-vap-gpt",
        help="run the pinned official VapGPT OTO native floor state on a SID-Bench split",
    )
    sid_vap_gpt.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_vap_gpt.add_argument("--output-dir", type=Path, required=True)
    sid_vap_gpt.add_argument("--device", default="cuda")
    sid_vap_gpt.add_argument("--threshold", type=float, default=0.8591171121643947)
    sid_vap_gpt.add_argument("--limit", type=int)
    sid_vap_gpt.add_argument("--local-files-only", action="store_true")
    sid_vap_gpt.add_argument("--quiet", action="store_true")
    sid_vap_gpt.set_defaults(handler=_run_sid_bench_vap_gpt)

    sid_calibrate = commands.add_parser(
        "calibrate-sidbench",
        help="freeze a two-stage interruption policy on a seeded calibration split",
    )
    sid_calibrate.add_argument("--run", type=Path, required=True)
    sid_calibrate.add_argument("--output-dir", type=Path, required=True)
    sid_calibrate.add_argument("--seed", type=int, default=0)
    sid_calibrate.add_argument("--fraction", type=float, default=0.3)
    sid_calibrate.add_argument("--nonbreak-budget", type=float, default=0.10)
    sid_calibrate.add_argument("--local-files-only", action="store_true")
    sid_calibrate.set_defaults(handler=_calibrate_sid_bench)

    sid_kyutai = commands.add_parser(
        "run-sidbench-kyutai",
        help="run pinned Kyutai Semantic VAD on a SID-Bench split as a native single stream",
    )
    sid_kyutai.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_kyutai.add_argument("--output-dir", type=Path, required=True)
    sid_kyutai.add_argument("--device", default="cuda")
    sid_kyutai.add_argument("--threshold", type=float, default=0.9735107421875)
    sid_kyutai.add_argument("--limit", type=int)
    sid_kyutai.add_argument("--local-files-only", action="store_true")
    sid_kyutai.add_argument("--quiet", action="store_true")
    sid_kyutai.set_defaults(handler=_run_sid_bench_kyutai)

    sid_dualturn = commands.add_parser(
        "run-sidbench-dualturn",
        help="run pinned DualTurn derived FVAD on a SID-Bench split",
    )
    sid_dualturn.add_argument("--split", choices=("en", "zh", "noise"), required=True)
    sid_dualturn.add_argument("--output-dir", type=Path, required=True)
    sid_dualturn.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    sid_dualturn.add_argument("--threads", type=int, default=4)
    sid_dualturn.add_argument("--threshold", type=float, default=0.9896328579889585)
    sid_dualturn.add_argument("--limit", type=int)
    sid_dualturn.add_argument("--local-files-only", action="store_true")
    sid_dualturn.add_argument("--quiet", action="store_true")
    sid_dualturn.set_defaults(handler=_run_sid_bench_dualturn)

    runtime_profile = commands.add_parser(
        "profile-runtime",
        help="profile cold start, warm ticks, RAM, VRAM, and process CPU in isolation",
    )
    runtime_profile.add_argument("--model", choices=("dualturn", "realtime-vap"), required=True)
    runtime_profile.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    runtime_profile.add_argument("--output", type=Path, required=True)
    runtime_profile.add_argument("--threads", type=int, default=1)
    runtime_profile.add_argument("--warmup-frames", type=int, default=64)
    runtime_profile.add_argument("--measurement-frames", type=int, default=256)
    runtime_profile.add_argument("--sampling-interval-ms", type=float, default=100)
    runtime_profile.add_argument("--seed", type=int, default=0)
    runtime_profile.add_argument("--local-files-only", action="store_true")
    runtime_profile.set_defaults(handler=_profile_runtime)

    runtime_report = commands.add_parser(
        "report-runtime",
        help="validate raw runtime traces and render a multi-profile comparison",
    )
    runtime_report.add_argument("profiles", type=Path, nargs="+")
    runtime_report.add_argument("--output-dir", type=Path, required=True)
    runtime_report.set_defaults(handler=_report_runtime)

    suite_report = commands.add_parser(
        "report-suite",
        help="discover eligible benchmark runs and render a provenance-checked comparison",
    )
    suite_report.add_argument("--lang", choices=("en", "zh", "all"), default="all")
    suite_report.add_argument("--results-root", type=Path, default=Path("results"))
    suite_report.add_argument("--runtime-summary", type=Path)
    suite_report.add_argument("--output-dir", type=Path, required=True)
    suite_report.set_defaults(handler=_report_suite)

    suite_plots = commands.add_parser(
        "plot-suite",
        help="render provenance-checked EOT Pareto and SID frozen-policy figures",
    )
    suite_plots.add_argument("--comparison-summary", type=Path, required=True)
    suite_plots.add_argument("--output-dir", type=Path, required=True)
    suite_plots.set_defaults(handler=_plot_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.exit(2, f"turn-eval: error: {error}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
