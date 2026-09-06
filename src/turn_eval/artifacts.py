from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from turn_eval.core.types import (
    EvaluationMode,
    ModelDescriptor,
    PredictionRecord,
    StatePredictionRecord,
)
from turn_eval.metrics.eot import PolicyGrid
from turn_eval.runner.clip_classification import ClipClassificationRun
from turn_eval.runner.decision_point import DecisionPointRun

ARTIFACT_SCHEMA_VERSION = 3
REQUIRED_RUN_FILES = frozenset(
    {
        "manifest.json",
        "predictions.parquet",
        "metrics.json",
        "runtime.json",
        "SUCCESS",
    }
)
EOT_RUN_FILES = frozenset({"policy.json", "summary.csv", "pareto.parquet"})
LEGACY_EOT_RUN_FILES = frozenset({"policy.json"})
STATE_RUN_FILES = frozenset({"label_mapping.json", "summary.csv", "confusion_matrix.csv"})


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    run_id: str,
    model: ModelDescriptor,
    mode: EvaluationMode,
    policy: PolicyGrid,
    model_options: Mapping[str, object] | None = None,
    dataset_metadata: Mapping[str, object] | None = None,
    benchmark_metadata: Mapping[str, object] | None = None,
    execution_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    commit = _git_value("rev-parse", "HEAD")
    dirty = bool(_git_value("status", "--porcelain"))
    dataset = (
        dict(dataset_metadata)
        if dataset_metadata is not None
        else {
            "repository": "builtin",
            "configuration": "synthetic-eot-m0",
            "split": "development",
            "revision": "v1",
            "checksum": content_hash(
                {
                    "examples": ["short-hold", "long-hold", "no-hold"],
                    "sample_rate": 1_000,
                    "revision": "v1",
                }
            ),
            "license": "CC0-1.0",
        }
    )
    benchmark = (
        dict(benchmark_metadata)
        if benchmark_metadata is not None
        else {
            "name": "synthetic-eot-m0",
            "revision": "v1",
            "official_scorer_revision": "builtin-eot-v1",
            "protocol": "decision_point",
            "task": "eot_hold",
            "language": "en",
            "locale": "en-US",
            "dataset_license": "CC0-1.0",
            "model_team_authored_benchmark": False,
            "possible_test_overlap": "none-synthetic-fixture",
        }
    )
    execution = {
        "device": "cpu",
        "thread_count": 1,
        "batch_size": 1,
        "seed": 0,
    }
    execution.update(execution_metadata or {})
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "name": model.name,
            "revision": model.revision,
            "artifact_checksum": model.artifact_checksum,
            "artifact_size_bytes": model.artifact_size_bytes,
            "native_sample_rate": model.native_sample_rate,
            "frame_interval_ms": model.frame_interval_ms,
            "context_window_s": model.context_window_s,
            "algorithmic_lookahead_ms": model.algorithmic_lookahead_ms,
            "natural_score_point_ms": model.natural_score_point_ms,
            "model_license": model.model_license,
            "code_license": model.code_license,
            "known_training_datasets": list(model.known_training_datasets),
            "allow_unverified_zero_shot": model.allow_unverified_zero_shot,
            "accepted_inputs": sorted(model.accepted_inputs),
            "output_signals": sorted(model.output_signals),
            "options": dict(model_options or {}),
        },
        "dataset": dataset,
        "benchmark": benchmark,
        "evaluation_mode": {key: value.value for key, value in asdict(mode).items()},
        "calibration": {
            "source": policy.calibration_source,
            "split_role": policy.split_role,
            "policy_hash": content_hash(policy.to_dict()),
        },
        "source": {"git_commit": commit, "dirty_worktree": dirty},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "huggingface_hub": _package_version("huggingface-hub"),
            "soundfile": _package_version("soundfile"),
            "onnxruntime": _package_version("onnxruntime"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "voxtral_realtime": _package_version("voxtral-realtime"),
            "hf_home": os.environ.get("HF_HOME"),
        },
        "execution": execution,
    }


def build_state_manifest(
    *,
    run_id: str,
    model: ModelDescriptor,
    mode: EvaluationMode,
    label_mapping: Mapping[str, object],
    model_options: Mapping[str, object] | None = None,
    dataset_metadata: Mapping[str, object] | None = None,
    benchmark_metadata: Mapping[str, object] | None = None,
    execution_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a standard manifest whose calibration artifact is a state mapping."""
    placeholder = PolicyGrid(
        thresholds=(0.5,),
        action_delays_ms=(0.0,),
        timeouts_ms=(1.0,),
        calibration_source="internal-state-manifest-placeholder",
    )
    manifest = build_manifest(
        run_id=run_id,
        model=model,
        mode=mode,
        policy=placeholder,
        model_options=model_options,
        dataset_metadata=dataset_metadata,
        benchmark_metadata=benchmark_metadata,
        execution_metadata=execution_metadata,
    )
    manifest["calibration"] = {
        "source": label_mapping.get("calibration_source", "none-native-model"),
        "split_role": label_mapping.get("split_role", "none"),
        "mapping_hash": content_hash(label_mapping),
    }
    return manifest


def runtime_metrics(run: DecisionPointRun | ClipClassificationRun) -> dict[str, object]:
    latencies = np.asarray(
        [record.inference_latency_ms for record in run.predictions], dtype=np.float64
    )

    def percentile(q: float) -> float | None:
        return float(np.quantile(latencies, q)) if len(latencies) else None

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "initialization_time_ms": run.initialization_time_ms,
        "inference_wall_s": run.inference_wall_s,
        "resampling_wall_s": run.resampling_wall_s,
        "audio_s": run.audio_s,
        "rtf": run.rtf,
        "realtime_multiplier": run.realtime_multiplier,
        "inference_latency_ms": {
            "count": len(latencies),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
        },
        "device": "cpu",
        "thread_count": 1,
        "batch_size": 1,
        "ram_peak_bytes": None,
        "gpu_vram_peak_bytes": None,
        "model_artifact_size_bytes": None,
    }


def _prediction_rows(records: Iterable[PredictionRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = asdict(record)
        row["label"] = record.label.value
        row["metadata_json"] = canonical_json(row.pop("metadata"))
        rows.append(row)
    return rows


def _state_prediction_rows(records: Iterable[StatePredictionRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = asdict(record)
        row["label"] = record.label.value
        row["predicted_label"] = (
            None if record.predicted_label is None else record.predicted_label.value
        )
        row["scores_json"] = canonical_json(row.pop("scores"))
        row["metadata_json"] = canonical_json(row.pop("metadata"))
        row["model_output_json"] = canonical_json(row.pop("model_output"))
        rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _operating_point_value(point: object, key: str) -> float | None:
    return float(point[key]) if isinstance(point, dict) else None


def _write_metric_tables(path: Path, metrics: Mapping[str, object]) -> None:
    points = metrics["operating_points"]
    if not isinstance(points, dict):
        raise ValueError("metrics operating_points must be a mapping")
    summary: dict[str, object] = {
        "false_cutoff_at_300ms": _operating_point_value(
            points.get("false_cutoff_at_300ms"), "false_cutoff_rate"
        ),
        "false_cutoff_at_600ms": _operating_point_value(
            points.get("false_cutoff_at_600ms"), "false_cutoff_rate"
        ),
        "latency_at_5pct_cutoff_ms": _operating_point_value(
            points.get("latency_at_5pct_cutoff"), "mean_latency_ms"
        ),
        "latency_at_10pct_cutoff_ms": _operating_point_value(
            points.get("latency_at_10pct_cutoff"), "mean_latency_ms"
        ),
    }
    diagnostics = metrics.get("diagnostics")
    if isinstance(diagnostics, dict):
        summary["auc"] = diagnostics.get("auc")
        summary["average_precision"] = diagnostics.get("average_precision")
    confidence = metrics.get("confidence_intervals")
    if isinstance(confidence, dict) and isinstance(confidence.get("intervals"), dict):
        for name, interval in confidence["intervals"].items():
            if isinstance(interval, dict):
                output_name = f"{name}_ms" if name.startswith("latency_") else name
                summary[f"{output_name}_ci_lower"] = interval.get("lower")
                summary[f"{output_name}_ci_upper"] = interval.get("upper")

    with path.joinpath("summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    frontier = metrics.get("pareto_frontier")
    if not isinstance(frontier, list) or not frontier:
        raise ValueError("metrics Pareto frontier must not be empty")
    pq.write_table(pa.Table.from_pylist(frontier), path / "pareto.parquet", compression="zstd")


def _write_state_metric_tables(path: Path, metrics: Mapping[str, object]) -> None:
    labels = metrics.get("labels")
    prediction_labels = metrics.get("prediction_labels", labels)
    matrix = metrics.get("confusion_matrix")
    per_class = metrics.get("per_class")
    if (
        not isinstance(labels, list)
        or not isinstance(prediction_labels, list)
        or not isinstance(matrix, list)
        or not isinstance(per_class, dict)
    ):
        raise ValueError("state metrics are missing labels, confusion matrix, or per-class values")
    summary: dict[str, object] = {
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
    }
    for label in labels:
        values = per_class[label]
        for key in (
            "precision",
            "recall",
            "f1",
            "support",
            "one_vs_rest_accuracy",
            "miss_rate",
            "false_alarm_rate",
        ):
            summary[f"{label}_{key}"] = values[key]
    with path.joinpath("summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with path.joinpath("confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gold\\predicted", *prediction_labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def make_run_id(
    model: ModelDescriptor,
    policy: PolicyGrid,
    model_options: Mapping[str, object] | None = None,
) -> str:
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = content_hash(
        {
            "model": model.name,
            "revision": model.revision,
            "lookahead_ms": model.algorithmic_lookahead_ms,
            "model_options": dict(model_options or {}),
            "policy": policy.to_dict(),
        }
    )[:8]
    return f"{created}-{suffix}"


def make_state_run_id(
    model: ModelDescriptor,
    label_mapping: Mapping[str, object],
    model_options: Mapping[str, object] | None = None,
) -> str:
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = content_hash(
        {
            "model": model.name,
            "revision": model.revision,
            "model_options": dict(model_options or {}),
            "label_mapping": dict(label_mapping),
        }
    )[:8]
    return f"{created}-{suffix}"


def write_run_artifacts(
    output_root: Path,
    *,
    run_id: str,
    manifest: Mapping[str, object],
    run: DecisionPointRun,
    policy: PolicyGrid,
    metrics: Mapping[str, object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"immutable run already exists: {destination}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=output_root))
    try:
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "policy.json", policy.to_dict())
        _write_json(temporary / "metrics.json", metrics)
        runtime = runtime_metrics(run)
        runtime["model_artifact_size_bytes"] = manifest.get("model", {}).get("artifact_size_bytes")
        execution = manifest.get("execution", {})
        runtime["device"] = execution.get("device")
        runtime["thread_count"] = execution.get("thread_count")
        runtime["batch_size"] = execution.get("batch_size")
        _write_json(temporary / "runtime.json", runtime)
        _write_metric_tables(temporary, metrics)
        table = pa.Table.from_pylist(_prediction_rows(run.predictions))
        pq.write_table(table, temporary / "predictions.parquet", compression="zstd")
        (temporary / "SUCCESS").write_text("ok\n")
        validate_run_artifacts(temporary)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def write_state_run_artifacts(
    output_root: Path,
    *,
    run_id: str,
    manifest: Mapping[str, object],
    run: ClipClassificationRun,
    label_mapping: Mapping[str, object],
    metrics: Mapping[str, object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"immutable run already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=output_root))
    try:
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "label_mapping.json", label_mapping)
        _write_json(temporary / "metrics.json", metrics)
        runtime = runtime_metrics(run)
        runtime["model_artifact_size_bytes"] = manifest.get("model", {}).get("artifact_size_bytes")
        execution = manifest.get("execution", {})
        runtime["device"] = execution.get("device")
        runtime["thread_count"] = execution.get("thread_count")
        runtime["batch_size"] = execution.get("batch_size")
        _write_json(temporary / "runtime.json", runtime)
        _write_state_metric_tables(temporary, metrics)
        table = pa.Table.from_pylist(_state_prediction_rows(run.predictions))
        pq.write_table(table, temporary / "predictions.parquet", compression="zstd")
        (temporary / "SUCCESS").write_text("ok\n")
        validate_run_artifacts(temporary)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def validate_run_artifacts(run_dir: Path) -> None:
    present = {path.name for path in run_dir.iterdir()}
    missing = REQUIRED_RUN_FILES - present
    if missing:
        raise ValueError(f"run is incomplete; missing: {', '.join(sorted(missing))}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2, ARTIFACT_SCHEMA_VERSION}:
        raise ValueError("unsupported manifest schema version")
    protocol = manifest.get("benchmark", {}).get("protocol")
    protocol_files = (
        STATE_RUN_FILES
        if protocol == "clip_classification"
        else EOT_RUN_FILES
        if schema_version >= 2
        else LEGACY_EOT_RUN_FILES
    )
    if missing_protocol := protocol_files - present:
        raise ValueError(f"run is incomplete; missing: {', '.join(sorted(missing_protocol))}")
    if protocol == "clip_classification":
        mapping = json.loads((run_dir / "label_mapping.json").read_text())
        if content_hash(mapping) != manifest["calibration"]["mapping_hash"]:
            raise ValueError("label mapping hash does not match manifest")
    else:
        policy = json.loads((run_dir / "policy.json").read_text())
        if content_hash(policy) != manifest["calibration"]["policy_hash"]:
            raise ValueError("policy hash does not match manifest")
    table = pq.read_table(run_dir / "predictions.parquet")
    required_columns = (
        {"example_id", "label", "predicted_label", "scores_json", "inference_latency_ms"}
        if protocol == "clip_classification"
        else {
            "example_id",
            "decision_point_id",
            "span_id",
            "label",
            "observed_until_s",
            "available_at_s",
            "p_eot",
        }
    )
    if missing_columns := required_columns - set(table.column_names):
        raise ValueError(f"prediction columns missing: {sorted(missing_columns)}")
    if table.num_rows == 0:
        raise ValueError("prediction artifact must not be empty")
