from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from turn_eval.artifacts import (
    build_manifest,
    build_state_manifest,
    make_run_id,
    make_state_run_id,
    validate_run_artifacts,
    write_run_artifacts,
    write_state_run_artifacts,
)
from turn_eval.benchmarks.synthetic import synthetic_decision_point_examples
from turn_eval.core.types import (
    AudioData,
    ChannelMode,
    ClipClassificationExample,
    ClipRequest,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    StatePrediction,
    SupportStatus,
    TranscriptMode,
    TurnState,
)
from turn_eval.metrics.eot import default_policy_grid, evaluate_eot_sweep
from turn_eval.metrics.state import evaluate_turn_states
from turn_eval.models.fixed_timeout import FixedTimeoutAdapter
from turn_eval.reports.summary import recompute_run_metrics
from turn_eval.runner.clip_classification import run_clip_classification
from turn_eval.runner.decision_point import run_decision_points


def _completed_run(tmp_path: Path) -> Path:
    model = FixedTimeoutAdapter()
    mode = EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        TranscriptMode.NONE,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )
    policy = default_policy_grid()
    run = run_decision_points(synthetic_decision_point_examples(), model, mode)
    metrics = evaluate_eot_sweep(run.predictions, policy)
    run_id = make_run_id(model.descriptor, policy)
    manifest = build_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        policy=policy,
        model_options={"timeout_ms": 500},
    )
    return write_run_artifacts(
        tmp_path,
        run_id=run_id,
        manifest=manifest,
        run=run,
        policy=policy,
        metrics=metrics,
    )


def test_artifacts_validate_and_metrics_recompute_without_model(tmp_path: Path) -> None:
    run_dir = _completed_run(tmp_path)
    validate_run_artifacts(run_dir)
    stored = json.loads((run_dir / "metrics.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert recompute_run_metrics(run_dir) == stored
    assert manifest["dataset"]["checksum"]
    assert manifest["benchmark"]["official_scorer_revision"] == "builtin-eot-v1"
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "pareto.parquet").is_file()


def test_run_directories_are_immutable(tmp_path: Path) -> None:
    run_dir = _completed_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    model = FixedTimeoutAdapter()
    mode = EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        TranscriptMode.NONE,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )
    policy = default_policy_grid()
    run = run_decision_points(synthetic_decision_point_examples(), model, mode)
    metrics = evaluate_eot_sweep(run.predictions, policy)
    with pytest.raises(FileExistsError, match="immutable"):
        write_run_artifacts(
            tmp_path,
            run_id=run_dir.name,
            manifest=manifest,
            run=run,
            policy=policy,
            metrics=metrics,
        )
    assert manifest["calibration"]["split_role"] == "development"


def test_policy_tampering_is_detected(tmp_path: Path) -> None:
    run_dir = _completed_run(tmp_path)
    policy_path = run_dir / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["thresholds"] = [0.1]
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="policy hash"):
        validate_run_artifacts(run_dir)


def test_state_artifacts_validate_and_recompute_without_model(tmp_path: Path) -> None:
    class Adapter:
        descriptor = ModelDescriptor(
            "state-fixture",
            "v1",
            16_000,
            frozenset({ExecutionMode.OFFLINE}),
            frozenset({"turn_state"}),
            frozenset({"zh"}),
        )

        def predict_clip(self, request: ClipRequest) -> StatePrediction:
            return StatePrediction(request.example_id, TurnState.COMPLETE, {"complete": 1.0})

    examples = (
        ClipClassificationExample(
            "complete",
            "zh",
            AudioData(np.zeros(1_600, dtype=np.float32), 16_000),
            TurnState.COMPLETE,
        ),
        ClipClassificationExample(
            "incomplete",
            "zh",
            AudioData(np.zeros(1_600, dtype=np.float32), 16_000),
            TurnState.INCOMPLETE,
        ),
    )
    mode = EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.OFFLINE,
        SignalProvenance.NATIVE,
        TranscriptMode.NONE,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )
    mapping = {
        "strategy": "native_argmax",
        "output_labels": ["complete", "incomplete"],
        "calibration_source": "none-native-model",
        "split_role": "none",
    }
    model = Adapter()
    run = run_clip_classification(examples, model, mode)
    metrics = evaluate_turn_states(run.predictions, (TurnState.COMPLETE, TurnState.INCOMPLETE))
    run_id = make_state_run_id(model.descriptor, mapping)
    manifest = build_state_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=mode,
        label_mapping=mapping,
        benchmark_metadata={
            "name": "state-fixture",
            "protocol": "clip_classification",
            "language": "zh",
        },
    )
    run_dir = write_state_run_artifacts(
        tmp_path,
        run_id=run_id,
        manifest=manifest,
        run=run,
        label_mapping=mapping,
        metrics=metrics,
    )

    validate_run_artifacts(run_dir)
    assert recompute_run_metrics(run_dir) == metrics
    assert (run_dir / "confusion_matrix.csv").is_file()
    mapping_path = run_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps({"output_labels": ["complete"]}))
    with pytest.raises(ValueError, match="mapping hash"):
        validate_run_artifacts(run_dir)
