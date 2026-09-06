from __future__ import annotations

import pytest

from turn_eval.benchmarks.synthetic import synthetic_decision_point_examples
from turn_eval.core.types import (
    ChannelMode,
    EvaluationMode,
    ExecutionMode,
    ServiceMode,
    SignalProvenance,
    StatePredictionRecord,
    SupportStatus,
    TranscriptMode,
    TurnState,
)
from turn_eval.metrics.eot import PolicyGrid, default_policy_grid, evaluate_eot_sweep
from turn_eval.metrics.state import eot_subset, evaluate_turn_states
from turn_eval.models.fixed_timeout import FixedTimeoutAdapter
from turn_eval.runner.decision_point import run_decision_points


def _mode() -> EvaluationMode:
    return EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        TranscriptMode.NONE,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )


def test_eot_sweep_produces_four_operating_points_and_frontier() -> None:
    model = FixedTimeoutAdapter(timeout_ms=500)
    run = run_decision_points(synthetic_decision_point_examples(), model, _mode())
    metrics = evaluate_eot_sweep(run.predictions, default_policy_grid())

    assert metrics["counts"] == {
        "prediction_rows": len(run.predictions),
        "spans": 5,
        "hold_spans": 2,
        "eot_spans": 3,
    }
    assert set(metrics["operating_points"]) == {
        "false_cutoff_at_300ms",
        "false_cutoff_at_600ms",
        "latency_at_5pct_cutoff",
        "latency_at_10pct_cutoff",
    }
    assert metrics["pareto_frontier"]
    assert metrics["diagnostics"]["auc"] == pytest.approx(0.75)
    assert metrics["diagnostics"]["average_precision"] == pytest.approx(0.75)


def test_policy_grid_rejects_test_set_tuning() -> None:
    with pytest.raises(ValueError, match="test split"):
        PolicyGrid((0.5,), (100.0,), (600.0,), "test", split_role="test")


def test_score_point_prediction_can_commit_after_a_later_action_delay() -> None:
    model = FixedTimeoutAdapter(timeout_ms=100)
    run = run_decision_points(
        synthetic_decision_point_examples(),
        model,
        _mode(),
        score_point_s=0.2,
    )
    policy = PolicyGrid(
        thresholds=(0.5,),
        action_delays_ms=(600.0,),
        timeouts_ms=(1_000.0,),
        calibration_source="fixture",
        score_point_ms=200.0,
    )

    point = evaluate_eot_sweep(run.predictions, policy)["policy_sweep"][0]

    assert point["mean_latency_ms"] == pytest.approx(600.0)
    assert point["false_cutoff_rate"] == pytest.approx(0.5)


def test_cluster_bootstrap_is_deterministic() -> None:
    model = FixedTimeoutAdapter(timeout_ms=500)
    run = run_decision_points(synthetic_decision_point_examples(), model, _mode())
    policy = PolicyGrid(
        thresholds=(0.5,),
        action_delays_ms=(300.0, 600.0),
        timeouts_ms=(1_000.0,),
        calibration_source="fixture",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    first = evaluate_eot_sweep(run.predictions, policy)["confidence_intervals"]
    second = evaluate_eot_sweep(run.predictions, policy)["confidence_intervals"]

    assert first == second
    assert first["groups"] == 3
    interval = first["intervals"]["false_cutoff_at_600ms"]
    assert 0 <= interval["lower"] <= interval["upper"] <= 1


def _state_record(
    index: int, label: TurnState, predicted: TurnState, latency_ms: float
) -> StatePredictionRecord:
    return StatePredictionRecord(
        example_id=f"example-{index}",
        group_id=f"group-{index // 2}",
        language="zh",
        label=label,
        predicted_label=predicted,
        scores={predicted.value: 1.0},
        inference_latency_ms=latency_ms,
    )


def test_turn_state_metrics_include_confusion_and_critical_errors() -> None:
    labels = (
        TurnState.COMPLETE,
        TurnState.INCOMPLETE,
        TurnState.BACKCHANNEL,
        TurnState.WAIT,
    )
    records = (
        _state_record(0, TurnState.COMPLETE, TurnState.COMPLETE, 10),
        _state_record(1, TurnState.INCOMPLETE, TurnState.COMPLETE, 20),
        _state_record(2, TurnState.BACKCHANNEL, TurnState.COMPLETE, 30),
        _state_record(3, TurnState.BACKCHANNEL, TurnState.BACKCHANNEL, 40),
        _state_record(4, TurnState.WAIT, TurnState.WAIT, 50),
    )

    metrics = evaluate_turn_states(records, labels)

    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    assert metrics["prediction_labels"] == [
        "complete",
        "incomplete",
        "backchannel",
        "wait",
        "abstain",
    ]
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0 + 2 / 3 + 1) / 4)
    assert metrics["weighted_f1"] == pytest.approx((0.5 + 0 + 2 * (2 / 3) + 1) / 5)
    assert metrics["balanced_accuracy"] == pytest.approx((1 + 0 + 0.5 + 1) / 4)
    assert metrics["per_class"]["incomplete"]["miss_rate"] == 1.0
    assert metrics["per_class"]["complete"]["false_alarm_rate"] == pytest.approx(0.5)
    assert metrics["critical_errors"]["incomplete_to_complete"]["rate"] == 1.0
    assert metrics["critical_errors"]["backchannel_to_complete"]["rate"] == 0.5
    assert metrics["inference_latency_ms"] == {"p50": 30.0, "p90": 46.0, "p95": 48.0}


def test_turn_state_abstention_counts_as_closed_set_error() -> None:
    record = StatePredictionRecord(
        example_id="abstain",
        group_id="group",
        language="zh",
        label=TurnState.COMPLETE,
        predicted_label=None,
        scores={"complete": 0.1, "incomplete": 0.1},
        inference_latency_ms=1.0,
    )

    metrics = evaluate_turn_states((record,), (TurnState.COMPLETE, TurnState.INCOMPLETE))

    assert metrics["accuracy"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 0, 1], [0, 0, 0]]
    assert metrics["abstentions"] == {"count": 1, "rate": 1.0}


def test_eot_subset_does_not_remap_other_states() -> None:
    records = (
        _state_record(0, TurnState.COMPLETE, TurnState.COMPLETE, 1),
        _state_record(1, TurnState.BACKCHANNEL, TurnState.BACKCHANNEL, 1),
    )

    assert [row.label for row in eot_subset(records)] == [TurnState.COMPLETE]
