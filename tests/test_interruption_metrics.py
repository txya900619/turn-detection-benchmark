from __future__ import annotations

import pytest

from turn_eval.core.types import InterruptionPredictionRecord
from turn_eval.metrics.interruption import evaluate_sid_bench


def _record(
    index: int,
    *,
    duration: float,
    gold: float | None,
    predicted: float | None,
    category: str,
) -> InterruptionPredictionRecord:
    return InterruptionPredictionRecord(
        example_id=f"clip-{index}",
        group_id=f"clip-{index}",
        language="en",
        category=category,
        duration_s=duration,
        gold_break_time_s=gold,
        predicted_break_time_s=predicted,
    )


def test_sid_bench_metrics_match_official_case_table_and_notebook_semantics() -> None:
    records = (
        _record(0, duration=2, gold=None, predicted=None, category="nonbreak"),
        _record(1, duration=2, gold=None, predicted=0.5, category="nonbreak"),
        _record(2, duration=3, gold=1, predicted=None, category="interruption_middle"),
        _record(3, duration=3, gold=1, predicted=0.5, category="interruption_middle"),
        _record(4, duration=3, gold=1, predicted=1.04, category="interruption_beginning"),
        _record(5, duration=3, gold=1, predicted=1.2, category="interruption_beginning"),
    )

    metrics = evaluate_sid_bench(records)

    assert metrics["false_interruption_rate"] == pytest.approx(2 / 6)
    assert metrics["official_mean_irl_s"] == pytest.approx(0.2)
    assert metrics["average_penalty_time_s"] == pytest.approx(7.2 / 6)
    assert metrics["interruption_recall"] == pytest.approx(0.5)
    assert metrics["response_latency_s"] == pytest.approx({"p50": 0.1, "p90": 0.18, "p95": 0.19})
    assert metrics["diagnostics"]["nonbreak_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["diagnostics"]["early_interruption_rate"] == pytest.approx(0.25)
    assert metrics["diagnostics"]["strict_paper_fir"] == pytest.approx(2 / 6)
    assert metrics["diagnostics"]["notebook_end_boundary_exemptions"] == 0
    assert metrics["counts"]["official_irl_samples"] == 1
    assert metrics["counts"]["outcomes"] == {
        "early_interruption": 1,
        "false_interruption_nonbreak": 1,
        "late_interruption": 1,
        "missed_interruption": 1,
        "true_negative": 1,
        "within_tolerance": 1,
    }


def test_sid_bench_metrics_expose_undefined_irl_when_there_is_no_late_detection() -> None:
    metrics = evaluate_sid_bench(
        (_record(0, duration=1, gold=None, predicted=None, category="nonbreak"),)
    )

    assert metrics["official_mean_irl_s"] is None
    assert metrics["interruption_recall"] is None
    assert metrics["response_latency_s"] == {"p50": None, "p90": None, "p95": None}


def test_sid_bench_preserves_notebook_nonbreak_end_boundary_exemption() -> None:
    metrics = evaluate_sid_bench(
        (_record(0, duration=1, gold=None, predicted=0.98, category="nonbreak"),)
    )

    assert metrics["false_interruption_rate"] == 0.0
    assert metrics["average_penalty_time_s"] == 0.0
    assert metrics["diagnostics"]["strict_paper_fir"] == 1.0
    assert metrics["diagnostics"]["nonbreak_false_positive_rate"] == 1.0
    assert metrics["diagnostics"]["notebook_end_boundary_exemptions"] == 1
