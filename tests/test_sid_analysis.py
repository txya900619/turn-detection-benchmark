from __future__ import annotations

import pytest

from turn_eval.reports.sid_analysis import analyze_sid_pair


def _row(
    example_id: str,
    category: str,
    gold: float | None,
    predicted: float | None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "category": category,
        "annotated_duration_s": 2.0,
        "gold_break_time_s": gold,
        "predicted_break_time_s": predicted,
    }


def test_sid_pair_reports_paired_fixed_policy_deltas() -> None:
    candidate = [
        _row("a", "interruption_middle", 1.0, 1.1),
        _row("b", "interruption_middle", 1.0, None),
        _row("c", "nonbreak", None, None),
        _row("d", "nonbreak", None, 0.2),
    ]
    reference = [
        _row("a", "interruption_middle", 1.0, 0.2),
        _row("b", "interruption_middle", 1.0, 1.2),
        _row("c", "nonbreak", None, 0.2),
        _row("d", "nonbreak", None, 0.2),
    ]

    result = analyze_sid_pair(candidate, reference, bootstrap_resamples=100, seed=3)

    assert result["overall"]["interruption_recall"]["candidate"] == 0.5
    assert result["overall"]["interruption_recall"]["reference"] == 0.5
    assert result["overall"]["strict_fir"]["candidate"] == 0.25
    assert result["overall"]["strict_fir"]["reference"] == 0.75
    nonbreak = result["by_category"]["nonbreak"]
    assert nonbreak["paired_counts"] == {
        "both_positive": 1,
        "candidate_only": 0,
        "reference_only": 1,
        "neither": 0,
    }


def test_sid_pair_rejects_mismatched_coverage() -> None:
    with pytest.raises(ValueError, match="identical non-empty"):
        analyze_sid_pair(
            [_row("a", "nonbreak", None, None)],
            [_row("b", "nonbreak", None, None)],
            bootstrap_resamples=10,
        )
